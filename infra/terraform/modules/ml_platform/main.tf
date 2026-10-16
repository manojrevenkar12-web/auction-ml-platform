# Shared, per-environment ML platform resources: encryption, buckets, the
# Glue database behind the offline feature store, the serving image repository,
# the SageMaker execution role and the alert topic.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  prefix     = "${var.name_prefix}-${var.env}"
  account_id = data.aws_caller_identity.current.account_id
  buckets    = toset(["artifacts", "offline-store", "data-capture"])
}

data "aws_iam_policy_document" "kms" {
  statement {
    sid       = "AccountAdmin"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:${data.aws_partition.current.partition}:iam::${local.account_id}:root"]
    }
  }
  statement {
    sid       = "AlarmsPublishToEncryptedTopic"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey*"]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }
  }
}

resource "aws_kms_key" "ml" {
  description             = "${local.prefix} ML artifacts, features, data capture"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy                  = data.aws_iam_policy_document.kms.json
  tags                    = var.tags
}

resource "aws_kms_alias" "ml" {
  name          = "alias/${local.prefix}"
  target_key_id = aws_kms_key.ml.key_id
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = "${local.prefix}-${each.key}-${local.account_id}"
  tags     = var.tags
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.ml.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "capture" {
  bucket = aws_s3_bucket.this["data-capture"].id
  rule {
    id     = "expire-capture"
    status = "Enabled"
    filter {}
    expiration {
      days = var.capture_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

resource "aws_glue_catalog_database" "offline" {
  name = replace("${local.prefix}_feature_store", "-", "_")
}

resource "aws_ecr_repository" "serving" {
  name                 = "${local.prefix}-serving"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ml.arn
  }
  tags = var.tags
}

data "aws_iam_policy_document" "assume_sagemaker" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "ReadModelArtifacts"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.this["artifacts"].arn, "${aws_s3_bucket.this["artifacts"].arn}/*"]
  }
  statement {
    sid       = "WriteCaptureAndOfflineStore"
    actions   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:GetBucketAcl", "s3:PutObjectAcl"]
    resources = flatten([for b in ["data-capture", "offline-store"] : [aws_s3_bucket.this[b].arn, "${aws_s3_bucket.this[b].arn}/*"]])
  }
  statement {
    sid       = "UseKey"
    actions   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey*", "kms:DescribeKey", "kms:CreateGrant"]
    resources = [aws_kms_key.ml.arn]
  }
  statement {
    sid       = "PullServingImage"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.serving.arn]
  }
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "OnlineFeatureReads"
    actions   = ["sagemaker:GetRecord", "sagemaker:BatchGetRecord", "sagemaker:PutRecord"]
    resources = ["arn:${data.aws_partition.current.partition}:sagemaker:*:${local.account_id}:feature-group/${local.prefix}-*"]
  }
  statement {
    sid       = "OfflineStoreCatalog"
    actions   = ["glue:GetTable", "glue:CreateTable", "glue:UpdateTable", "glue:GetDatabase", "glue:BatchCreatePartition"]
    resources = ["*"]
  }
  statement {
    sid       = "Telemetry"
    actions   = ["cloudwatch:PutMetricData", "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["*"]
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${local.prefix}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.assume_sagemaker.json
  tags               = var.tags
}

resource "aws_iam_role_policy" "sagemaker" {
  name   = "${local.prefix}-sagemaker-execution"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.execution.json
}

resource "aws_sns_topic" "alerts" {
  name              = "${local.prefix}-model-alerts"
  kms_master_key_id = aws_kms_key.ml.id
  tags              = var.tags
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == null ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
