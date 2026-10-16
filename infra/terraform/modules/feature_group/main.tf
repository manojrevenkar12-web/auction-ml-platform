# One SageMaker Feature Store feature group per batch group in feature_repo/catalog.yaml.
# Online store for serving lookups, offline store (S3 + Glue) for point-in-time
# training joins. entity_key / event_time / known_at mirror the local store.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60"
    }
  }
}

variable "prefix" {
  type = string
}

variable "group" {
  description = "Feature group name as in the catalog (underscores allowed)."
  type        = string
}

variable "features" {
  description = "feature name -> catalog type (numeric | categorical)"
  type        = map(string)
}

variable "role_arn" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "offline_bucket" {
  type = string
}

variable "glue_database" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}

locals {
  name = substr("${var.prefix}-${replace(var.group, "_", "-")}", 0, 64)
  type = { numeric = "Fractional", categorical = "String" }
}

resource "aws_sagemaker_feature_group" "this" {
  feature_group_name             = local.name
  record_identifier_feature_name = "entity_key"
  event_time_feature_name        = "event_time"
  role_arn                       = var.role_arn
  description                    = "aucml batch feature group ${var.group}"

  feature_definition {
    feature_name = "entity_key"
    feature_type = "String"
  }
  feature_definition {
    feature_name = "event_time"
    feature_type = "String"
  }
  feature_definition {
    feature_name = "known_at"
    feature_type = "String"
  }
  dynamic "feature_definition" {
    for_each = var.features
    content {
      feature_name = feature_definition.key
      feature_type = local.type[feature_definition.value]
    }
  }

  online_store_config {
    enable_online_store = true
    security_config {
      kms_key_id = var.kms_key_arn
    }
  }

  offline_store_config {
    s3_storage_config {
      s3_uri     = "s3://${var.offline_bucket}/feature-groups"
      kms_key_id = var.kms_key_arn
    }
    data_catalog_config {
      catalog    = "AwsDataCatalog"
      database   = var.glue_database
      table_name = replace(local.name, "-", "_")
    }
    disable_glue_table_creation = false
    table_format                = "Iceberg"
  }

  tags = var.tags
}

output "name" {
  value = aws_sagemaker_feature_group.this.feature_group_name
}

output "arn" {
  value = aws_sagemaker_feature_group.this.arn
}
