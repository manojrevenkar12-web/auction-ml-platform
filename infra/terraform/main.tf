# Root module: one apply per environment (dev / staging / prod), selected by
#   tofu init  -backend-config=environments/<env>.backend.hcl
#   tofu apply -var-file=environments/<env>.tfvars -var-file=environments/<env>.models.tfvars.json
# The models file is rewritten by the promotion workflow when a version moves lanes.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      project     = "aucml"
      environment = var.env
      managed_by  = "terraform"
    }
  }
}

locals {
  # Single source of truth for feature groups: the same catalog the platform reads.
  catalog = yamldecode(file("${path.module}/../../feature_repo/catalog.yaml"))
  batch_groups = {
    for name, g in local.catalog.groups : name => g
    if g.kind == "batch" && !contains(var.excluded_feature_groups, name)
  }
}

module "platform" {
  source      = "./modules/ml_platform"
  env         = var.env
  alert_email = var.alert_email
}

module "feature_group" {
  source         = "./modules/feature_group"
  for_each       = local.batch_groups
  prefix         = module.platform.prefix
  group          = each.key
  features       = each.value.features
  role_arn       = module.platform.execution_role_arn
  kms_key_arn    = module.platform.kms_key_arn
  offline_bucket = module.platform.bucket_names["offline-store"]
  glue_database  = module.platform.glue_database
}

module "endpoint" {
  source   = "./modules/model_endpoint"
  for_each = var.models

  prefix                  = module.platform.prefix
  model                   = each.key
  model_version           = each.value.version
  model_data_url          = each.value.model_data_url
  image_uri               = "${module.platform.serving_repository_url}:${var.serving_image_tag}"
  role_arn                = module.platform.execution_role_arn
  kms_key_arn             = module.platform.kms_key_arn
  capture_bucket          = module.platform.bucket_names["data-capture"]
  alerts_topic_arn        = module.platform.alerts_topic_arn
  instance_type           = var.instance_type
  min_instances           = var.min_instances
  max_instances           = var.max_instances
  primary_metric          = each.value.primary_metric
  metric_direction        = each.value.metric_direction
  performance_critical    = each.value.performance_critical
  prediction_psi_critical = each.value.prediction_psi_critical
  latency_p95_ms          = each.value.latency_p95_ms
}
