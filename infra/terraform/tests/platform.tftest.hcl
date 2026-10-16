# Plan-only test with a mocked AWS provider: proves the root module wires the
# catalog into feature groups and the registry lanes into endpoints, without
# touching an AWS account. Run: tofu test

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }
  mock_data "aws_partition" {
    defaults = {
      partition = "aws"
    }
  }
  mock_resource "aws_sns_topic" {
    defaults = {
      arn = "arn:aws:sns:eu-central-1:123456789012:aucml-prod-model-alerts"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/aucml-prod-sagemaker-execution"
    }
  }
  mock_resource "aws_kms_key" {
    defaults = {
      arn = "arn:aws:kms:eu-central-1:123456789012:key/00000000-0000-0000-0000-000000000000"
    }
  }
  mock_resource "aws_ecr_repository" {
    defaults = {
      repository_url = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/aucml-prod-serving"
      arn            = "arn:aws:ecr:eu-central-1:123456789012:repository/aucml-prod-serving"
    }
  }
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }
}

variables {
  env               = "prod"
  serving_image_tag = "abc1234"
  models = {
    price_estimator = {
      version                 = "price_estimator-20251101-deadbeef"
      model_data_url          = "s3://aucml-prod-artifacts-123456789012/models/price_estimator/price_estimator-20251101-deadbeef/model.tar.gz"
      primary_metric          = "mape"
      metric_direction        = "lower"
      performance_critical    = 0.12
      prediction_psi_critical = 0.15
      latency_p95_ms          = 100
    }
    sale_probability = {
      version                 = "sale_probability-20251101-cafef00d"
      model_data_url          = "s3://aucml-prod-artifacts-123456789012/models/sale_probability/sale_probability-20251101-cafef00d/model.tar.gz"
      primary_metric          = "auc"
      metric_direction        = "higher"
      performance_critical    = 0.40
      prediction_psi_critical = 0.15
      latency_p95_ms          = 100
    }
  }
}

run "feature_groups_come_from_the_catalog" {
  command = plan

  assert {
    condition     = length(module.feature_group) == 5
    error_message = "expected the 5 production batch groups from feature_repo/catalog.yaml"
  }
  assert {
    condition     = !contains(keys(module.feature_group), "lane_cost_56d_by_booking")
    error_message = "the deliberately wrong demo group must never be provisioned"
  }
  assert {
    condition     = module.feature_group["seller_90d"].name == "aucml-prod-seller-90d"
    error_message = "feature group names must be prefixed and use dashes"
  }
}

run "one_endpoint_per_deployed_model" {
  command = plan

  assert {
    condition     = length(module.endpoint) == 2
    error_message = "expected one endpoint per model in the prod lane"
  }
  assert {
    condition     = module.endpoint["sale_probability"].endpoint_name == "aucml-prod-sale-probability"
    error_message = "unexpected endpoint name"
  }
}

run "rejects_non_package_uris" {
  command = plan

  variables {
    models = {
      price_estimator = {
        version                 = "x"
        model_data_url          = "s3://bucket/some/other/file.joblib"
        primary_metric          = "mape"
        metric_direction        = "lower"
        performance_critical    = 0.12
        prediction_psi_critical = 0.15
        latency_p95_ms          = 100
      }
    }
  }

  expect_failures = [var.models]
}
