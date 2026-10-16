# One real-time SageMaker endpoint per model: the packaged model.tar.gz served by
# the aucml image, data capture on (feeds drift monitoring), blue/green canary
# deployment with automatic rollback on errors, target-tracking autoscaling, and
# alarms on errors, latency and the platform's own monitoring metrics.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60"
    }
  }
}

locals {
  slug          = replace(var.model, "_", "-")
  endpoint_name = substr("${var.prefix}-${local.slug}", 0, 63)
  # model and endpoint-config names change with the package so updates are
  # create-before-destroy and the endpoint switches with a canary.
  pkg_hash = substr(sha1(var.model_data_url), 0, 8)
}

resource "aws_sagemaker_model" "this" {
  name               = substr("${var.prefix}-${local.slug}-${local.pkg_hash}", 0, 63)
  execution_role_arn = var.role_arn

  primary_container {
    image          = var.image_uri
    model_data_url = var.model_data_url
    environment = {
      AUCML_FEATURE_STORE        = "sagemaker"
      AUCML_FG_PREFIX            = var.prefix
      AUCML_MODEL_VERSION        = var.model_version
      SAGEMAKER_PROGRAM          = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "/opt/ml/model/code"
    }
  }

  tags = merge(var.tags, { model = var.model, version = var.model_version })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint_configuration" "this" {
  name_prefix = substr("${var.prefix}-${local.slug}-", 0, 40)
  kms_key_arn = var.kms_key_arn

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.this.name
    initial_instance_count = var.min_instances
    instance_type          = var.instance_type
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = var.capture_percent
    destination_s3_uri          = "s3://${var.capture_bucket}/${var.model}"
    kms_key_id                  = var.kms_key_arn
    capture_options {
      capture_mode = "Input"
    }
    capture_options {
      capture_mode = "Output"
    }
    capture_content_type_header {
      json_content_types = ["application/json"]
    }
  }

  tags = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name

  deployment_config {
    blue_green_update_policy {
      traffic_routing_configuration {
        type                     = "CANARY"
        wait_interval_in_seconds = 300
        canary_size {
          type  = "CAPACITY_PERCENT"
          value = 10
        }
      }
      termination_wait_in_seconds          = 120
      maximum_execution_timeout_in_seconds = 3600
    }
    auto_rollback_configuration {
      alarms {
        alarm_name = aws_cloudwatch_metric_alarm.errors.alarm_name
      }
      alarms {
        alarm_name = aws_cloudwatch_metric_alarm.latency.alarm_name
      }
    }
  }

  tags = var.tags
}

resource "aws_appautoscaling_target" "this" {
  min_capacity       = var.min_instances
  max_capacity       = var.max_instances
  resource_id        = "endpoint/${aws_sagemaker_endpoint.this.name}/variant/AllTraffic"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "invocations" {
  name               = "${local.endpoint_name}-invocations"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.this.resource_id
  scalable_dimension = aws_appautoscaling_target.this.scalable_dimension
  service_namespace  = aws_appautoscaling_target.this.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = var.target_invocations_per_instance
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
  }
}

# --- alarms ---------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${local.endpoint_name}-5xx"
  alarm_description   = "Model container returning 5xx: page and roll back a deployment in flight."
  namespace           = "AWS/SageMaker"
  metric_name         = "Invocation5XXErrors"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    EndpointName = local.endpoint_name
    VariantName  = "AllTraffic"
  }
  alarm_actions = [var.alerts_topic_arn]
  ok_actions    = [var.alerts_topic_arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "latency" {
  alarm_name          = "${local.endpoint_name}-p95-latency"
  alarm_description   = "p95 model latency above the gate's budget."
  namespace           = "AWS/SageMaker"
  metric_name         = "ModelLatency"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = var.latency_p95_ms * 1000 # ModelLatency is in microseconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    EndpointName = local.endpoint_name
    VariantName  = "AllTraffic"
  }
  alarm_actions = [var.alerts_topic_arn]
  tags          = var.tags
}

# Emitted by the weekly monitoring job as CloudWatch EMF (aucml.monitoring.alerts.EmfSink).
resource "aws_cloudwatch_metric_alarm" "performance" {
  alarm_name          = "${local.endpoint_name}-performance"
  alarm_description   = "Delayed-label ${var.primary_metric} past the critical limit vs the release reference."
  namespace           = "aucml/monitoring"
  metric_name         = "perf.${var.primary_metric}"
  statistic           = "Maximum"
  period              = 604800
  evaluation_periods  = 1
  threshold           = var.performance_critical
  comparison_operator = var.metric_direction == "lower" ? "GreaterThanThreshold" : "LessThanThreshold"
  treat_missing_data  = "missing"
  dimensions = {
    Model = var.model
  }
  alarm_actions = [var.alerts_topic_arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "prediction_drift" {
  alarm_name          = "${local.endpoint_name}-prediction-drift"
  alarm_description   = "PSI of the prediction distribution past its calibrated critical threshold."
  namespace           = "aucml/monitoring"
  metric_name         = "psi.__prediction__"
  statistic           = "Maximum"
  period              = 604800
  evaluation_periods  = 1
  threshold           = var.prediction_psi_critical
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "missing"
  dimensions = {
    Model = var.model
  }
  alarm_actions = [var.alerts_topic_arn]
  tags          = var.tags
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.this.name
}

output "model_name" {
  value = aws_sagemaker_model.this.name
}
