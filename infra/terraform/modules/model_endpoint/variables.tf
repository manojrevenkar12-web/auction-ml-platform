variable "prefix" {
  type = string
}

variable "model" {
  description = "Model name as in models/<name>/model.yaml"
  type        = string
}

variable "model_version" {
  description = "Registry version (run id) being deployed."
  type        = string
}

variable "model_data_url" {
  description = "s3:// URI of the model.tar.gz produced by `aucml release`."
  type        = string
  validation {
    condition     = can(regex("^s3://.+/model\\.tar\\.gz$", var.model_data_url))
    error_message = "model_data_url must be an s3:// URI ending in model.tar.gz."
  }
}

variable "image_uri" {
  type = string
}

variable "role_arn" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "capture_bucket" {
  type = string
}

variable "alerts_topic_arn" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "ml.c6i.large"
}

variable "min_instances" {
  type    = number
  default = 1
}

variable "max_instances" {
  type    = number
  default = 2
}

variable "target_invocations_per_instance" {
  type    = number
  default = 600
}

variable "capture_percent" {
  type    = number
  default = 100
}

variable "latency_p95_ms" {
  type    = number
  default = 100
}

variable "primary_metric" {
  type = string
}

variable "metric_direction" {
  type = string
  validation {
    condition     = contains(["lower", "higher"], var.metric_direction)
    error_message = "metric_direction must be lower or higher."
  }
}

variable "performance_critical" {
  type = number
}

variable "prediction_psi_critical" {
  type = number
}

variable "tags" {
  type    = map(string)
  default = {}
}
