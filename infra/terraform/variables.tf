variable "env" {
  type = string
}

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "alert_email" {
  type    = string
  default = null
}

variable "serving_image_tag" {
  description = "Immutable tag of the aucml serving image in ECR (git SHA)."
  type        = string
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

variable "excluded_feature_groups" {
  description = "Catalog groups that exist only for tests/demos and are never provisioned."
  type        = list(string)
  default     = ["lane_cost_56d_by_booking", "auction_outcome"]
}

variable "models" {
  description = "Models deployed in this environment, written by the promotion workflow from the registry lane."
  type = map(object({
    version                 = string
    model_data_url          = string
    primary_metric          = string
    metric_direction        = string
    performance_critical    = number
    prediction_psi_critical = number
    latency_p95_ms          = number
  }))
  default = {}
  validation {
    condition     = alltrue([for m in values(var.models) : can(regex("^s3://.+/model\\.tar\\.gz$", m.model_data_url))])
    error_message = "Every model_data_url must be an s3:// URI of a packaged model.tar.gz."
  }
}
