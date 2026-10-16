variable "env" {
  description = "dev | staging | prod"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.env)
    error_message = "env must be dev, staging or prod."
  }
}

variable "name_prefix" {
  type    = string
  default = "aucml"
}

variable "capture_retention_days" {
  description = "How long endpoint data capture is kept (drift monitoring reads the last few weeks)."
  type        = number
  default     = 90
}

variable "alert_email" {
  description = "Optional e-mail subscribed to the alert topic."
  type        = string
  default     = null
}

variable "tags" {
  type    = map(string)
  default = {}
}
