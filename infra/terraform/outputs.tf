output "endpoints" {
  value = { for k, m in module.endpoint : k => m.endpoint_name }
}

output "feature_groups" {
  value = { for k, f in module.feature_group : k => f.name }
}

output "serving_repository_url" {
  value = module.platform.serving_repository_url
}

output "alerts_topic_arn" {
  value = module.platform.alerts_topic_arn
}
