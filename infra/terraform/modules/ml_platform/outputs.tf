output "prefix" {
  value = local.prefix
}

output "kms_key_arn" {
  value = aws_kms_key.ml.arn
}

output "bucket_names" {
  value = { for k, b in aws_s3_bucket.this : k => b.bucket }
}

output "execution_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "glue_database" {
  value = aws_glue_catalog_database.offline.name
}

output "serving_repository_url" {
  value = aws_ecr_repository.serving.repository_url
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
