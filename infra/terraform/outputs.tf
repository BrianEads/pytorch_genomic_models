output "s3_dataset_bucket_arn" {
  description = "ARN of the S3 bucket for datasets and checkpoints."
  value       = module.storage.bucket_arn
}

output "s3_dataset_bucket_name" {
  description = "Name of the S3 bucket for datasets and checkpoints."
  value       = module.storage.bucket_name
}

output "efs_filesystem_id" {
  description = "EFS filesystem ID for ParallelCluster shared storage (null if disabled)."
  value       = var.enable_efs ? module.efs_mount[0].filesystem_id : null
}

output "efs_dns_name" {
  description = "EFS DNS name for mount targets."
  value       = var.enable_efs ? module.efs_mount[0].dns_name : null
}

output "imagebuilder_pipeline_arn" {
  description = "ARN of the GPU training ImageBuilder pipeline."
  value       = module.imagebuilder.pipeline_arn
}

output "ami_ssm_parameter_name" {
  description = "SSM parameter path for the latest GPU training AMI."
  value       = module.imagebuilder.ami_ssm_parameter_name
}

output "monitoring_sns_topic_arn" {
  description = "SNS topic ARN for cost and idle alerts."
  value       = module.monitoring.sns_topic_arn
}

output "pcluster_config_path" {
  description = "Relative path to the ParallelCluster configuration template."
  value       = "../pcluster/cluster_config.yaml"
}
