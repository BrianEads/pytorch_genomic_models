variable "aws_region" {
  type        = string
  description = "AWS region to deploy resources into."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Tag prefix applied to all resources for cost allocation and identification."
  default     = "pytorch-genomic"
}

variable "environment" {
  type        = string
  description = "Deployment environment label: 'dev' or 'prod'."
  default     = "dev"
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

variable "owner" {
  type        = string
  description = "Resource owner tag for cost allocation."
  default     = "infra-provisioner"
}

variable "key_pair_name" {
  type        = string
  description = "Name of an existing EC2 key pair for SSH fallback. Primary access is via SSM Session Manager."
  default     = ""
}

variable "s3_dataset_bucket" {
  type        = string
  description = "S3 bucket for datasets and checkpoints. Dev (DFW account): cs-cp-bifx-dfw-pytorch-genomic-data. Prod: pytorch-genomic-datasets."
  default     = "pytorch-genomic-datasets"
}

variable "enable_efs" {
  type        = bool
  description = "Whether to provision an EFS filesystem for shared multi-node dataset access."
  default     = true
}

variable "efs_throughput_mode" {
  type        = string
  description = "EFS throughput mode: 'bursting' or 'provisioned'."
  default     = "bursting"
}

variable "imagebuilder_schedule" {
  type        = string
  description = "Cron schedule expression for the ImageBuilder pipeline. Default: weekly Sunday 04:00 UTC."
  default     = "cron(0 4 ? * SUN *)"
}

variable "budget_alert_usd" {
  type        = number
  description = "Monthly cost threshold in USD for the CloudWatch budget alert."
  default     = 200
}

variable "auto_stop_idle_minutes" {
  type        = number
  description = "Consecutive minutes of ParallelCluster head node CPU idle before auto-stop alarm fires."
  default     = 60
}

variable "alert_email" {
  type        = string
  description = "Email address for SNS budget and cost anomaly alerts."
  default     = ""
}

# SSM parameter paths — written by Service Catalog products; read as data sources
variable "ssm_vpc_id_path" {
  type        = string
  description = "SSM Parameter Store path for the VPC ID provisioned by Service Catalog."
  default     = "/bayer/platform/networking/vpc_id"
}

variable "ssm_private_subnet_ids_path" {
  type        = string
  description = "SSM Parameter Store path for comma-separated private subnet IDs."
  default     = "/bayer/platform/networking/private_subnet_ids"
}

variable "ssm_s3_endpoint_id_path" {
  type        = string
  description = "SSM Parameter Store path for the S3 VPC Gateway endpoint ID."
  default     = "/bayer/platform/networking/s3_vpc_endpoint_id"
}
