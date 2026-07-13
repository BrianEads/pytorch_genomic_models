variable "aws_region" {
  type        = string
  description = "AWS region for the DFW dev/testing account."
  default     = "us-east-1"
}

variable "s3_dataset_bucket" {
  type        = string
  description = "DFW account S3 bucket for raw (data-fetch-wizard), tokenised (Goal 3), and training checkpoints."
  default     = "cs-cp-bifx-dfw-pytorch-genomic-data"
}

variable "s3_raw_prefix" {
  type        = string
  description = "S3 prefix for raw files written by data-fetch-wizard."
  default     = "raw/"
}

variable "s3_tokenised_prefix" {
  type        = string
  description = "S3 prefix for tokenised outputs from Goal 3 curation pipelines."
  default     = "tokenised/"
}

variable "s3_manifests_prefix" {
  type        = string
  description = "S3 prefix for DatasetManifest JSON files."
  default     = "manifests/"
}

variable "budget_alert_usd" {
  type    = number
  default = 100
}

variable "alert_email" {
  type    = string
  default = ""
}

variable "key_pair_name" {
  type    = string
  default = ""
}
