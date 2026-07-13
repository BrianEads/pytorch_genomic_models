variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "s3_dataset_bucket" {
  type = string
}

variable "vpc_id" {
  type        = string
  description = "VPC ID from Service Catalog SSM parameter (used in bucket policy conditions)."
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Goal        = "goal-4"
  }
}

resource "aws_s3_bucket" "datasets" {
  bucket = var.s3_dataset_bucket
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Expected prefix layout (DFW dev account):
#   raw/         — data-fetch-wizard staged files
#   tokenised/   — Goal 3 curation pipeline outputs
#   manifests/   — DatasetManifest JSON files
#   checkpoints/ — Goal 2 training checkpoints

resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    id     = "expire-old-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    transition {
      days          = 30
      storage_class = "GLACIER_IR"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "keep-latest-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
