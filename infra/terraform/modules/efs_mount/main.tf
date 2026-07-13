variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "throughput_mode" {
  type    = string
  default = "bursting"
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Goal        = "goal-4"
  }
}

resource "aws_efs_file_system" "data" {
  creation_token   = "${var.project_name}-${var.environment}-data"
  encrypted        = true
  throughput_mode  = var.throughput_mode
  performance_mode = "generalPurpose"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-efs"
  })

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }
}

resource "aws_security_group" "efs" {
  name        = "${var.project_name}-${var.environment}-efs"
  description = "Allow NFS from private subnets for EFS mount targets"
  vpc_id      = var.vpc_id

  ingress {
    description = "NFS from VPC"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_efs_mount_target" "data" {
  count = length(var.private_subnet_ids)

  file_system_id  = aws_efs_file_system.data.id
  subnet_id       = var.private_subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_ssm_parameter" "filesystem_id" {
  name  = "/bayer/pytorch-genomic/efs/filesystem_id"
  type  = "String"
  value = aws_efs_file_system.data.id
  tags  = local.common_tags
}
