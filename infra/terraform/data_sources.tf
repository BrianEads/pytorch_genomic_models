# Service Catalog networking outputs — read from SSM Parameter Store.
# Confirm paths with: aws ssm get-parameters-by-path --path "/bayer/platform/networking/" --recursive

data "aws_ssm_parameter" "vpc_id" {
  name = var.ssm_vpc_id_path
}

data "aws_ssm_parameter" "private_subnet_ids" {
  name = var.ssm_private_subnet_ids_path
}

data "aws_ssm_parameter" "s3_endpoint_id" {
  name = var.ssm_s3_endpoint_id_path
}

locals {
  vpc_id             = data.aws_ssm_parameter.vpc_id.value
  private_subnet_ids = split(",", data.aws_ssm_parameter.private_subnet_ids.value)
  s3_endpoint_id     = data.aws_ssm_parameter.s3_endpoint_id.value
}
