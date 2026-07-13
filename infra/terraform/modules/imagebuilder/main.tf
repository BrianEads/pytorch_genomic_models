# Variables are defined in variables.tf

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Goal        = "goal-4"
  }

  component_files = [
    "cuda_install.yaml",
    "miniconda_env.yaml",
    "pytorch_stack.yaml",
    "project_deps.yaml",
    "efs_utils.yaml",
  ]
}

resource "aws_imagebuilder_component" "gpu_training" {
  for_each = toset(local.component_files)

  name     = "${var.project_name}-${replace(each.value, ".yaml", "")}"
  platform = "Linux"
  version  = "1.0.0"
  data     = file("${path.module}/components/${each.value}")
  tags     = local.common_tags
}

resource "aws_imagebuilder_image_recipe" "gpu_training" {
  name         = "${var.project_name}-gpu-training"
  parent_image = "arn:aws:imagebuilder:${var.aws_region}:aws:image/amazon-linux-2-x86-gpu/latest"
  version      = "1.0.0"

  dynamic "component" {
    for_each = local.component_files
    content {
      component_arn = aws_imagebuilder_component.gpu_training[component.value].arn
    }
  }

  tags = local.common_tags
}

resource "aws_imagebuilder_infrastructure_configuration" "builder" {
  name                          = "${var.project_name}-gpu-builder"
  instance_profile_name         = "${var.project_name}-imagebuilder-instance-profile"
  instance_types                = ["g4dn.xlarge"]
  subnet_id                     = var.private_subnet_ids[0]
  terminate_instance_on_failure = true
  tags                          = local.common_tags
}

resource "aws_imagebuilder_distribution_configuration" "gpu_training" {
  name = "${var.project_name}-gpu-training-dist"

  distribution {
    region = var.aws_region

    ami_distribution_configuration {
      name = "${var.project_name}-gpu-training-{{ imagebuilder:buildDate }}"
    }
  }

  tags = local.common_tags
}

resource "aws_imagebuilder_image_pipeline" "gpu_training" {
  name                             = "${var.project_name}-gpu-training"
  image_recipe_arn                 = aws_imagebuilder_image_recipe.gpu_training.arn
  infrastructure_configuration_arn = aws_imagebuilder_infrastructure_configuration.builder.arn
  distribution_configuration_arn   = aws_imagebuilder_distribution_configuration.gpu_training.arn

  schedule {
    schedule_expression = var.imagebuilder_schedule
  }

  tags = local.common_tags
}

resource "aws_ssm_parameter" "ami_latest" {
  name  = "/bayer/pytorch-genomic/ami/gpu-training-latest"
  type  = "String"
  value = "pending-first-build"
  tags  = local.common_tags

  lifecycle {
    ignore_changes = [value]
  }
}
