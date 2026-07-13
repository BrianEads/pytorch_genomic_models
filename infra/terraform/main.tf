module "storage" {
  source = "./modules/storage"

  project_name      = var.project_name
  environment       = var.environment
  s3_dataset_bucket = var.s3_dataset_bucket
  vpc_id            = local.vpc_id
}

module "efs_mount" {
  count  = var.enable_efs ? 1 : 0
  source = "./modules/efs_mount"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = local.vpc_id
  private_subnet_ids = local.private_subnet_ids
  throughput_mode    = var.efs_throughput_mode
}

module "imagebuilder" {
  source = "./modules/imagebuilder"

  project_name          = var.project_name
  environment           = var.environment
  aws_region            = var.aws_region
  vpc_id                = local.vpc_id
  private_subnet_ids    = local.private_subnet_ids
  imagebuilder_schedule = var.imagebuilder_schedule
}

module "monitoring" {
  source = "./modules/monitoring"

  project_name           = var.project_name
  environment            = var.environment
  budget_alert_usd       = var.budget_alert_usd
  auto_stop_idle_minutes = var.auto_stop_idle_minutes
  alert_email            = var.alert_email
}
