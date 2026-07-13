# Prod environment — multi-queue spot training and hyperparameter sweep.

module "infra" {
  source = "../../"

  environment       = "prod"
  s3_dataset_bucket = var.s3_dataset_bucket
  budget_alert_usd  = var.budget_alert_usd
  alert_email       = var.alert_email
  key_pair_name     = var.key_pair_name
}
