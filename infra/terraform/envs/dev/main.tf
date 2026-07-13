# Dev environment — single g4dn.xlarge queue, on-demand compute.

module "infra" {
  source = "../../"

  environment       = "dev"
  s3_dataset_bucket = var.s3_dataset_bucket
  budget_alert_usd  = var.budget_alert_usd
  alert_email       = var.alert_email
  key_pair_name     = var.key_pair_name
}
