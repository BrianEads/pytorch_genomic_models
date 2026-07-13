variable "s3_dataset_bucket" {
  type    = string
  default = "pytorch-genomic-datasets"
}

variable "budget_alert_usd" {
  type    = number
  default = 500
}

variable "alert_email" {
  type    = string
  default = ""
}

variable "key_pair_name" {
  type    = string
  default = ""
}
