terraform {
  backend "s3" {
    bucket         = "pytorch-genomic-terraform-state"
    key            = "pytorch-genomic/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "pytorch-genomic-terraform-locks"
    encrypt        = true
  }
}
