terraform {
  backend "s3" {
    bucket         = "cs-cp-bifx-dfw-pytorch-genomic-terraform-state"
    key            = "pytorch-genomic/dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "cs-cp-bifx-dfw-pytorch-genomic-terraform-locks"
    encrypt        = true
  }
}
