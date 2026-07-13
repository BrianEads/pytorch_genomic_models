output "bucket_arn" {
  value = aws_s3_bucket.datasets.arn
}

output "bucket_name" {
  value = aws_s3_bucket.datasets.id
}
