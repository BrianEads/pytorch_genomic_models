output "pipeline_arn" {
  value = aws_imagebuilder_image_pipeline.gpu_training.arn
}

output "ami_ssm_parameter_name" {
  value = aws_ssm_parameter.ami_latest.name
}
