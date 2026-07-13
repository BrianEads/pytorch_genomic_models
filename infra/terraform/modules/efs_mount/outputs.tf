output "filesystem_id" {
  value = aws_efs_file_system.data.id
}

output "dns_name" {
  value = aws_efs_file_system.data.dns_name
}

output "ssm_parameter_name" {
  value = aws_ssm_parameter.filesystem_id.name
}
