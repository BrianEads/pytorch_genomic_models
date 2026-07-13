# Variables are defined in variables.tf

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Goal        = "goal-4"
  }
}

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-${var.environment}-alerts"
  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "email" {
  count = var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "billing" {
  alarm_name          = "${var.project_name}-${var.environment}-monthly-budget"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600
  statistic           = "Maximum"
  threshold           = var.budget_alert_usd
  alarm_description   = "Monthly AWS spend exceeds budget threshold"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  dimensions = {
    Currency = "USD"
  }
}

resource "aws_ce_anomaly_monitor" "project" {
  name              = "${var.project_name}-${var.environment}-cost-anomaly"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"
}

resource "aws_ce_anomaly_subscription" "project" {
  name      = "${var.project_name}-${var.environment}-anomaly-sub"
  frequency = "DAILY"
  monitor_arn_list = [
    aws_ce_anomaly_monitor.project.arn,
  ]

  subscriber {
    type    = "SNS"
    address = aws_sns_topic.alerts.arn
  }

  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = ["100"]
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }
}
