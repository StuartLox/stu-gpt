variable "cluster_endpoint" {
  description = "The k8s cluster endpoint"
  type        = string
}

variable "cluster_id" {
  description = "The k8s cluster id"
  type        = string
}

variable "aws_iam_instance_profile_karpenter_name" {
  description = "AWS Instance profile"
  type        = string
}

variable "karpenter_controller_role_arn" {
  default = "Karpenter role arn"
  type    = string
}