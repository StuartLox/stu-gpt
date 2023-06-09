#
# Variables Configuration
#
variable "cluster_name" {}

variable "vpc_id" {
  description = "VPC ID "
}

variable "eks_subnets" {
  description = "Master subnet ids"
  type        = list(string)
}

variable "worker_subnet" {
  type = list(string)
}

variable "subnet_ids" {
  type        = list(string)
  description = "List of all subnet in cluster"
}

variable "kubernetes_server_instance_sg" {
  description = "Kubenetes control server security group"
}

variable "instance_ami" {
  description = "AMI for Instance"
}

variable "instance_type" {
  default = "t3.medium"
}

variable "eks_cluster_version" {
  default = "1.25"
}
