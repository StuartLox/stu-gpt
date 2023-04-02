variable "region" {
  default = "ap-southeast-2"
}

variable "instance_ami" {
  default = "ami-0159ec8365aea1724" # EKS AMI Sydney
}

variable "instance_type" {
  default = "t3.medium"
}

variable "key" {
  default = ""
}

variable "cluster_name" {
  description = "Cluster Name"
  default     = "personal-eks"
}

variable "server_name" {
  description = "Ec2 Server Name"
  default     = "worker_nodes"
}

variable "vpc_name" {
  description = "VPC name"
  default     = "personal-vpc"
}

