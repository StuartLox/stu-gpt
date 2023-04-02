
variable "vpc_id" {
  description = "Unique Idenifier of the VPC"
}

variable "instance_type" {
  description = "instance type for K8s creation Server - Eg:t2.micro"
}

variable "instance_ami" {
  description = "AMI for Instance"
}

variable "instance_key" {
  description = "Key for k8s Server"
}

variable "owner" {
  description = "Owner of the application"
  default     = "stu"
}

variable "k8_subnet" {
  description = "Subnet for the K8s instances"
}