terraform {
  cloud {
    organization = "stuart-lox"

    workspaces {
      name = "prod-stu-gpt"
    }
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.6"
    }

  }
}

provider "aws" {
  region  = "${var.region}"
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
}
