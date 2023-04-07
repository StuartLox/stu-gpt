resource "helm_release" "karpenter_helm" {
  namespace        = "karpenter"
  create_namespace = true

  name       = "karpenter"
  repository = "https://charts.karpenter.sh"
  chart      = "karpenter"
  version    = "0.16.3"

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = var.karpenter_controller_role_arn
  }

  set {
    name  = "clusterName"
    value = var.cluster_id
  }

  set {
    name  = "clusterEndpoint"
    value = var.cluster_endpoint
  }

  set {
    name  = "aws.defaultInstanceProfile"
    value = var.aws_iam_instance_profile_karpenter_name
  }
}
