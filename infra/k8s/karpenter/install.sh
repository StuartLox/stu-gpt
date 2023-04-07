set -x

export VERSION=0.16.3
export CLUSTER_NAME=personal-eks
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export KARPENTER_IAM_ROLE_ARN="arn:aws:iam::415646757531:role/karpenter-controller"
export CLUSTER_ENDPOINT="$(aws eks describe-cluster --name ${CLUSTER_NAME} --query "cluster.endpoint" --output text)"


helm upgrade --install --namespace karpenter --create-namespace \
  karpenter karpenter/karpenter \
  --version ${VERSION} \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="arn:aws:iam::415646757531:role/karpenter-controller" \
  --set clusterName=${CLUSTER_NAME} \
  --set clusterEndpoint=${CLUSTER_ENDPOINT} \
  --set aws.defaultInstanceProfile=KarpenterNodeInstanceProfile \
  --wait # for the defaulting webhook to install before creating a Provisioner
