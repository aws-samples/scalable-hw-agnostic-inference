#!/bin/bash -x
DLC_ECR_ACCOUNT="763104351884"

DLC_NEURON_IMAGE="pytorch-inference-neuronx"
DLC_NEURON_TAG=$BASE_IMAGE_TAG
DLC_NEURON_ECR=$DLC_ECR_ACCOUNT".dkr.ecr.us-west-2.amazonaws.com"
if [ "$IMAGE_TAG" == "amd64-neuron" ]; then
  docker logout
  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $DLC_NEURON_ECR
  docker pull $DLC_NEURON_ECR/$DLC_NEURON_IMAGE:$DLC_NEURON_TAG
  dlc_xla_image_id=$(docker images | grep $DLC_ECR_ACCOUNT | grep $DLC_NEURON_IMAGE | awk '{print $3}')
  docker tag $dlc_xla_image_id $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$DLC_NEURON_TAG
  docker logout
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO
  docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$DLC_NEURON_TAG
fi
DLC_CUDA_IMAGE="pytorch-inference"
DLC_CUDA_TAG=$BASE_IMAGE_TAG
DLC_CUDA_ECR=$DLC_ECR_ACCOUNT".dkr.ecr.us-east-1.amazonaws.com"
if [ "$IMAGE_TAG" == "amd64-cuda" ]; then
  docker logout
  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $DLC_CUDA_ECR
  docker pull $DLC_CUDA_ECR/$DLC_CUDA_IMAGE:$DLC_CUDA_TAG
  dlc_cuda_image_id=$(docker images | grep $DLC_ECR_ACCOUNT | grep $DLC_CUDA_TAG | awk '{print $3}')
  docker tag $dlc_cuda_image_id $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$DLC_CUDA_TAG
  docker logout
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO
  docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$DLC_CUDA_TAG
fi
DLC_ARM_CPU_IMAGE="pytorch-inference-graviton"
DLC_ARM_CPU_TAG=$BASE_IMAGE_TAG
DLC_ARM_CPU_ECR=$DLC_ECR_ACCOUNT".dkr.ecr.us-east-1.amazonaws.com"
if [ "$IMAGE_TAG" == "aarch64-cpu" ]; then
  docker logout
  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $DLC_ARM_CPU_ECR
  docker pull $DLC_ARM_CPU_ECR/$DLC_ARM_CPU_IMAGE:$DLC_ARM_CPU_TAG
  dlc_arm_cpu_image_id=$(docker images | grep $DLC_ECR_ACCOUNT | grep $DLC_ARM_CPU_TAG | awk '{print $3}')
  docker tag $dlc_arm_cpu_image_id $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$DLC_ARM_CPU_TAG
  docker logout
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO
  docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$DLC_ARM_CPU_TAG
fi

docker images

ASSETS="-assets"
export IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$IMAGE_TAG$ASSETS
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $IMAGE
docker build -t $IMAGE --build-arg ai_chip=$IMAGE_TAG  -f Dockerfile-assets .
docker push $IMAGE
