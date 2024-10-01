
* Fork https://github.com/aws-samples/scalable-hw-agnostic-inference and populate the `GITHUB_USER` and `GITHUB_OAUTH_TOKEN` based on `Settings/Developer Settings/Personal access tokens`.
* Check the latest [DLC](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) for `BASE_IMAGE_AMD_XLA_TAG` and `BASE_IMAGE_AMD_CUD_TAG` values.
* Export the following variables
```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)
export AWS_REGION=us-west-2
export BASE_IMAGE_AMD_XLA_TAG=1.13.1-neuronx-py310-sdk2.20.0-ubuntu20.04
export BASE_IMAGE_AMD_CUD_TAG=2.3.0-gpu-py311-cu121-ubuntu20.04-ec2
export BASE_IMAGE_ARM_CPU_TAG=2.3.0-cpu-py311-ubuntu20.04-ec2
export IMAGE_AMD_XLA_TAG=amd64-neuron
export IMAGE_AMD_CUD_TAG=amd64-cuda
export IMAGE_ARM_CPU_TAG=aarch64-cpu
export BASE_REPO=stablediffusion
export BASE_TAG=multiarch-ubuntu
export BASE_AMD_TAG=amd64
export GITHUB_BRANCH=master
export GITHUB_USER=yahavb
export GITHUB_REPO=scalable-hw-agnostic-inference
export MODEL_DIR=sd2_compile_dir
export CF_STACK=hw-agnostic-inference-cdk
```
* Install needed packages

```bash
npm uninstall -g aws-cdk
npm install -g aws-cdk
```

* Deploy the pipeline

```bash
cd ci-build
./deploy-pipeline.sh
```
