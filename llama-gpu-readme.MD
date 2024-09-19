# Llama 3 8B on EKS with Nvidia GPU's

## Prerequisites

* If you followed the sd-gpu-readme.MD instructions, you would have created an EKS cluster with Karpenter, setup Karpenter Environment Variables, Installed the AWS Load Balancer Controller, Deployed the NVIDIA Device Plugin and the NVIDIA Nodepool. If all these steps have been followed, then you can move on with this guide. 
* Ensure that you have a HuggingFace account created and have read and accepted the Meta Llama 3 8B community license agreement found here: huggingface.co/meta-llama/Meta-Llama-3-8B. This will ensure that the model can be downloaded and accessed from the HuggingFace gated repository and we will not face issues with it later on in the process.


## Deploy HuggingFace Secret

* Make sure to create and apply a secret file for your HuggingFace Token as this will be a necessary step in order to deploy Llama. Get your HuggingFace token from https://huggingface.co/settings/tokens and make sure to keep track of the token value. Create a file called hfsecret.yaml, and paste the YAML code below into the file and replace the [TOKEN] with a Base64 Encoded version of your hugging face token. This can be done by running echo -n "your-huggingface-token" | base64, in terminal and copying the Base64 encoded version and replacing it with [TOKEN]. This is because Kubernetes Secrets stores data as base64-encoded values. Once set, we can then apply the secret by running this command: kubectl apply -f hfsecrets.yaml. If using GitHub, make sure to add the secrets file within your .gitignore to ensure that your HuggingFace token does not get pushed to your repository. 

apiVersion: v1
kind: Secret
metadata:
  name: hf-secrets
type: Opaque
data:
  HUGGINGFACE_TOKEN: [TOKEN]

Also note, there is a difference between the sd-gpu-deploy.yaml file and llama-gpu-deploy.yaml file. We are adding a few more parameters to the container environment variables, including our new HuggingFace Token and MaxNewTokens. 


## Deploy Lllama

* This file aims to deploy stable diffusion 2.1 onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
* cat llama-gpu-deploy.yaml | envsubst | kubectl apply -f -


## Deploy Service

* We are deploying a service file called sd-gpu-svc.yaml focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-gpu. 
* kubectl apply -f llama-gpu-svc.yaml


## Deploy Ing

* We will be deploying an ingress file called sd-gpu-ing.yaml which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
* kubectl apply -f llama-gpu-ing.yaml


## Using Llama 

* The link is now available by running kubectl get ing. Copy and paste the address into your browser and you will be prompted by a Gradio interface that is connected to the EKS pod running Stable Diffusion 2.1. Enter your prompt and an image will be returned. Add /serve at the end of the address to view the interface.
* kubectl get ing


## View GPU Utilization in Real Time 

* If you have the terminal next to the browser with Stable Diffusion open, you can view the GPU utilization in real time by logging into the pod and running nvitop.
    * kubectl exec -it sd-gpu-845c8bdc49-5wdc4 -- bash
    * nvidia-smi 
    * nvitop
    * exit