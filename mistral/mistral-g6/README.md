# Mistral 7B Instruct v0.3 on EKS with Nvidia GPU's

## Prerequisites

* Ensure that you have a HuggingFace account created and have read and accepted the Mistral 7B Instruct v0.3 community license agreement found here: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3. This will ensure that the model can be downloaded and accessed from the HuggingFace gated repository and we will not face issues with it later on in the process.


## Deploy Mistral

* This file aims to deploy stable diffusion 2.1 onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
```
cat mistral-g6-deploy.yaml | envsubst | kubectl apply -f -
```

## Deploy Service

* We are deploying a service file focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-gpu. 
```
kubectl apply -f mistral-g6-svc.yaml
```

## Deploy Ing

* We will be deploying an ingress file which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
```
kubectl apply -f mistral-g6-ing.yaml
```

## Using Mistral 

* The link is now available by running kubectl get ing. Copy and paste the address into your browser and you will be prompted by a Gradio interface that is connected to the EKS pod running Stable Diffusion 2.1. Enter your prompt and an image will be returned. Add /serve at the end of the address to view the interface.
```
kubectl get ing
```

## View GPU Utilization in Real Time 

* If you have the terminal next to the browser with Stable Diffusion open, you can view the GPU utilization in real time by logging into the pod and running nvitop.
```
kubectl exec -it [POD NAME] -- bash
nvidia-smi 
nvitop
exit
```