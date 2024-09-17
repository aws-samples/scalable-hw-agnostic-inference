# Stable Diffusion 2.1 on EKS with AWS Inferentia 

## Prerequisites

The prerequisites to following this guide are to ensure that you have created an EKS cluster with Karpenter installed, as well as the AWS Load Balancer Controller. Both of these steps can be found in the sd-gpu-readme.MD guide. If following the sd-gpu-readme.MD first, make sure that you have these two environment variables set (KarpenterNodeRole and KarpenterDiscoveryTag) on top of the ones set in the readme from serving-container-build folder. 

## Deploy my-scheduler

* We will apply this scheduler file as a general scheduler as EKS does not natively support the neuron scheduler extension, and this will alleviate this issue by creating a generalized scheduler. 
* kubectl apply -f my-scheduler.yml


## Deploy Neuron High Priority

* We run this file to emphasize that this is a high priority workload.
* kubectl apply -f k8s-neuron-high-priority.yml


## Deploy Neuron Scheduler

* This file is required for scheduling pods that require more than one Neuron core or device resource. 
* kubectl apply -f k8s-neuron-scheduler-eks.yml


## Deploy Neuron Device Plugin RBAC (Role based access control)

* Enables permission for neuron device plugin to update the node and pod annotations.
* kubectl apply -f k8s-neuron-device-plugin-rbac.yml


## Deploy Neuron Device Plugin

* We deploy this file as a daemonset and expoeses Neuron cores and devices to kubernetes as a resource. 
* kubectl apply -f k8s-neuron-device-plugin.yml


## Deploy Neuron Nodepool

* Deploying a nodepool in Kubernetes is a group of nodes within a cluster that have the same configuration. 
* kubectl apply -f amd-inf2-nodepool.yaml


## Deploy Stable Diffusion

* This file aims to deploy stable diffusion 2.1 onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
* cat sd-inf-deploy.yaml | envsubst | kubectl apply -f -


## Deploy Service

* We are deploying a service file called sd-inf-svc.yaml focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-inf. 
* kubectl apply -f sd-inf-svc.yaml


## Deploy Ing

* We will be deploying an ingress file called sd-inf-ing.yaml which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
* kubectl apply -f sd-inf-ing.yaml


## Using Stable Diffusion 

* The link is now available by running kubectl get ing. Copy and paste the address into your browser and you will be prompted by a Gradio interface that is connected to the EKS pod running Stable Diffusion 2.1. Enter your prompt and an image will be returned. Add /serve at the end of the address to view the interface.
* kubectl get ing


## View GPU Utilization in Real Time 

* If you have the terminal next to the browser with Stable Diffusion open, you can view the GPU utilization in real time by logging into the pod and running nvitop.
    * kubectl exec -it sd-gpu-845c8bdc49-5wdc4 -- bash
    * neuron-ls
    * neuron-top
    * exit

