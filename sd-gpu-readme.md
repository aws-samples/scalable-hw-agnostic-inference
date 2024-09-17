# Stable Diffusion 2.1 on EKS with Nvidia GPU's

## Create an EKS Cluster with Karpenter

* The first step is to create an EKS cluster with Karpenter installed. Follow the Karpenter docs to do so. 


## Set Karpenter Environment Variables

* These variables are created after following the EKS Cluster creation with Karpenter step. We will be exporting these variables to our terminal using the export command, export KarpenterNodeRole=[Fill in Here] and export KarpenterDiscoveryTag=[Fill in Here]. 
* Find the KarpenterNodeRole in the IAM roles console, it will follow a format similar to: KarpenterNodeRole-jon-karpenter-demo. The correct role will be KarpenterNodeRole-[EKS Cluster Name].
* To find the KarpenterDiscoveryTag, either go to EC2 console, security groups, and filter through the tag with the key: karpenter.sh/discovery. Click on the right security group, then go to tags. OR go to VPC, subets, filter through the tag with the key: karpenter.sh/discovery. It will look something like jon-karpenter-demo as the value. 


## Install AWS Load Balancer Controller

* You can use the load balancer controller to expose your EKS apps to the internet. We will be creating a target group for the stable diffusion pod which we will cover in a later section within sd-gpu-svc.yaml. Follow this guide to install the Load Balancer Controller.


## Deploy NVIDIA Device Plugin

* We deploy the NVIDIA device plugin to enable you to schedule container workloads and it also manages GPUs as a resources. 
```bash
kubectl apply -f nvidia-device-plugin-daemonset.yaml
```

## Deploy Nvidia Nodepool

* Deploying a nodepool in Kubernetes is a group of nodes within a cluster that have the same configuration. 
```bash
kubectl apply -f amd-nvidia-nodepool.yaml
```


## Deploy Stable Diffusion

* This file aims to deploy stable diffusion 2.1 onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
```bash
cat sd-gpu-deply.yaml | envsubst | kubectl apply -f -
```

## Deploy Service

* We are deploying a service file called sd-gpu-svc.yaml focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-gpu. 
```bash
kubectl apply -f sd-gpu-svc.yaml
```


## Deploy Ing

* We will be deploying an ingress file called sd-gpu-ing.yaml which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
```bash
kubectl apply -f sd-gpu-ing.yaml
```


## Using Stable Diffusion 

* The link is now available by running kubectl get ing. Copy and paste the address into your browser and you will be prompted by a Gradio interface that is connected to the EKS pod running Stable Diffusion 2.1. Enter your prompt and an image will be returned. Add /serve at the end of the address to view the interface.
```bash
kubectl get ing
```


## View GPU Utilization in Real Time 

* If you have the terminal next to the browser with Stable Diffusion open, you can view the GPU utilization in real time by logging into the pod and running nvitop.
```bash
kubectl exec -it sd-gpu-845c8bdc49-5wdc4 -- bash
nvidia-smi 
nvitop
exit
```


## Debugging Commands
```bash
kubectl get nodepool
kubectl get pods
kubectl get pod -A
kubectl logs -n [NAMESPACE] [POD NAME]
kubectl describe pod [POD NAME]
```
### Issues
    * If there is an authorization issue where a role is not allowed to execute a particular command, you can find the role in the IAM roles console and add an inline policy to mitigate it. 
    * One such issue may arise in the application load balancer where you will need to add an inline policy to that specific IAM role. Delete the existing load balancer and it will reload, along with the updated IAM role.
    * If the address for accessing stable diffusion returns a blank screen, even when adding /serve at the end, we should check the target group in the ALB console. If the health status is unhealthy, but we are able to confirm that the pod is healthy by logging into the pod with the `kubectl exec -it [POD NAME] — bash` command, and running `curl 127.0.0.1:8000/health` , and it returns that the pod is healthy, then we should check the security group for the cluster that ends with ClusterSharedNodeSecurityGroup and add an inbound rule for Custom TCP, port range 8000, from anywhere 0.0.0.0/0. Try accessing the address/serve again. 

