# Llama 3 8B on EKS with AWS Inferentia

## Prerequisites

* Follow prerequisites from sd-inf-readme.md, as well as deploying: my-scheduler, neuron high priority, neuron scheduler, neuron device plugin rbac, neuron device plugin, neuron nodepool. If all these steps have been completed, you can then move onto the remaining steps. 

## Deploy Llama

* This file aims to deploy stable diffusion 2.1 onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
```
cat llama-inf-deploy.yaml | envsubst | kubectl apply -f -
```

## Deploy Service

* We are deploying a service file focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-inf. 
```
kubectl apply -f llama-inf-svc.yaml
```

## Deploy Ing

* We will be deploying an ingress file which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
```
kubectl apply -f llama-inf-ing.yaml
```

## Using Llama

* The link is now available by running kubectl get ing. Copy and paste the address into your browser and you will be prompted by a Gradio interface that is connected to the EKS pod running Stable Diffusion 2.1. Enter your prompt and an image will be returned. Add /serve at the end of the address to view the interface.
```
kubectl get ing
```

## View GPU Utilization in Real Time 

* If you have the terminal next to the browser with Stable Diffusion open, you can view the GPU utilization in real time by logging into the pod and running nvitop.
```
kubectl exec -it [POD NAME] -- bash
neuron-ls
neuron-top
exit
```

## Debugging Tips

```
kubectl get po -n kube-system | grep karpenter
kubectl logs -n [NAMESPACE] [POD NAME]
```

* If you followed the sd-inf-readme.md before this inf readme, there may be an issue where running these steps can lead to multiple restarts on the llama-inf-deploy.yaml file which tries to deploy the llama pod. It was found that requesting an increased quota for inf2 instances was the fix for this and was noted by an error message within calling the logs for karpenter, indicating that there was insufficient capacity and a VcpuLimitExceeded. Head to the AWS console, Service Quotas dashboard, search for Amazon Elastic Compute Cloud (Amazon EC2) and request for an increase in Running On-Demand Inf instances. In this case, we requested for an increase of 100 quota which equates to an inf2.24xlarge instance and can handle both the sd-inf and llama3-inf pods. 