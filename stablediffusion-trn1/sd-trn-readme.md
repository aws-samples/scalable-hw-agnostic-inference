## Deploy Neuron Nodepool

* Deploying a nodepool in Kubernetes is a group of nodes within a cluster that have the same configuration. 
```
cat amd-trn-nodepool.yaml | envsubst | kubectl apply -f -
```

## Deploy Stable Diffusion

* This file aims to deploy stable diffusion 2.1 onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
```
cat sd-trn-deploy.yaml | envsubst | kubectl apply -f -
```

## Deploy Service

* We are deploying a service file called sd-inf-svc.yaml focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-inf. 
```
kubectl apply -f sd-trn-svc.yaml
```

## Deploy Ing

* We will be deploying an ingress file called sd-inf-ing.yaml which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
```
kubectl apply -f sd-trn-ing.yaml
```

## Using Stable Diffusion 

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