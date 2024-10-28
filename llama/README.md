# Llama 3 8B on EKS with AWS


## Deploy Nodepool

* Deploying a nodepool in Kubernetes is a group of nodes within a cluster that have the same configuration. Refer to the nodepools folder in the root of the directory to choose the specific Nodepool you want to deploy.
```
cat [NODEPOOL FILE].yaml | envsubst | kubectl apply -f -
```

## Deploy Llama

* This file aims to deploy the model onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
```
cat llama-[INSTANCE]-deploy.yaml | envsubst | kubectl apply -f -
```

## Deploy Service

* We are deploying a service file focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:sd-inf. 
```
kubectl apply -f llama-[INSTANCE]-svc.yaml
```

## Deploy Ing

* We will be deploying an ingress file which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
```
kubectl apply -f llama-[INSTANCE]-ing.yaml
```

## Using Llama 

* The link is now available by running kubectl get ing. Copy and paste the address into your browser and you will be prompted by a Gradio interface that is connected to the EKS pod running Llama. Enter your prompt and an image will be returned. Add /serve at the end of the address to view the interface.
```
kubectl get ing
```

## View Utilization in Real Time 

* If you have the terminal next to the browser, you can view the GPU or accelerator utilization in real time.
```
kubectl exec -it [POD NAME] -- bash
nvidia-smi 
nvitop
exit
```

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

* There may be an issue where running these steps can lead to multiple restarts on the llama-inf-deploy.yaml file which tries to deploy the llama pod. It was found that requesting an increased quota for inf2 instances was the fix for this and was noted by an error message within calling the logs for karpenter, indicating that there was insufficient capacity and a VcpuLimitExceeded. Head to the AWS console, Service Quotas dashboard, search for Amazon Elastic Compute Cloud (Amazon EC2) and request for an increase in Running On-Demand Inf instances. In this case, we requested for an increase of 100 quota which equates to an inf2.24xlarge instance and can handle both the sd-inf and llama3-inf pods. 