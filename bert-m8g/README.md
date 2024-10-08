## Prerequisites
* Follow the steps from sd-gpu-readme.md

## Deploy GP Nodepool

* Deploying a nodepool in Kubernetes is a group of nodes within a cluster that have the same configuration. In this nodepool, we will be configuring C7g instances as our main EC2 instance that will power Bert. 
```
cd nodepools
cat gp-nodepool.yaml | envsubst | kubectl apply -f -
```
* New nodepool for this (CPU related). Make sure to use the correct requirements found within this file: https://karpenter.sh/docs/concepts/nodepools/

## Deploy BERT

* This file aims to deploy BERT onto an EKS pod. We will be using the envsubst command which replaces all variables in this file with environment variables, so make sure that the correct variables are set and align with the what will be replaced in the file.
```
cat bert-gp-deploy.yaml | envsubst | kubectl apply -f -
```

## Deploy Service

* We are deploying a service file called bert-gp-svc.yaml focused on exposing an application running in our cluster. We define the service to expose port 80, and the pods to have a targetPort of 8000, meaning that the service will route traffic from port 80 on the service to port 8000 on the pods that match the label app:bert-cpu. 
```
kubectl apply -f bert-gp-svc.yaml
```

## Deploy Ing

* We will be deploying an ingress file called bert-gp-ing.yaml which focuses on exposing HTTP routes from outside the cluster to services within the cluster. 
```
kubectl apply -f bert-gp-ing.yaml
```
