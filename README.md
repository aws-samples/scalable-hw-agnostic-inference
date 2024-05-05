# scalable-hw-agnostic-inference

The growing interest in gen-AI applications is driving demand for compute accelerators, resulting in increased inference costs and a shortage of compute capacity. This has led to the introduction of more compute accelerator options, such as NVIDIA and Inferentia. However, each option introduces novel methods for running AI applications on the compute accelerator and requires different code implementations, such as Neuron and CUDA SDKs. Here, we present methods to reduce AI accelerator and CPU costs and minimize compute capacity constraints. 

We explore the benefits of NVIDIA-based accelerators A10G (G5), L4(G6), A100(P4), and AWS Inferentia (inf2) instances suitable for inference of a fun model that produces unique photorealistic images from text and image prompts, Stable Diffusion.

This example uses PyTorch and demonstrates the steps required to compile the model and store it on HuggingFace. We then build, deploy, and run the model on all accelerators using AWS Deep Learning Containers.

Additionally, we show how to scale K8s deployment size based on critical metrics published to CloudWatch, such as inference latency and throughput, with KEDA. Finally, we demonstrate how Karpenter schedules the optimal accelerator instance that meets price and performance requirements.

### Build-time
![alt text](/aws-gpu-neuron-eks-sample-model-build.png)
### Deploy-time
![alt text](/aws-gpu-neuron-eks-sample-model-deploy.png)
### Run-time
![alt text](/aws-gpu-neuron-eks-sample-model-run.png)
