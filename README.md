# scalable-hw-agnostic-inference

The growing interest in gen-AI applications is driving demand for compute accelerators, resulting in increased inference costs and a shortage of compute capacity. This has led to the introduction of more compute accelerator options, such as NVIDIA and Inferentia. However, each option introduces novel methods for running AI applications on the compute accelerator and requires different code implementations, such as Neuron and CUDA SDKs. Here, we present methods to reduce AI accelerator and CPU costs and minimize compute capacity constraints. 

We explore the benefits of NVIDIA-based accelerators A10G (G5), L4(G6), A100(P4), and AWS Inferentia (inf2) instances suitable for inference of a fun model that produces unique photorealistic images from text and image prompts, Stable Diffusion.

This example uses PyTorch and demonstrates the steps required to compile the model and store it on HuggingFace. We then build, deploy, and run the model on all accelerators using AWS Deep Learning Containers.

Additionally, we show how to scale K8s deployment size based on critical metrics published to CloudWatch, such as inference latency and throughput, with KEDA. Finally, we demonstrate how Karpenter schedules the optimal accelerator instance that meets price and performance requirements.

### Build-time
To construct an accelerator-agnostic inference service, the application processing unit must be packaged to support various accelerators and invoked dynamically based on the accelerator it operates on. Our build process incorporates accelerator-specific software packages such as optimum.neuron and diffusers tailored for NVIDIA and AWS Inferentia.

The initial build process entails configuring the K8s node using the EKS Kubernetes Worker AMI for Machine Learning Accelerated Workloads on Amazon Linux 2 image. This image contains essential components like Kubelet, IAM authenticator, containerd, NVIDIA, and Neuron drivers, along with kernel-relevant headers.

Subsequently, we utilize AWS Deep Learning Containers (DLC) to conceal the AI chip specialized SDK atop the K8s node. These preconfigured Docker images are thoroughly tested with the latest NVIDIA and Neuron deep learning frameworks. During the build process, the relevant DLC is pulled for each variant and augmented with the inference application-specific code, such as computer vision models (Step 2). These customized images are then pushed to a single ECR repository and adorned with the supported accelerator and AI frameworks (Step 3).

Following this, the process assembles the model on the designated AI chip by deploying a K8S job. This action triggers Karpenter to launch the appropriate EC2 instance, such as G5, G6, P4, or Inf2, using the EKS Kubernetes Worker AMI for Machine Learning (Step 4). The job subsequently invokes the specific DLC, compiles the model graphs, and uploads them to HuggingFace (Step 5).

![alt text](/aws-gpu-neuron-eks-sample-model-build.png)
### Deploy-time
![alt text](/aws-gpu-neuron-eks-sample-model-deploy.png)
### Run-time
![alt text](/aws-gpu-neuron-eks-sample-model-run.png)
