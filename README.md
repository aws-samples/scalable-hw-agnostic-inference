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

The deployment phase (scheduling time) involves deploying Kubernetes constructs such as pods, services, ingress, and node pools (Step 1). We configure a Kubernetes deployment per compute accelerator because the OCI image is built on accelerator-specific DLC. Thus, we configure a Karpenter node pool per compute accelerator to launch Kubernetes nodes with the compute-specific accelerator.

The HW accelerator requires advertising its capabilities, such as accelerator cores (e.g., nvidia.com/gpu or aws.amazon.com/neuron), to enable Karpenter to right-size the EC2 instance it will launch (Step 2). Therefore, we deploy daemon sets, namely nvidia-device-plugin and neuron-device-plugin, to allow Kubernetes to discover and utilize NVIDIA GPU and Inferentia Neuron resources available on a node. These plugins enable Kubernetes to schedule GPU and Inferentia workloads efficiently by providing visibility into available device resources. They also allow them to be allocated to pods that require acceleration.

The NVIDA Karpenter nodepool we allow `g5` and `g6` instances that powers the NVIDIA A10G and L4 core. Once the pod is scheduled on a node, the PyTorch code is invoked, initiating the HuggingFace pipeline customized for the accelerator it runs on. For instance, NeuronStableDiffusionPipeline for Inferentia and StableDiffusionPipeline for GPU. Subsequently, it retrieves the appropriate pre-trained compiled model from HuggingFace and initiates the inference endpoint (Step 3).
```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: amd-nvidia
spec:
  template:
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["g5","g6"]
```
Similarly, in the AWS Inferntia Karpenter nodepool we allow the `inf1` and `inf2` instances.
```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: amd-neuron
spec:
  template:
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["inf1","inf2"]
```

We use an Application Load Balancer (ALB)-based Ingress to control traffic flowing to GPU-based and Inferentia-based node pools. We created an Application Load Balancer using the AWS Load Balancer Controller to distribute incoming requests among the different pods. It controls the traffic routing in the ingress by adding an ingress.kubernetes.io/actions.weighted-routing annotation. You can adjust the weight in the example below to meet your needs. Note, `inf-svc` and `gpu-svc` denotes the k8s services that masks the Inferentia and GPU k8s pods (Step 4).  
```json
    alb.ingress.kubernetes.io/actions.forward-multiple-tg: >
      {
       "type":"forward","forwardConfig":
       {
         "targetGroups":[
           {
             "serviceName":"inf-svc",
             "servicePort":80,
             "weight":50
           },
           {
             "serviceName":"gpu-svc",
             "servicePort":80,
             "weight":50
           }
         ],
         "targetGroupStickinessConfig":
           {
             "enabled":true,
             "durationSeconds":200
           }
       }
     }
```

Additionally, it may take some time for the pods to launch into the model pipeline. It is essential to build readiness and health probes that inform the ALB which pods to target and which pods to recycle if they fail. Therefore we implemented a ping-like gRPC API returns the status of a model in the ModelServer, similar to [pytorch Health check API](https://pytorch.org/serve/inference_api.html#health-check-api). We use it to define the pod inference health and readiness to accept predictions requests. 
First, we define it in the pod specification:
```yaml
      containers:
      - name: app
      ...
       readinessProbe:
          httpGet:
            path: /readiness
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
```

Second, we define it in the ALB ingress:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gradio-mix-ing
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '10'
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: '9'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '10'
    alb.ingress.kubernetes.io/success-codes: '200-301'
```

![alt text](/aws-gpu-neuron-eks-sample-model-deploy.png)
### Run-time
Now that we have enabled the model inference endpoint on both GPU and Inferentia instances, our next steps involve scaling the compute capacity to meet user demand and fine-tuning system performance. To achieve this, we utilize Amazon CloudWatch Container Insights with Enhanced Observability for EKS. This service automatically discovers critical health metrics from AWS accelerators such as Inferentia and NVIDIA GPUs. By leveraging Container Insights dashboards, we can visualize these pre-configured metrics, allowing us to monitor the accelerated infrastructure and optimize workload usage effectively.

Additionally, we use KEDA, a Kubernetes Event-driven Autoscaling tool, to manage the number of required pods. With KEDA, users can define metrics and thresholds, enabling automatic scaling based on workload demands. In our case, we prioritize optimizing for latency in the inference process. 
![alt text](/aws-gpu-neuron-eks-sample-model-run.png)

Therefore, we configure KEDA to trigger additional pods when throughput fails to exceed 200 RPM for GPUs and 300 RPM for Inferentia. This is becasue the Neuron core utilization saturates at 300 RPM for Inferentia and 200 RPM for GPUs (`targetMetricValue`). We also maintained minimal capacity of 2 pods (`minReplicaCount`. Below is the GPU `ScaledObject` specification:
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: stable-diffusion-gpu-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: stable-diffusion-gpu
  minReplicaCount: 2
  triggers:
    - type: aws-cloudwatch
      metadata:
        namespace: AWS/ApplicationELB
        expression: SELECT SUM(HTTPCode_Target_2XX_Count) 
                    FROM SCHEMA("AWS/ApplicationELB", LoadBalancer,TargetGroup) 
                    WHERE TargetGroup = 'targetgroup/tgname/guid' 
                    AND LoadBalancer = 'app/albname/guid'
        metricName: HTTPCode_Target_2XX_Count
        minMetricValue: "2"
        targetMetricValue: "300"
        metricUnit: Count
        awsRegion: us-west-2
```

![alt_text](/aws-gpu-neuron-eks-sample-perf.png)
![alt_text](/aws-gpu-neuron-eks-sample-perf2.png)
