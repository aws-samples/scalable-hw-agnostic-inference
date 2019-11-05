# EKS Quickstart Game-Server Sample

This repo contains an initial set of cluster components for deploying containerized game-server to be installed and
configured by [eksctl](https://eksctl.io) through GitOps. It was build based on the GitOps [tutorial](https://eksctl.io/usage/experimental/gitops-flux/#creating-your-own-quick-start-profile)

## Components

- game server deployment with its rbac and reginal config map under [game-server](./game-server)
- [Cluster autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler) -- to [automatically add/remove nodes](https://aws.amazon.com/premiumsupport/knowledge-center/eks-cluster-autoscaler-setup/) to/from your cluster based on its usage.
- [Prometheus](https://prometheus.io/) (its [Alertmanager](https://prometheus.io/docs/alerting/alertmanager/), its [operator](https://github.com/coreos/prometheus-operator), its [`node-exporter`](https://github.com/prometheus/node_exporter), [`kube-state-metrics`](https://github.com/kubernetes/kube-state-metrics), and [`metrics-server`](https://github.com/kubernetes-incubator/metrics-server)) -- for powerful metrics & alerts.
- [Grafana](https://grafana.com) -- for a rich way to visualize metrics via dashboards you can create, explore, and share.
- [Kubernetes dashboard](https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/) -- Kubernetes' standard dashboard.
- [Fluentd](https://www.fluentd.org/) & Amazon's [CloudWatch agent](https://aws.amazon.com/cloudwatch/) -- for cluster & containers' [log collection, aggregation & analytics in CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Container-Insights-setup-logs.html).

## Pre-requisites

A running EKS cluster with [IAM policies](https://eksctl.io/usage/iam-policies/) for:

- The game-server deployment assumed an image is deployed to ECR. The game-server pipeline is defined in [containerized-game-servers](https://github.com/aws-samples/containerized-game-servers)
- The game-server instance uses host-network:true so no ingress controller or LB is needed. 
- The game-server pods need permisions to publish its status to an SQS queue
- auto-scaler
- CloudWatch

[Here](https://github.com/weaveworks/eksctl/blob/master/examples/eks-quickstart-app-dev.yaml) is a sample `ClusterConfig` manifest that shows how to enable these policies.

**N.B.**: policies are configured at node group level.
Therefore, depending on your use-case, you may want to:

- add these policies to all node groups,
- add [node selectors](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/) to the ALB ingress, auto-scaler and CloudWatch pods, so that they are deployed on the nodes configured with these policies.

## How to deploy the template?

- Populate the cluster name by replacing `{{.ClusterName}}`
- Populate the region name by replacing `{{.Region}}` e.g. `us-west-2`
- This example does not use Helm, hence `--with-helm=false`
- The last argument is the profile/template repo, e.g., `git@github.com:yahavb/game-server-gitops-profile.git`
- the `--git-url` is the destination git repo that the sys/devops will use to manage the cluster i.e. editing, adding, removing files to induce changes in the cluster. 

```
export EKSCTL_EXPERIMENTAL=true
eksctl enable profile -r `{{.Region}}` --with-helm=false \
--git-url git@github.com:yahavb/weave-workshop.git \
--git-email email@me.com --cluster {{.ClusterName}} \
git@github.com:aws-samples/amazon-eks-profile-for-gameserver.git
```

## How to access workloads

The game-server pod runs on the ephermal port range over UDP. It is required to configure a security group that allows the access to the game-servers port ranges. 

