"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.PipelineStack = void 0;
const aws_cdk_lib_1 = require("aws-cdk-lib");
const ecr = __importStar(require("aws-cdk-lib/aws-ecr"));
const codebuild = __importStar(require("aws-cdk-lib/aws-codebuild"));
const codepipeline = __importStar(require("aws-cdk-lib/aws-codepipeline"));
const codepipeline_actions = __importStar(require("aws-cdk-lib/aws-codepipeline-actions"));
const iam = __importStar(require("aws-cdk-lib/aws-iam"));
const secretsmanager = __importStar(require("aws-cdk-lib/aws-secretsmanager"));
const cdk = __importStar(require("aws-cdk-lib/core"));
class PipelineStack extends aws_cdk_lib_1.Stack {
    constructor(scope, id, props) {
        super(scope, id, props);
        const BASE_REPO = new aws_cdk_lib_1.CfnParameter(this, "BASEREPO", { type: "String" });
        const BASE_IMAGE_AMD_XLA_TAG = new aws_cdk_lib_1.CfnParameter(this, "BASEIMAGEAMDXLATAG", { type: "String" });
        const BASE_IMAGE_AMD_CUD_TAG = new aws_cdk_lib_1.CfnParameter(this, "BASEIMAGEAMDCUDTAG", { type: "String" });
        const BASE_IMAGE_ARM_CPU_TAG = new aws_cdk_lib_1.CfnParameter(this, "BASEIMAGEARMCPUTAG", { type: "String" });
        const IMAGE_AMD_XLA_TAG = new aws_cdk_lib_1.CfnParameter(this, "IMAGEAMDXLATAG", { type: "String" });
        const IMAGE_AMD_CUD_TAG = new aws_cdk_lib_1.CfnParameter(this, "IMAGEAMDCUDTAG", { type: "String" });
        const IMAGE_ARM_CPU_TAG = new aws_cdk_lib_1.CfnParameter(this, "IMAGEARMCPUTAG", { type: "String" });
        const GITHUB_OAUTH_TOKEN = new aws_cdk_lib_1.CfnParameter(this, "GITHUBOAUTHTOKEN", { type: "String" });
        const GITHUB_USER = new aws_cdk_lib_1.CfnParameter(this, "GITHUBUSER", { type: "String" });
        const GITHUB_REPO = new aws_cdk_lib_1.CfnParameter(this, "GITHUBREPO", { type: "String" });
        const GITHUB_BRANCH = new aws_cdk_lib_1.CfnParameter(this, "GITHUBBRANCH", { type: "String" });
        // uncomment when you test the stack and dont want to manually delete the ecr registry 
        const base_registry = new ecr.Repository(this, `base_repo`, {
            repositoryName: BASE_REPO.valueAsString,
            imageScanOnPush: true
        });
        //const base_registry = ecr.Repository.fromRepositoryName(this,`base_repo`,BASE_REPO.valueAsString)
        //create a roleARN for codebuild 
        const buildRole = new iam.Role(this, 'BaseCodeBuildDeployRole', {
            roleName: 'hwagnisticBaseCodeBuildDeployRole',
            assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
        });
        buildRole.addToPolicy(new iam.PolicyStatement({
            resources: ['*'],
            actions: ['ssm:*', 's3:*'],
        }));
        const githubSecret = new secretsmanager.Secret(this, 'githubSecret', {
            secretObjectValue: {
                token: aws_cdk_lib_1.SecretValue.unsafePlainText(GITHUB_OAUTH_TOKEN.valueAsString)
            },
        });
        const githubOAuthToken = aws_cdk_lib_1.SecretValue.secretsManager(githubSecret.secretArn, { jsonField: 'token' });
        new cdk.CfnOutput(this, 'githubOAuthTokenRuntimeOutput1', {
            //value: SecretValue.secretsManager("githubtoken",{jsonField: "token"}).toString()
            value: githubSecret.secretValueFromJson('token').toString()
        });
        new cdk.CfnOutput(this, 'githubOAuthTokenRuntimeOutput2', {
            value: aws_cdk_lib_1.SecretValue.secretsManager(githubSecret.secretArn, { jsonField: "token" }).toString()
        });
        const base_image_amd_cud_build = new codebuild.Project(this, `ImageCudAmdBuild`, {
            environment: { privileged: true, buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_3 },
            cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),
            role: buildRole,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: "0.2",
                env: {
                    'exported-variables': [
                        'AWS_ACCOUNT_ID', 'AWS_REGION', 'BASE_REPO', 'IMAGE_AMD_CUD_TAG', 'BASE_IMAGE_AMD_CUD_TAG'
                    ],
                },
                phases: {
                    build: {
                        commands: [
                            `export AWS_ACCOUNT_ID="${this.account}"`,
                            `export AWS_REGION="${this.region}"`,
                            `export BASE_REPO="${BASE_REPO.valueAsString}"`,
                            `export IMAGE_TAG="${IMAGE_AMD_CUD_TAG.valueAsString}"`,
                            `export BASE_IMAGE_TAG="${BASE_IMAGE_AMD_CUD_TAG.valueAsString}"`,
                            `cd app`,
                            `chmod +x ./build.sh && ./build.sh`
                        ],
                    }
                },
                artifacts: {
                    files: ['imageDetail.json']
                },
            }),
        });
        const base_image_arm_cpu_build = new codebuild.Project(this, `ImageCpuArmBuild`, {
            environment: { privileged: true, buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_2 },
            cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),
            role: buildRole,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: "0.2",
                env: {
                    'exported-variables': [
                        'AWS_ACCOUNT_ID', 'AWS_REGION', 'BASE_REPO', 'IMAGE_ARM_CPU_TAG', 'BASE_IMAGE_ARM_CPU_TAG'
                    ],
                },
                phases: {
                    build: {
                        commands: [
                            `export AWS_ACCOUNT_ID="${this.account}"`,
                            `export AWS_REGION="${this.region}"`,
                            `export BASE_REPO="${BASE_REPO.valueAsString}"`,
                            `export IMAGE_TAG="${IMAGE_ARM_CPU_TAG.valueAsString}"`,
                            `export BASE_IMAGE_TAG="${BASE_IMAGE_ARM_CPU_TAG.valueAsString}"`,
                            `cd app`,
                            `chmod +x ./build.sh && ./build.sh`
                        ],
                    }
                },
                artifacts: {
                    files: ['imageDetail.json']
                },
            }),
        });
        const base_image_amd_xla_build = new codebuild.Project(this, `ImageXlaAmdBuild`, {
            environment: { privileged: true, buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_3 },
            cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),
            role: buildRole,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: "0.2",
                env: {
                    'exported-variables': [
                        'AWS_ACCOUNT_ID', 'AWS_REGION', 'BASE_REPO', 'IMAGE_AMD_XLA_TAG', 'BASE_IMAGE_AMD_XLA_TAG'
                    ],
                },
                phases: {
                    build: {
                        commands: [
                            `export AWS_ACCOUNT_ID="${this.account}"`,
                            `export AWS_REGION="${this.region}"`,
                            `export BASE_REPO="${BASE_REPO.valueAsString}"`,
                            `export IMAGE_TAG="${IMAGE_AMD_XLA_TAG.valueAsString}"`,
                            `export BASE_IMAGE_TAG="${BASE_IMAGE_AMD_XLA_TAG.valueAsString}"`,
                            `cd app`,
                            `chmod +x ./build.sh && ./build.sh`
                        ],
                    }
                },
                artifacts: {
                    files: ['imageDetail.json']
                },
            }),
        });
        const assets_image_xla_amd_build = new codebuild.Project(this, `AssetsImageXlaAmdBuild`, {
            environment: { privileged: true, buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_3 },
            cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),
            role: buildRole,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: "0.2",
                env: {
                    'exported-variables': [
                        'AWS_ACCOUNT_ID', 'AWS_REGION', 'BASE_REPO', 'IMAGE_AMD_XLA_TAG', 'BASE_IMAGE_AMD_XLA_TAG'
                    ],
                },
                phases: {
                    build: {
                        commands: [
                            `export AWS_ACCOUNT_ID="${this.account}"`,
                            `export AWS_REGION="${this.region}"`,
                            `export BASE_REPO="${BASE_REPO.valueAsString}"`,
                            `export IMAGE_TAG="${IMAGE_AMD_XLA_TAG.valueAsString}"`,
                            `export BASE_IMAGE_TAG="${BASE_IMAGE_AMD_XLA_TAG.valueAsString}"`,
                            `cd app`,
                            `chmod +x ./build-assets.sh && ./build-assets.sh`
                        ],
                    }
                },
                artifacts: {
                    files: ['imageDetail.json']
                },
            }),
        });
        const assets_image_cpu_arm_build = new codebuild.Project(this, `AssetsImageCpuArmBuild`, {
            environment: { privileged: true, buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_2 },
            cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),
            role: buildRole,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: "0.2",
                env: {
                    'exported-variables': [
                        'AWS_ACCOUNT_ID', 'AWS_REGION', 'BASE_REPO', 'IMAGE_ARM_CPU_TAG', 'BASE_IMAGE_ARM_CPU_TAG'
                    ],
                },
                phases: {
                    build: {
                        commands: [
                            `export AWS_ACCOUNT_ID="${this.account}"`,
                            `export AWS_REGION="${this.region}"`,
                            `export BASE_REPO="${BASE_REPO.valueAsString}"`,
                            `export IMAGE_TAG="${IMAGE_ARM_CPU_TAG.valueAsString}"`,
                            `export BASE_IMAGE_TAG="${BASE_IMAGE_ARM_CPU_TAG.valueAsString}"`,
                            `cd app`,
                            `chmod +x ./build-assets.sh && ./build-assets.sh`
                        ],
                    }
                },
                artifacts: {
                    files: ['imageDetail.json']
                },
            }),
        });
        const assets_image_cud_amd_build = new codebuild.Project(this, `AssetsImageCudAmdBuild`, {
            environment: { privileged: true, buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_3 },
            cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),
            role: buildRole,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: "0.2",
                env: {
                    'exported-variables': [
                        'AWS_ACCOUNT_ID', 'AWS_REGION', 'BASE_REPO', 'IMAGE_AMD_CUD_TAG', 'BASE_IMAGE_AMD_CUD_TAG'
                    ],
                },
                phases: {
                    build: {
                        commands: [
                            `export AWS_ACCOUNT_ID="${this.account}"`,
                            `export AWS_REGION="${this.region}"`,
                            `export BASE_REPO="${BASE_REPO.valueAsString}"`,
                            `export IMAGE_TAG="${IMAGE_AMD_CUD_TAG.valueAsString}"`,
                            `export BASE_IMAGE_TAG="${BASE_IMAGE_AMD_CUD_TAG.valueAsString}"`,
                            `cd app`,
                            `chmod +x ./build-assets.sh && ./build-assets.sh`
                        ],
                    }
                },
                artifacts: {
                    files: ['imageDetail.json']
                },
            }),
        });
        //we allow the buildProject principal to push images to ecr
        base_registry.grantPullPush(assets_image_cud_amd_build.grantPrincipal);
        base_registry.grantPullPush(assets_image_xla_amd_build.grantPrincipal);
        base_registry.grantPullPush(assets_image_cpu_arm_build.grantPrincipal);
        base_registry.grantPullPush(base_image_amd_xla_build.grantPrincipal);
        base_registry.grantPullPush(base_image_amd_cud_build.grantPrincipal);
        base_registry.grantPullPush(base_image_arm_cpu_build.grantPrincipal);
        // here we define our pipeline and put together the assembly line
        const sourceOutput = new codepipeline.Artifact();
        const basebuildpipeline = new codepipeline.Pipeline(this, `BuildBasePipeline`);
        basebuildpipeline.addStage({
            stageName: 'Source',
            actions: [
                new codepipeline_actions.GitHubSourceAction({
                    actionName: 'GitHub_Source',
                    owner: GITHUB_USER.valueAsString,
                    repo: GITHUB_REPO.valueAsString,
                    branch: GITHUB_BRANCH.valueAsString,
                    output: sourceOutput,
                    oauthToken: aws_cdk_lib_1.SecretValue.secretsManager("githubtoken", { jsonField: "token" }),
                    trigger: codepipeline_actions.GitHubTrigger.WEBHOOK,
                    //oauthToken: SecretValue.unsafePlainText(GITHUB_OAUTH_TOKEN.valueAsString)
                })
            ]
        });
        basebuildpipeline.addStage({
            stageName: 'ImageBuild',
            actions: [
                new codepipeline_actions.CodeBuildAction({
                    actionName: 'AssetsImageXlaAmdBuild',
                    input: sourceOutput,
                    runOrder: 1,
                    project: assets_image_xla_amd_build
                }),
                new codepipeline_actions.CodeBuildAction({
                    actionName: 'AssetsImageCudAmdBuild',
                    input: sourceOutput,
                    runOrder: 1,
                    project: assets_image_cud_amd_build
                }),
                new codepipeline_actions.CodeBuildAction({
                    actionName: 'AssetsImageCpuArmBuild',
                    input: sourceOutput,
                    runOrder: 1,
                    project: assets_image_cpu_arm_build
                }),
                new codepipeline_actions.CodeBuildAction({
                    actionName: 'BaseImageAmdXlaBuild',
                    input: sourceOutput,
                    runOrder: 2,
                    project: base_image_amd_xla_build
                }),
                new codepipeline_actions.CodeBuildAction({
                    actionName: 'BaseImageArmCpuBuild',
                    input: sourceOutput,
                    runOrder: 2,
                    project: base_image_arm_cpu_build
                }),
                new codepipeline_actions.CodeBuildAction({
                    actionName: 'BaseImageAmdCudBuild',
                    input: sourceOutput,
                    runOrder: 2,
                    project: base_image_amd_cud_build
                })
            ]
        });
    }
}
exports.PipelineStack = PipelineStack;
