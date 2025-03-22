import os
from huggingface_hub import login
hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
max_model_len=os.environ['MAX_MODEL_LEN']
max_num_seqs=os.environ['MAX_NUM_SEQS']
tensor_parallel_size=os.environ['TENSOR_PARALLEL_SIZE']
model_name=os.environ['MODEL_NAME']
compiled_model_name=os.environ['COMPILED_MODEL_NAME']
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_COMPILED_ARTIFACTS']=model_name


login(hf_token,add_to_git_credential=True)

from vllm import LLM, SamplingParams
llm = LLM(
    model=model_name,
    max_num_seqs=max_num_seqs,
    max_model_len=max_model_len,
    device="neuron",
    override_neuron_config={},
    tensor_parallel_size=tensor_parallel_size)

prompts = [
    "The president of the United States is",
]
sampling_params = SamplingParams(top_k=10, temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
