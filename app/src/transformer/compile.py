import argparse
import copy
import neuronx_distributed
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux \
    import FluxTransformer2DModel
from model import (TracingTransformerEmbedderWrapper,
                   TracingTransformerBlockWrapper,
                   TracingSingleTransformerBlockWrapper,
                   TracingTransformerOutLayerWrapper,
                   init_transformer)
from huggingface_hub import login
from huggingface_hub import whoami
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
try:
  user_info = whoami()
  print(f"Already logged in as {user_info['name']}")
except:
  login(hf_token,add_to_git_credential=True)

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
TP_DEGREE=8


def trace_transformer_embedders():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingTransformerEmbedderWrapper(
        transformer.x_embedder, transformer.context_embedder,
        transformer.time_text_embed, transformer.pos_embed)
    return mod_pipe_transformer_f, {}


def trace_transformer_blocks():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingTransformerBlockWrapper(
        transformer, transformer.transformer_blocks)
    return mod_pipe_transformer_f, {}


def trace_single_transformer_blocks():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingSingleTransformerBlockWrapper(
        transformer, transformer.single_transformer_blocks)
    return mod_pipe_transformer_f, {}


def trace_transformer_out_layers():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    init_transformer(transformer)

    mod_pipe_transformer_f = TracingTransformerOutLayerWrapper(
        transformer.norm_out, transformer.proj_out)
    return mod_pipe_transformer_f, {}


def trace_transformer(height, width, max_sequence_length):
    hidden_states = torch.rand([1, height * width // 256, 3072],
                               dtype=torch.bfloat16)
    timestep = torch.rand([1], dtype=torch.bfloat16)
    guidance = torch.rand([1], dtype=torch.float32)
    pooled_projections = torch.rand([1, 768], dtype=torch.bfloat16)
    txt_ids = torch.rand([1, max_sequence_length, 3], dtype=torch.bfloat16)
    img_ids = torch.rand([1, height * width // 256, 3], dtype=torch.bfloat16)
    sample_inputs = hidden_states, timestep, guidance, pooled_projections, \
        txt_ids, img_ids

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_transformer_embedders,
        sample_inputs,
        tp_degree=TP_DEGREE,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args="""--model-type=unet-inference"""
    )

    torch_neuronx.async_load(model)

    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/embedders')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model

    hidden_states = torch.rand([1, height * width // 256, 3072],
                               dtype=torch.bfloat16)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],
                                       dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    image_rotary_emb = torch.rand(
        [1, 1, height * width // 256 + max_sequence_length, 64, 2, 2],
        dtype=torch.bfloat16)
    sample_inputs = hidden_states, encoder_hidden_states, \
        temb, image_rotary_emb

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_transformer_blocks,
        sample_inputs,
        tp_degree=TP_DEGREE,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args="""--model-type=unet-inference"""
    )

    torch_neuronx.async_load(model)

    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/transformer_blocks')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model

    hidden_states = torch.rand(
        [1, height * width // 256 + max_sequence_length, 3072],
        dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    image_rotary_emb = torch.rand(
        [1, 1, height * width // 256 + max_sequence_length, 64, 2, 2],
        dtype=torch.bfloat16)
    sample_inputs = hidden_states, temb, image_rotary_emb

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_single_transformer_blocks,
        sample_inputs,
        tp_degree=TP_DEGREE,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args="""--model-type=unet-inference"""
    )

    torch_neuronx.async_load(model)

    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/single_transformer_blocks')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model

    hidden_states = torch.rand(
        [1, height * width // 256 + max_sequence_length, 3072],
        dtype=torch.bfloat16)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],
                                       dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    sample_inputs = hidden_states, encoder_hidden_states, temb

    model = neuronx_distributed.trace.parallel_model_trace(
        trace_transformer_out_layers,
        sample_inputs,
        tp_degree=TP_DEGREE,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args="""--model-type=unet-inference"""
    )

    torch_neuronx.async_load(model)

    model_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                  'compiled_model/out_layers')
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hh",
        "--height",
        type=int,
        default=1024,
        help="height of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1024,
        help="width of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=512,
        help="maximum sequence length for the text embeddings"
    )
    args = parser.parse_args()
    trace_transformer(
        args.height,
        args.width,
        args.max_sequence_length)

