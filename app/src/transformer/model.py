import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, \
    PixArtAlphaTextProjection
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers \
    import ColumnParallelLinear, RowParallelLinear

DTYPE=torch.bfloat16

class TracingTransformerEmbedderWrapper(nn.Module):
    def __init__(
            self,
            x_embedder,
            context_embedder,
            time_text_embed,
            pos_embed):
        super().__init__()
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.time_text_embed = time_text_embed
        self.pos_embed = pos_embed

    def forward(
            self,
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            txt_ids,
            img_ids):

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)
        return hidden_states, temb, image_rotary_emb


class TracingTransformerBlockWrapper(nn.Module):
    def __init__(self, transformer, transformerblock):
        super().__init__()
        self.transformerblock = transformerblock
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb):
        for block in self.transformerblock:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )
        return encoder_hidden_states, hidden_states


class TracingSingleTransformerBlockWrapper(nn.Module):
    def __init__(self, transformer, transformerblock):
        super().__init__()
        self.transformerblock = transformerblock
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(self, hidden_states, temb, image_rotary_emb):
        for block in self.transformerblock:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )
        return hidden_states


class TracingTransformerOutLayerWrapper(nn.Module):
    def __init__(self, norm_out, proj_out):
        super().__init__()
        self.norm_out = norm_out
        self.proj_out = proj_out

    def forward(self, hidden_states, encoder_hidden_states, temb):
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        return (self.proj_out(hidden_states),)


class TracingSingleTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_hidden_dim = None

        self.norm = None
        self.proj_mlp = None
        self.act_mlp = None
        self.proj_out = None
        self.proj_out_2 = None

        self.attn = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        gate = gate.unsqueeze(1)
        hidden_states = gate * (self.proj_out(attn_output)
                                + self.proj_out_2(mlp_hidden_states))
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


def get_sharded_data(data, dim):
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    per_partition_size = \
        data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[
            per_partition_size * tp_rank: per_partition_size * (tp_rank + 1)
            ].clone().to(torch.bfloat16)
    elif dim == 1:
        return data[:,
                    per_partition_size * tp_rank: per_partition_size *
                    (tp_rank + 1)
                    ].clone().to(torch.bfloat16)
    else:
        raise Exception(
            f"Partiton value of 0,1 are supported, found {dim}."
            )


def shard_attn(attn: Attention):
    attn.heads = 3

    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        attn.to_q.out_features,
        bias=(attn.to_q.bias is not None),
        gather_output=False)
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if attn.to_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del (orig_q)

    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        attn.to_k.out_features,
        bias=(attn.to_k.bias is not None),
        gather_output=False)
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if attn.to_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del (orig_k)

    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        attn.to_v.out_features,
        bias=(attn.to_v.bias is not None),
        gather_output=False)
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if attn.to_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del (orig_v)

    orig_q_proj = attn.add_q_proj
    attn.add_q_proj = ColumnParallelLinear(
        attn.add_q_proj.in_features,
        attn.add_q_proj.out_features,
        bias=(attn.add_q_proj.bias is not None),
        gather_output=False)
    attn.add_q_proj.weight.data = get_sharded_data(orig_q_proj.weight.data, 0)
    if attn.add_q_proj.bias is not None:
        attn.add_q_proj.bias.data = get_sharded_data(orig_q_proj.bias.data, 0)
    del (orig_q_proj)

    orig_k_proj = attn.add_k_proj
    attn.add_k_proj = ColumnParallelLinear(
        attn.add_k_proj.in_features,
        attn.add_k_proj.out_features,
        bias=(attn.add_k_proj.bias is not None),
        gather_output=False)
    attn.add_k_proj.weight.data = get_sharded_data(orig_k_proj.weight.data, 0)
    if attn.add_k_proj.bias is not None:
        attn.add_k_proj.bias.data = get_sharded_data(orig_k_proj.bias.data, 0)
    del (orig_k_proj)

    orig_v_proj = attn.add_v_proj
    attn.add_v_proj = ColumnParallelLinear(
        attn.add_v_proj.in_features,
        attn.add_v_proj.out_features,
        bias=(attn.add_v_proj.bias is not None),
        gather_output=False)
    attn.add_v_proj.weight.data = get_sharded_data(orig_v_proj.weight.data, 0)
    if attn.add_v_proj.bias is not None:
        attn.add_v_proj.bias.data = get_sharded_data(orig_v_proj.bias.data, 0)
    del (orig_v_proj)

    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        attn.to_out[0].in_features,
        attn.to_out[0].out_features,
        bias=(attn.to_out[0].bias is not None),
        input_is_parallel=True)
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if attn.to_out[0].bias is not None:
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del (orig_out)

    orig_out = attn.to_add_out
    attn.to_add_out = RowParallelLinear(
        attn.to_add_out.in_features,
        attn.to_add_out.out_features,
        bias=(attn.to_add_out.bias is not None),
        input_is_parallel=True)
    attn.to_add_out.weight.data = get_sharded_data(orig_out.weight.data, 1)
    if attn.to_add_out.bias is not None:
        attn.to_add_out.bias.data = orig_out.bias.data.detach()
    del (orig_out)
    return attn


def shard_attn_lite(block):
    attn = block.attn
    attn.heads = 3

    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        attn.to_q.out_features,
        bias=(attn.to_q.bias is not None),
        gather_output=False)
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if attn.to_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del (orig_q)

    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        attn.to_k.out_features,
        bias=(attn.to_k.bias is not None),
        gather_output=False)
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if attn.to_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del (orig_k)

    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        attn.to_v.out_features,
        bias=(attn.to_v.bias is not None),
        gather_output=False)
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if attn.to_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del (orig_v)

    orig_mlp = block.proj_mlp
    block.proj_mlp = ColumnParallelLinear(
        block.proj_mlp.in_features,
        block.proj_mlp.out_features,
        bias=(block.proj_mlp.bias is not None),
        gather_output=False)
    block.proj_mlp.weight.data = get_sharded_data(orig_mlp.weight.data, 0)
    if block.proj_mlp.bias is not None:
        block.proj_mlp.bias.data = get_sharded_data(orig_mlp.bias.data, 0)
    del (orig_mlp)

    orig_out = block.proj_out
    out_features = block.proj_out.out_features
    bias = block.proj_out.bias
    block.proj_out = RowParallelLinear(
        3072,
        out_features,
        bias=(bias is not None),
        input_is_parallel=True)
    block.proj_out.weight.data = get_sharded_data(
        orig_out.weight.data[..., 0:3072], 1)
    if block.proj_out.bias is not None:
        block.proj_out.bias.data = orig_out.bias.data.detach()

    block.proj_out_2 = RowParallelLinear(
        12288,
        out_features,
        bias=False,
        input_is_parallel=True)
    block.proj_out_2.weight.data = get_sharded_data(
        orig_out.weight.data[..., 3072:15360], 1)
    del (orig_out)

    return attn


def shard_ff(ff: FeedForward) -> FeedForward:
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        ff.net[0].proj.in_features,
        ff.net[0].proj.out_features,
        bias=(ff.net[0].proj.bias is not None),
        gather_output=False)
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if ff.net[0].proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del (orig_proj)
    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        ff.net[2].in_features,
        ff.net[2].out_features,
        bias=(ff.net[2].bias is not None),
        input_is_parallel=True)
    if ff.net[2].bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    del (orig_linear)
    return ff


def init_transformer(transformer):
    timestep_embedder: TimestepEmbedding = \
        transformer.time_text_embed.timestep_embedder
    orig_linear_1 = timestep_embedder.linear_1
    timestep_embedder.linear_1 = ColumnParallelLinear(
        timestep_embedder.linear_1.in_features,
        timestep_embedder.linear_1.out_features,
        bias=(timestep_embedder.linear_1.bias is not None),
        gather_output=False)
    timestep_embedder.linear_1.weight.data = \
        get_sharded_data(orig_linear_1.weight.data, 0)
    if timestep_embedder.linear_1.bias is not None:
        timestep_embedder.linear_1.bias.data = \
            get_sharded_data(orig_linear_1.bias.data, 0)
    del (orig_linear_1)
    orig_linear_2 = timestep_embedder.linear_2
    timestep_embedder.linear_2 = RowParallelLinear(
        timestep_embedder.linear_2.in_features,
        timestep_embedder.linear_2.out_features,
        bias=(timestep_embedder.linear_2.bias is not None),
        input_is_parallel=True)
    if timestep_embedder.linear_2.bias is not None:
        timestep_embedder.linear_2.bias.data = orig_linear_2.bias.data.detach()
    timestep_embedder.linear_2.weight.data = \
        get_sharded_data(orig_linear_2.weight.data, 1)
    del (orig_linear_2)

    guidance_embedder: TimestepEmbedding = \
        transformer.time_text_embed.guidance_embedder
    orig_linear_1 = guidance_embedder.linear_1
    guidance_embedder.linear_1 = ColumnParallelLinear(
        guidance_embedder.linear_1.in_features,
        guidance_embedder.linear_1.out_features,
        bias=(guidance_embedder.linear_1.bias is not None),
        gather_output=False)
    guidance_embedder.linear_1.weight.data = \
        get_sharded_data(orig_linear_1.weight.data, 0)
    if guidance_embedder.linear_1.bias is not None:
        guidance_embedder.linear_1.bias.data = \
            get_sharded_data(orig_linear_1.bias.data, 0)
    del (orig_linear_1)
    orig_linear_2 = guidance_embedder.linear_2
    guidance_embedder.linear_2 = RowParallelLinear(
        guidance_embedder.linear_2.in_features,
        guidance_embedder.linear_2.out_features,
        bias=(guidance_embedder.linear_2.bias is not None),
        input_is_parallel=True)
    if guidance_embedder.linear_2.bias is not None:
        guidance_embedder.linear_2.bias.data = orig_linear_2.bias.data.detach()
    guidance_embedder.linear_2.weight.data = \
        get_sharded_data(orig_linear_2.weight.data, 1)
    del (orig_linear_2)

    text_embedder: PixArtAlphaTextProjection = \
        transformer.time_text_embed.text_embedder
    orig_linear_1 = text_embedder.linear_1
    text_embedder.linear_1 = ColumnParallelLinear(
        text_embedder.linear_1.in_features,
        text_embedder.linear_1.out_features,
        bias=(text_embedder.linear_1.bias is not None),
        gather_output=False)
    text_embedder.linear_1.weight.data = \
        get_sharded_data(orig_linear_1.weight.data, 0)
    if text_embedder.linear_1.bias is not None:
        text_embedder.linear_1.bias.data = \
            get_sharded_data(orig_linear_1.bias.data, 0)
    del (orig_linear_1)
    orig_linear_2 = text_embedder.linear_2
    text_embedder.linear_2 = RowParallelLinear(
        text_embedder.linear_2.in_features,
        text_embedder.linear_2.out_features,
        bias=(text_embedder.linear_2.bias is not None),
        input_is_parallel=True)
    if text_embedder.linear_2.bias is not None:
        text_embedder.linear_2.bias.data = orig_linear_2.bias.data.detach()
    text_embedder.linear_2.weight.data = \
        get_sharded_data(orig_linear_2.weight.data, 1)
    del (orig_linear_2)

    for block_idx, block in enumerate(transformer.transformer_blocks):
        block.attn = shard_attn(block.attn)
        block.ff = shard_ff(block.ff)
        block.ff_context = shard_ff(block.ff_context)

    for block_idx, block in enumerate(transformer.single_transformer_blocks):
        newblock = TracingSingleTransformerBlock()
        newblock.mlp_hidden_dim = block.mlp_hidden_dim
        newblock.norm = block.norm
        newblock.proj_mlp = block.proj_mlp
        newblock.act_mlp = block.act_mlp
        newblock.proj_out = block.proj_out
        newblock.proj_out_2 = None
        newblock.attn = block.attn
        transformer.single_transformer_blocks[block_idx] = newblock
        block = newblock
        block.attn = shard_attn_lite(block)

