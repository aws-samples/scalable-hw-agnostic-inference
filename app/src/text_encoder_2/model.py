import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Union, Tuple

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
DTYPE=torch.bfloat16

def get_sharded_data(data: torch.Tensor, dim: int) -> torch.Tensor:
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    per_partition_size = data.shape[dim] // tp_size

    if dim == 0:
        return (
            data[per_partition_size * tp_rank : per_partition_size * (tp_rank + 1)]
            .clone()
            .to(DTYPE)
        )
    elif dim == 1:
        return (
            data[:, per_partition_size * tp_rank : per_partition_size * (tp_rank + 1)]
            .clone()
            .to(DTYPE)
        )
    else:
        raise ValueError("Partition dimension must be 0 or 1.")

def shard_t5_attention(t5_attention):
    # Shard q
    orig_q = t5_attention.q
    t5_attention.q = ColumnParallelLinear(
        orig_q.in_features,
        orig_q.out_features,
        bias=(orig_q.bias is not None),
        gather_output=True
    )
    t5_attention.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    # T5 uses bias=False by default, but we handle it just in case
    if orig_q.bias is not None:
        t5_attention.q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    # Shard k
    orig_k = t5_attention.k
    t5_attention.k = ColumnParallelLinear(
        orig_k.in_features,
        orig_k.out_features,
        bias=(orig_k.bias is not None),
        gather_output=True
    )
    t5_attention.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if orig_k.bias is not None:
        t5_attention.k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # Shard v
    orig_v = t5_attention.v
    t5_attention.v = ColumnParallelLinear(
        orig_v.in_features,
        orig_v.out_features,
        bias=(orig_v.bias is not None),
        gather_output=True
    )
    t5_attention.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if orig_v.bias is not None:
        t5_attention.v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard o
    orig_o = t5_attention.o
    t5_attention.o = RowParallelLinear(
        orig_o.in_features,
        orig_o.out_features,
        bias=(orig_o.bias is not None),
        input_is_parallel=False
    )
    t5_attention.o.weight.data = get_sharded_data(orig_o.weight.data, 1)
    if orig_o.bias is not None:
        t5_attention.o.bias.data = orig_o.bias.data.detach()
    del orig_o

def shard_t5_ff(ff_block):
    # Helper function for ColumnParallel
    def make_column_parallel(orig_layer):
        from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
        new_layer = ColumnParallelLinear(
            orig_layer.in_features,
            orig_layer.out_features,
            bias=(orig_layer.bias is not None),
            gather_output=False
        )
        new_layer.weight.data = get_sharded_data(orig_layer.weight.data, 0)
        if orig_layer.bias is not None:
            new_layer.bias.data = get_sharded_data(orig_layer.bias.data, 0)
        return new_layer

    # Helper function for RowParallel
    def make_row_parallel(orig_layer):
        from neuronx_distributed.parallel_layers.layers import RowParallelLinear
        new_layer = RowParallelLinear(
            orig_layer.in_features,
            orig_layer.out_features,
            bias=(orig_layer.bias is not None),
            input_is_parallel=True
        )
        # For RowParallel, we shard dimension=1
        new_layer.weight.data = get_sharded_data(orig_layer.weight.data, 1)
        if orig_layer.bias is not None:
            new_layer.bias.data = orig_layer.bias.data.detach()
        return new_layer

    if hasattr(ff_block, "wi") and hasattr(ff_block, "wo"):
        orig_wi = ff_block.wi
        ff_block.wi = make_column_parallel(orig_wi)
        del orig_wi

        orig_wo = ff_block.wo
        ff_block.wo = make_row_parallel(orig_wo)
        del orig_wo

    elif hasattr(ff_block, "wi_0") and hasattr(ff_block, "wi_1") and hasattr(ff_block, "wo"):
        orig_wi_0 = ff_block.wi_0
        ff_block.wi_0 = make_column_parallel(orig_wi_0)
        del orig_wi_0

        orig_wi_1 = ff_block.wi_1
        ff_block.wi_1 = make_column_parallel(orig_wi_1)
        del orig_wi_1

        orig_wo = ff_block.wo
        ff_block.wo = make_row_parallel(orig_wo)
        del orig_wo

    else:
        raise ValueError(
            f"Unsupported T5 FF block type: {type(ff_block).__name__}. "
            f"Expected T5DenseReluDense or T5DenseGatedActDense."
        )

def init_text_encoder_2(t5_encoder):
    encoder_stack = t5_encoder.encoder  # T5Stack
    for block in encoder_stack.block:
        # block.layer[0] => T5LayerSelfAttention
        # block.layer[1] => T5LayerFF
        attn = block.layer[0].SelfAttention
        shard_t5_attention(attn)
        ff = block.layer[1].DenseReluDense
        shard_t5_ff(ff)


class TracingT5TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        encoder_outputs = self.neuron_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

