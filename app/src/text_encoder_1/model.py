import torch
import torch.nn as nn
from transformers.modeling_outputs \
    import BaseModelOutputWithPooling
from typing import Optional, Union, Tuple


class TracingCLIPTextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return self.neuron_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

