import torch
import torch.nn as nn

DTYPE=torch.bfloat16

class TracingVAEDecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        latents: torch.Tensor
    ):
        latents = latents.to(torch.float32)
        return self.decoder(
            latents
        )

