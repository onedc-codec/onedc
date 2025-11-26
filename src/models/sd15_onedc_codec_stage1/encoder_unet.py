from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d_blocks import AttnDownBlock2D
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils import deprecate, is_torch_version, logging



class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=3, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(self.out_channels_, self.out_channels_, kernel_size=3, padding=1, bias=False)

        if self.in_channels != self.out_channels_:
            self.nin_shortcut = nn.Conv2d(self.out_channels_, self.out_channels_, kernel_size=1, bias=False)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.functional.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.functional.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual


def forward_enc(
        self,
        sample: torch.Tensor,
    ) -> Tuple:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """

        # 1. time
        timesteps = torch.tensor([999], dtype=torch.long, device=sample.device)
        
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.time_proj(timesteps)
        
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = None
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)
        z_sample = sample

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps
            
        y_sample = sample

        return y_sample, z_sample


def prepare_unet_encoder(in_ch: int, out_ch:int, ch_config=(512, 768, 768)):
    # 1. create a UNet2DModel
    feedforward_model = UNet2DModel(
        in_channels=in_ch,
        out_channels=out_ch,
        down_block_types=("AttnDownBlock2D","AttnDownBlock2D","DownBlock2D"),
        up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D"),
        block_out_channels=ch_config,
        layers_per_block=2,    
    )
    internal_in_ch = ch_config[0]
    internal_out_ch = ch_config[-1]
            
    # 2. replace the input conv
    feedforward_model.conv_in = nn.Sequential(
        ResnetBlock(in_ch, internal_in_ch),
        ResnetBlock(internal_in_ch, internal_in_ch),
        ResnetBlock(internal_in_ch, internal_in_ch),
        nn.Conv2d(internal_in_ch, internal_in_ch, kernel_size=3, padding=1, stride=2),
    )
    
    # 3. replace the forward function
    feedforward_model.forward = forward_enc.__get__(feedforward_model)
    
    return feedforward_model


if __name__ == "__main__":
    from torchinfo import summary
    
    feedforward_model = prepare_unet_encoder(in_ch=320, out_ch=512).cuda()
    summary(feedforward_model)
    # # del feedforward_model
    
    # dummy input
    sample = torch.randn(1, 320, 64, 64).cuda()
    timestep = torch.ones(1, dtype=torch.long).cuda() * 999
    
    res = feedforward_model(sample)
    pass
