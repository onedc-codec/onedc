from typing import Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from einops import rearrange
from diffusers import AutoencoderKL
from diffusers.utils import deprecate, is_torch_version, logging


def windowed_attn(hidden_states: torch.Tensor, attn: nn.Module, patch_size: int, is_training: bool):
     # 1. if training, we use rearrange to split the hidden_states into patches
    if is_training:
        w_h, w_w = patch_size, patch_size
        nw_h, nw_w = hidden_states.shape[2] // w_h, hidden_states.shape[3] // w_w
        hidden_states = rearrange(hidden_states, 'b c (nw_h h) (nw_w w) -> (b nw_h nw_w) c h w', 
                                  h=w_h, w=w_w, nw_h=nw_h, nw_w=nw_w)
        hidden_states = hidden_states.contiguous()
        hidden_states = attn(hidden_states)
        hidden_states = rearrange(hidden_states, '(b nw_h nw_w) c h w -> b c (nw_h h) (nw_w w)', 
                                  h=w_h, w=w_w, nw_h=nw_h, nw_w=nw_w)
        hidden_states = hidden_states.contiguous()
    else:
        b, c, H, W = hidden_states.shape
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                hidden_states_patch = hidden_states[:, :, i:i+patch_size, j:j+patch_size].contiguous()
                hidden_states_patch = attn(hidden_states_patch)
                hidden_states[:, :, i:i+patch_size, j:j+patch_size] = hidden_states_patch
        hidden_states = hidden_states.contiguous()
    return hidden_states
    


def windowed_attn_forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            if attn is not None:
                hidden_states = windowed_attn(hidden_states, attn, self.attn_patch, self.training)
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
        else:
            if attn is not None:
                hidden_states = windowed_attn(hidden_states, attn, self.attn_patch, self.training)
        
            hidden_states = resnet(hidden_states, temb)

    return hidden_states


class AutoencoderKL_patch_attn(AutoencoderKL):
    def __init__(self, in_channels = 3, out_channels = 3, down_block_types = ..., up_block_types = ..., block_out_channels = ..., layers_per_block = 1, act_fn = "silu", latent_channels = 4, norm_num_groups = 32, sample_size = 32, scaling_factor = 0.18215, shift_factor = None, latents_mean = None, latents_std = None, force_upcast = True, use_quant_conv = True, use_post_quant_conv = True, mid_block_add_attention = True):
        super().__init__(in_channels, out_channels, down_block_types, up_block_types, block_out_channels, layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size, scaling_factor, shift_factor, latents_mean, latents_std, force_upcast, use_quant_conv, use_post_quant_conv, mid_block_add_attention)
        
        # replace function
        self.encoder.mid_block.forward = windowed_attn_forward.__get__(self.encoder.mid_block, self.encoder.mid_block.__class__)
        self.decoder.mid_block.forward = windowed_attn_forward.__get__(self.decoder.mid_block, self.decoder.mid_block.__class__)
        
        # set default patch size
        self.set_attn_patch(64)
    
    
    def set_attn_patch(self, attn_patch: int):
        assert attn_patch > 0, "attn_patch must be greater than 0"
        self.attn_patch = attn_patch
        self.encoder.mid_block.attn_patch = attn_patch
        self.decoder.mid_block.attn_patch = attn_patch