import math
import random as rd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from einops import rearrange, pack, unpack

from vector_quantize_pytorch import FSQ
from modules.blocks.swin import DualSwinBlock
from modules.vqgan.blocks import ResnetBlock, AttnBlock
from modules.dcvc import DepthConvBlock4, ResidualBlockUpsample
from modules.entropy.compression_model import CompressionModel
from modules.entropy.utils import encode_i, decode_i, write_to_file, read_from_file, filesize

from .encoder_unet import prepare_unet_encoder


def get_ResnetBlock_group(ch, num):
    g = [ResnetBlock(ch) for i in range(num)]
    return g
    

def get_ResnetBlock_Attn_group(ch, res_num, attn_num):
    g_r = [ResnetBlock(ch) for i in range(res_num)]
    g_a = [AttnBlock(ch) for i in range(attn_num)]
    g = g_r + g_a
    return g


def get_upsample(in_ch, out_ch=None):
    g = [
        nn.Conv2d(in_ch, in_ch*4, kernel_size=1),
        nn.PixelShuffle(2),
    ]
    if out_ch is None:
        out_ch = in_ch
    g.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
    return g


def get_bottleneck_group(ch):
    return [
        ResnetBlock(ch),
        AttnBlock(ch),
        ResnetBlock(ch),
    ]
    

class Encoder(nn.Module):
    def __init__(self, in_ch=3, cond_ch=4, out_ch=128, unet_ch_config=(512, 768, 768)):
        super().__init__()
        ch_emb = 192
        ch_8x = 320
        ch_16x = unet_ch_config[0]
        ch_sem = unet_ch_config[-1]
        
        self.pix_emb = nn.Conv2d(in_ch, ch_emb, kernel_size=8, stride=8, padding=0)
        self.pix_fusion = nn.Conv2d(ch_emb+cond_ch, ch_8x, kernel_size=1)
        self.unet = prepare_unet_encoder(ch_8x, ch_16x, unet_ch_config)
        self.trans_coding = nn.Sequential(
            *get_bottleneck_group(ch_16x),
            DepthConvBlock4(ch_16x, ch_16x, inplace=False),
            DepthConvBlock4(ch_16x, out_ch, inplace=False),
        )
        self.network_context_manager = None
        
    
    def set_context_manager(self, network_context_manager):
        self.network_context_manager = network_context_manager

   
    def forward(self, x, cond):
        x_emb = self.pix_emb(x)
        x_emb = self.pix_fusion(torch.cat([x_emb, cond], dim=1))
        if self.network_context_manager:
            with self.network_context_manager:
                y, sem = self.unet(x_emb)
            y = y.to(x.dtype)
            sem = sem.to(x.dtype)
        else:
            y, sem = self.unet(x_emb)
        y = self.trans_coding(y)
        return y, sem


class Decoder(nn.Module):
    def __init__(self, in_ch=128, internal_ch=512, semantic_ch=768, out_ch=256):
        super().__init__()
        ch_8x = internal_ch // 2
        ch_16x = internal_ch
        
        self.trans_coding = nn.Sequential(
            DepthConvBlock4(in_ch, ch_16x, inplace=False),
            DepthConvBlock4(ch_16x, ch_16x, inplace=False),
        )
        self.blocks = nn.Sequential(
            *get_ResnetBlock_group(ch_16x, 3),
            *get_upsample(in_ch=ch_16x, out_ch=ch_8x),
            *get_ResnetBlock_group(ch_8x, 3),
        )
        self.sem_up = nn.Sequential(
            ResidualBlockUpsample(semantic_ch, ch_16x, inplace=False),
            DepthConvBlock4(ch_16x, ch_16x, inplace=False),
            ResidualBlockUpsample(ch_16x, ch_8x, inplace=False),
            DepthConvBlock4(ch_8x, ch_8x, inplace=False),
            ResidualBlockUpsample(ch_8x, ch_8x, inplace=False),
        )
        self.conv_out = DepthConvBlock4(ch_8x*2, out_ch, inplace=False)
        
    def forward(self, y_hat, sem_hat):
        y_hat = self.trans_coding(y_hat)
        y_hat = self.blocks(y_hat)
        sem_hat = self.sem_up(sem_hat)
        return self.conv_out(torch.cat([y_hat, sem_hat], dim=1))


class HyperEncoder(nn.Module):
    def __init__(self, y_ch, sem_ch, internal_ch, z_fsq_levels):
        super().__init__()
        
        self.y_trans_coding = nn.Sequential(
            DepthConvBlock4(y_ch, y_ch, inplace=False),
            nn.Conv2d(y_ch, y_ch, kernel_size=3, padding=1, stride=2),
            DepthConvBlock4(y_ch, y_ch, inplace=False),
            nn.Conv2d(y_ch, y_ch, kernel_size=3, padding=1, stride=2),
        )
        self.fusion = nn.Sequential(
            DepthConvBlock4(y_ch+sem_ch, sem_ch, inplace=False),
            AttnBlock(sem_ch),
            DepthConvBlock4(sem_ch, internal_ch, inplace=False),
            AttnBlock(internal_ch),
            DepthConvBlock4(internal_ch, internal_ch, inplace=False),
            nn.Conv2d(internal_ch, len(z_fsq_levels), 1),
        )
        
        
    def forward(self, y, sem):
        z = self.y_trans_coding(y)
        z = self.fusion(torch.cat([z, sem], dim=1))
        return z


class HyperDecoder(nn.Module):
    def __init__(self, entropy_ch, z_fsq_levels):
        super().__init__()
        
        self.feat_in = nn.Sequential(
            nn.Conv2d(len(z_fsq_levels), entropy_ch, 1, stride=1),
            nn.LeakyReLU(),
        )
        self.to_entropy = nn.Sequential(
            DepthConvBlock4(entropy_ch, entropy_ch, inplace=False),
            ResidualBlockUpsample(entropy_ch, entropy_ch, 2, inplace=False),
            DepthConvBlock4(entropy_ch, entropy_ch, inplace=False),
            ResidualBlockUpsample(entropy_ch, entropy_ch, 2, inplace=False),
            DepthConvBlock4(entropy_ch, entropy_ch, inplace=False),
        )
        
        
    def forward(self, z_hat):
        z_hat = self.feat_in(z_hat)
        z_entropy = self.to_entropy(z_hat)
        z_semantic = z_hat
        return z_entropy, z_semantic
    
    
class SemanticAdaptor(nn.Module):
    def __init__(self, entropy_ch, semantic_ch):
        super().__init__()
        self.to_semantic = nn.Sequential(
            DepthConvBlock4(entropy_ch, semantic_ch, inplace=False),
            *get_ResnetBlock_Attn_group(semantic_ch, res_num=1, attn_num=2),
            *get_ResnetBlock_Attn_group(semantic_ch, res_num=1, attn_num=2),
            DepthConvBlock4(semantic_ch, semantic_ch, inplace=False),
        )
    
    
    def forward(self, x):
        return self.to_semantic(x)
    
    
class IntraNoAR(CompressionModel):
    def __init__(self, 
                 cond_ch=4, 
                 ctrl_ch=320, 
                 internal_ch=512, 
                 bottleneck_ch=128, 
                 unet_ch_config=(512, 768, 768),
                 z_fsq_levels=[4,4,4,4,4,4,4]):
        super().__init__(y_distribution='gaussian', z_channel=bottleneck_ch, ec_thread=False, stream_part=1)
        N = bottleneck_ch
        semantic_ch = unet_ch_config[-1]

        self.enc = Encoder(3, cond_ch, bottleneck_ch, unet_ch_config)
        self.dec = Decoder(bottleneck_ch, internal_ch, semantic_ch, ctrl_ch)
        self.semantic_adaptor = SemanticAdaptor(N, semantic_ch)
        
        self.hyper_enc = HyperEncoder(N, semantic_ch, internal_ch, z_fsq_levels)
        self.hyper_dec = HyperDecoder(N, z_fsq_levels)
        self.z_vq = FSQ(z_fsq_levels)
        self.z_fsq_levels = z_fsq_levels

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock4(N, N * 2, inplace=False),
            DepthConvBlock4(N * 2, N * 2, inplace=False),
        )
        self.y_spatial_prior_reduction = nn.Conv2d(N * 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock4(N * 2, N * 2, inplace=False)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock4(N * 2, N * 2, inplace=False)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock4(N * 2, N * 2, inplace=False)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock4(N * 2, N * 2, inplace=False),
            DepthConvBlock4(N * 2, N * 2, inplace=False),
            DepthConvBlock4(N * 2, N * 2, inplace=False),
        )
        
        assert np.round(np.log2(self.z_vq.codebook_size)) == np.log2(self.z_vq.codebook_size)      # for coding.
        self.index_unit_length = int(np.log2(self.z_vq.codebook_size))
        self.ds = 64
        self.cond_ds = 8
        self.debug = False
        self.network_context_manager = None
        
        
    def set_context_manager(self, network_context_manager):
        self.network_context_manager = network_context_manager
        self.enc.set_context_manager(network_context_manager)
        
    
    def _forward_enc(self, x, cond):
        y, sem = self.enc(x, cond)
        z = self.hyper_enc(y, sem)
        z_hat, z_vq_indices = self.z_vq(z)

        params, z_semantic = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        
        # quant_result: (y_res, y_q, y_hat, scales_hat)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)
        
        enc_result = (y, z, z_hat, z_vq_indices, params)
        quant_result = (y_res, y_q, y_hat, scales_hat)
        return enc_result, quant_result, z_semantic
        

    def _forward(self, x, cond, fix_encoder=False):
        B, _, H, W = x.shape
        pixel_num = H * W
        
        # encoder with quantization
        if fix_encoder:
            with torch.no_grad():
                enc_result, quant_result, z_semantic = self._forward_enc(x, cond)
                y_res, y_q, y_hat, scales_hat = tuple(item.detach() for item in quant_result)
        else:
            enc_result, quant_result, z_semantic = self._forward_enc(x, cond)
            y_res, y_q, y_hat, scales_hat = quant_result

        # decoder
        y_semantic = self.semantic_adaptor(z_semantic)
        x_hat = self.dec(y_hat, y_semantic)

        # calculate
        if self.training:
            y_for_bit = self.add_noise(y_res)
        else:
            y_for_bit = y_q
        # noise bits
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)        
        bpp_y = torch.mean(torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num)
        # hard quant bits
        bits_hard_y = self.get_y_gaussian_bits(y_q.detach(), scales_hat)         
        bpp_hard_y = torch.mean(torch.sum(bits_hard_y, dim=(1, 2, 3)) / pixel_num)

        bits = torch.sum(bpp_y) * pixel_num
        bpp = bpp_y

        result = {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_hard_y": bpp_hard_y,
            "y_semantic": y_semantic,
            "z_semantic": z_semantic,
        }
        return result
    
    
    def forward(self, x, cond, fix_encoder=False, fix_codec=False):
        if fix_codec:
            with torch.no_grad():
                res_dict = self._forward(x, cond, fix_encoder=True)
                return_dict = {k: v.detach() for k, v in res_dict.items()}
        else:
            return_dict = self._forward(x, cond, fix_encoder)
        return return_dict


    def _encode_caption(self, caption):
        caption_bytes = caption.encode('utf-8')
        caption_length = len(caption_bytes)
        return {
            'bit_stream_caption': caption_bytes,
            'caption_length': caption_length,
        }

    
    @torch.no_grad()
    def encode(self, x, cond, pic_width, pic_height, fp=None):
        compressed = self._compress(x, cond)
        compressed.update(self._encode_caption(""))
        compressed.update({
            'bit_stream_caption': '',
            'pic_width': pic_width,
            'pic_height': pic_height,
        })
        bit_stream_total = encode_i(**compressed)
        if fp:
            write_to_file(bit_stream_total, fp)
            file_size = filesize(fp)
        else:
            file_size = 0.0
        
        # count bit num
        bits_y = len(compressed['bit_stream_y']) * 8
        bits_z = len(compressed['bit_stream_z']) * 8
        bits_caption = len(compressed['bit_stream_caption']) * 8
        bits_total = len(bit_stream_total) * 8
        bpp_y = bits_y / (pic_height * pic_width)
        bpp_z = bits_z / (pic_height * pic_width)
        bpp_caption = bits_caption / (pic_height * pic_width)
        bpp = bits_total / (pic_height * pic_width)
        bpp_file = file_size * 8 / (pic_height * pic_width)
        bpp_dict = {
            'bits_y': bits_y,
            'bits_z': bits_z,
            'bits_caption': bits_caption,
            'bits_total': bits_total,
            'bpp_y': bpp_y,
            'bpp_z': bpp_z,
            'bpp_caption': bpp_caption,
            'bpp': bpp,
            'bpp_file': bpp_file,
        }
        return bit_stream_total, bpp_dict
    
    
    @torch.no_grad()
    def decode(self, fp=None, stream=None):
        assert fp or stream
        if stream:
            bit_stream_total = stream
        else:
            bit_stream_total = read_from_file(fp)
        dec_dict = decode_i(bit_stream_total, self.index_unit_length, self.ds)
        
        x_hat, y_semantic = self._decompress(**dec_dict)
        orig_HW = (dec_dict['height'], dec_dict['width'])
        pad_HW = (dec_dict['pad_height'], dec_dict['pad_width'])
        pad_size = dec_dict['pad_tuple']
        return x_hat, y_semantic, orig_HW, pad_HW, pad_size


    def _compress(self, x, cond):
        index_unit_length = self.index_unit_length
        
        y, sem = self.enc(x, cond)
        z = self.hyper_enc(y, sem)
        z_hat, z_vq_indices = self.z_vq(z)
        
        # only for debug
        if self.debug:
            self.z_hat = z_hat
            self.z_vq_indices = z_vq_indices
            self.z_dtype = z_hat.dtype

        params, z_semantic = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
                y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        # print("y", y.abs().max(), y.abs().mean())
        self.entropy_coder.reset()
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0, skip_thres=self.force_zero_thres)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1, skip_thres=self.force_zero_thres)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2, skip_thres=self.force_zero_thres)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3, skip_thres=self.force_zero_thres)
        self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        # encode index
        index = z_vq_indices
        index = index.reshape(-1).detach().cpu().numpy()
        index_str = ''.join([bin(index[i])[2:].zfill(index_unit_length) for i in range(len(index))])
        num_bytes = (len(index_str) + 7) // 8  
        index_byte = int(index_str, 2)  
        bit_stream_index = index_byte.to_bytes(num_bytes, 'big')

        result = {
            "bit_stream_y": bit_stream,
            "bit_stream_z" : bit_stream_index,
        }
        return result


    def _decompress(self, bit_stream_y, bit_stream_z, pad_height, pad_width, bit_stream_caption=None, **kwargs):
        index_unit_length = self.index_unit_length
        device = next(self.parameters()).device

        self.entropy_coder.set_stream(bit_stream_y)
        z_size = (pad_height // self.ds), (pad_width // self.ds)

        # decode index
        index_hat_byte = int.from_bytes(bit_stream_z, 'big')  
        index_hat_str = bin(index_hat_byte)[2:].zfill(z_size[0] * z_size[1] * index_unit_length)
        index_hat_list = [int(index_hat_str[index_unit_length * i:index_unit_length * (i + 1)], 2) for i in range(z_size[0] * z_size[1])]
        index_hat = torch.tensor(index_hat_list).to(device).reshape(1, *z_size)  # B H W

        z_hat = self.z_vq.indices_to_codes(index_hat)
        
        # get autocast
        if torch.is_autocast_enabled():
            z_hat = z_hat.to(torch.get_autocast_gpu_dtype())
        
        # only for debug
        if self.debug:
            assert torch.equal(z_hat, self.z_hat)
            assert torch.equal(index_hat, self.z_vq_indices)
            assert z_hat.dtype == self.z_dtype

        params, z_semantic = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior,
                                                self.y_spatial_prior_reduction)

        y_semantic = self.semantic_adaptor(z_semantic)
        x_hat = self.dec(y_hat, y_semantic)
        return x_hat, y_semantic



class AuxDecoder(nn.Module):
    def __init__(self, in_ch=256, out_ch=4):
        super().__init__()
        self.blocks = nn.Sequential(
            *get_ResnetBlock_Attn_group(in_ch, res_num=2, attn_num=1),
            *get_ResnetBlock_group(in_ch, 1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        
    
    def forward(self, x):
        return self.blocks(x)
    
    
class Codeformer(nn.Module):
    def __init__(self, in_ch=768, codebook_size=1024):
        super().__init__()
        self.up_sample = nn.Sequential(
            DepthConvBlock4(in_ch, in_ch, inplace=False),
            nn.Conv2d(in_ch, in_ch*4, kernel_size=1),
            nn.PixelShuffle(2),
            DepthConvBlock4(in_ch, in_ch, inplace=False),
        )
        
        head_dim = 64
        num_heads = in_ch // head_dim
        mlp_dim = in_ch * 4
        self.blocks = nn.Sequential(
            DualSwinBlock(in_ch, num_heads, head_dim, mlp_dim, window_size=16, 
                          use_pos_embedding=True, relative_pos_embedding=False),
            DualSwinBlock(in_ch, num_heads, head_dim, mlp_dim, window_size=16, 
                          use_pos_embedding=False, relative_pos_embedding=False),
            DualSwinBlock(in_ch, num_heads, head_dim, mlp_dim, window_size=16, 
                          use_pos_embedding=False, relative_pos_embedding=False),
        )
        
        self.mlp_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, kernel_size=1),
            nn.GroupNorm(16, in_ch*4),
            nn.GELU(),
            nn.Conv2d(in_ch*4, in_ch, kernel_size=1),
            nn.GroupNorm(16, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, codebook_size, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self, x):
        h = self.up_sample(x)
        h = self.blocks(h)
        logits = self.mlp_head(h)
        probs = self.softmax(logits)
        return logits, probs