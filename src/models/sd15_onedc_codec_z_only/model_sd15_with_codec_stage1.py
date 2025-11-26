import torch, accelerate, omegaconf
from torch import nn

from einops import rearrange
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers import DDIMScheduler

from utils import load_safetensor
from modules.entropy.utils import get_padding_size
from modules.vae.autoencoders_patch_attn import AutoencoderKL_patch_attn
from modules.dmd.utils import get_x0_from_noise, NoOpContext, DummyModule
from models.sd15_onedc_codec_stage1.decoder_unet import prepare_unet_for_codec
from models.sd15_onedc_codec_z_only.codec_module import IntraNoAR


class SD15_1step_codec_stage1(nn.Module):
    def __init__(self, args: omegaconf.OmegaConf, accelerator: accelerate.Accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator
        self.model_id = args.model_id = "runwayml/stable-diffusion-v1-5"
        self.vae_model_id = args.vae_model_id = "madebyollin/taesd"
        self.vae_dim = 4
        self.num_visuals = args.num_visuals
        self.use_fp16 = args.use_fp16
        # large vae settings
        self.vae_attn_patch = args.vae_attn_patch
        self.use_large_vae = args.use_large_vae
        # lora or tunning unet
        self.use_lora = args.lora_config is not None
        self.lora_config = args.lora_config
        self.tune_unet = False

        print("[INFO] SD 1.5 codec, semantic hyper-prior, type2, use mean from z to recon image.")
        
        # build vae
        self.vae = AutoencoderTiny.from_pretrained(
            args.vae_model_id, torch_dtype=torch.float32
        ).float().to(accelerator.device)
        self.vae.requires_grad_(False)
        self.vae_large = AutoencoderKL_patch_attn.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="vae", torch_dtype=torch.float32
        ).float().to(accelerator.device)
        self.vae_large.requires_grad_(False)
        self.vae_large.set_attn_patch(self.vae_attn_patch)
        self.vae_large.enable_gradient_checkpointing()      # to save memory for vae large

        if self.use_large_vae:
            self.vae = DummyModule().float().to(accelerator.device)
        else:
            del self.vae.encoder
            del self.vae_large.decoder
            
        # build the unet model for codec
        self.feedforward_model = prepare_unet_for_codec(
            in_ch = 320,
            pretrained_path = args.unet_ckpt,
            lora_config = args.lora_config,
        )
        self.feedforward_model.manually_set_grad(
            is_lora=self.use_lora, 
            is_all_model=self.tune_unet
        )
        self.feedforward_model = self.feedforward_model.to(accelerator.device)
        
        # build codec model
        self.codec_model = IntraNoAR(
            cond_ch=self.vae_dim,
            ctrl_ch=320,
            internal_ch=args.codec.internal_ch,
            bottleneck_ch=args.codec.bottleneck_ch,
            unet_ch_config=args.codec.unet_ch_config,
            z_fsq_levels=args.codec.z_fsq_levels
        ).to(accelerator.device)
        self.codec_model.requires_grad_(True)

        # load other ckpt if provided, only run this as a base class.
        if self.__class__ == SD15_1step_codec_stage1:
            self.load_part_ckpt()

        # gradient checkpoint
        self.gradient_checkpointing = args.gradient_checkpointing
        if args.gradient_checkpointing:
            self.feedforward_model.enable_gradient_checkpointing()
        
        # diffusion scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id, subfolder="scheduler"
        )
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(accelerator.device)
        self.num_denoising_step = args.num_denoising_step
        self.num_train_timesteps = args.num_train_timesteps
        self.denoising_step_list = torch.tensor(
            list(range(self.num_train_timesteps-1, 0, -(self.num_train_timesteps//self.num_denoising_step))),
            dtype=torch.long 
        )
        self.conditioning_timestep = args.conditioning_timestep
        self.denoising_timestep = args.denoising_timestep           # 1000 for sd1.5
        self.timestep_interval = self.denoising_timestep//self.num_denoising_step
        
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()
        # self.codec_model.set_context_manager(self.network_context_manager)        # we use bf16 for all model, thus no need to set codec.
        torch.cuda.empty_cache()
        
        self.freeze_codec = True
        self.freeze_codec_encoder = True
        
        
    def load_part_ckpt(self):
        args = self.args
        accelerator = self.accelerator
        
        # load codec ckpt if provided
        if args.codec_ckpt is not None:
            accelerator.print(f"[INFO] loading codec ckpt only from {args.codec_ckpt}")
            codec_path = args.codec_ckpt
            codec_sd = load_safetensor(codec_path, map_location="cpu")
            print(self.codec_model.load_state_dict(codec_sd, strict=True))
            
        # load basic unet ckpt if provided
        if args.unet_ckpt_lora is not None:
            accelerator.print(f"[INFO] loading unet + lora ckpt only from {args.unet_ckpt_lora}")
            generator_path = args.unet_ckpt_lora
            generator_sd = load_safetensor(generator_path, map_location="cpu")
            print(self.feedforward_model.load_state_dict(generator_sd, strict=False))


    def prepare(self):
        """
        The sequence of prepare determines the order of the saved model's name.
        See annotation for checkpoint name.
        """
        self.feedforward_model = self.accelerator.prepare(self.feedforward_model)   # pytorch_model.bin
        self.codec_model = self.accelerator.prepare(self.codec_model)               # pytorch_model_1.bin
        
    
    def train(self):
        self.feedforward_model.train()
        self.codec_model.train()
            
    
    def eval(self):
        self.feedforward_model.eval()
        self.codec_model.eval()


    def vae_decode_image(self, latents):
        vae_model = self.vae_large if self.use_large_vae else self.vae
        latents = 1 / vae_model.config.scaling_factor * latents
        image = vae_model.decode(latents).sample.float()
        return image 


    @torch.no_grad()
    def vae_encode_image(self, image):
        latents = self.vae_large.encode(image).latent_dist.sample()
        latents = self.vae_large.config.scaling_factor * latents
        return latents.float().detach()


    @torch.no_grad()
    def vqgan_encode_image(self, image):
        image = image * 0.5 + 0.5       # the maskgit vqgan model takes [0,1] input
        quant_latent, codebook_indices = self.vqgan.encode(image, get_quant=True)
        return quant_latent, codebook_indices


    @torch.no_grad()
    def forward_codec_unet(self, x_pix, x_latent, timesteps, unet_added_conditions):
        # 1. codec encoding
        enc_dict = self.codec_model(
            x=x_pix,
            cond=x_latent,
            fix_codec=self.freeze_codec,
            fix_encoder=self.freeze_codec_encoder,
        )
        unet_input = enc_dict['x_hat']  
        unet_semantic = enc_dict['y_semantic']
        unet_semantic = rearrange(unet_semantic, 'b c h w -> b (h w) c').contiguous()

        # 2. decode image with unet
        sample, dummy_noise = self.feedforward_model(
            sample=unet_input,
            timestep=timesteps.long(),
            encoder_hidden_states=unet_semantic,
            added_cond_kwargs=unet_added_conditions,
        )

        # assume epsilon prediction 
        student_x0_pred = get_x0_from_noise(
            dummy_noise.double(), sample.double(), self.alphas_cumprod.double(), timesteps
        ).float()
        
        return student_x0_pred, enc_dict


    @torch.no_grad()
    def forward(self, image):
        accelerator = self.accelerator
        x_pix = image
        x_latent = self.vae_encode_image(image)
        ode_timesteps = torch.ones(x_pix.shape[0], device=accelerator.device, dtype=torch.long) * self.conditioning_timestep
        ode_unet_added_conditions = None

        with self.network_context_manager:
            student_x0_pred, enc_dict = self.forward_codec_unet(
                x_pix, x_latent, ode_timesteps, ode_unet_added_conditions
            )
            ce_loss = torch.tensor(0.0, device=accelerator.device)
            vqgan_mse_loss = torch.tensor(0.0, device=accelerator.device)

        ode_pred_image = self.vae_decode_image(student_x0_pred.float())
        enc_dict['x_latent'] = x_latent
        enc_dict['x_latent_recon'] = student_x0_pred
        enc_dict['code_ce_loss'] = ce_loss
        enc_dict['code_mse_loss'] = vqgan_mse_loss
        return enc_dict, ode_pred_image

