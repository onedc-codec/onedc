import torch, accelerate, omegaconf
from torch import nn
import gc

from piq import LPIPS
from peft import LoraConfig
from einops import rearrange
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers import UNet2DConditionModel, DDIMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer

from utils import load_safetensor
from modules.entropy.utils import get_padding_size
from modules.dmd.utils import get_x0_from_noise, NoOpContext, DummyModule
from models.sd15_onedc_codec_stage1.model_sd15_with_codec_stage1 import SD15_1step_codec_stage1

from modules.dmd.sd_guidance import SDGuidance      # original dmd model.
        

class SD15_1step_codec_stage2_dmd(SD15_1step_codec_stage1):
    def __init__(self, args, accelerator):
        args.use_codeformer = False
        args.use_large_vae = True
        args.freeze_codec = True
        args.freeze_codec_encoder = True
        super().__init__(args, accelerator)
        
        accelerator.print(f"[INFO] Start DMD training with guidance.")
        
        # delete useless models
        self.codeformer = DummyModule()
        self.vqgan = DummyModule()
        
        # guidance model
        self.guidance_model = SDGuidance(args, accelerator).to(accelerator.device)
        self.gan_alone = self.guidance_model.gan_alone
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.noise_scheduler = self.guidance_model.scheduler
        
        # guidance loss related
        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight 
        self.cls_on_clean_image = args.cls_on_clean_image 
        self.gen_cls_loss = args.gen_cls_loss 
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        
        # create text encoder for dmd
        self.tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder"
        ).to(accelerator.device).eval()
        self.text_encoder.requires_grad_(False)
        
        # uncound condition
        with torch.no_grad():
            uncond_input_ids = self.tokenizer(
                    [""], max_length=self.tokenizer.model_max_length, 
                    return_tensors="pt", padding="max_length", truncation=True
                ).input_ids.to(accelerator.device)
            self.uncond_embedding = self.text_encoder(uncond_input_ids)[0].detach()
        
        # here we fix codec
        self.codec_model.requires_grad_(False)
        
        # load ckpt
        self.load_part_ckpt()
        self.eval()
        
        # free memory
        gc.collect()
        torch.cuda.empty_cache()
    
    
    def load_part_ckpt(self):
        super().load_part_ckpt()
        
        # load guidance ckpt if provided
        if self.args.guidance_ckpt is not None:
            if self.accelerator.is_main_process:
                print(f"[INFO] loading guidance ckpt only from {self.args.guidance_ckpt}")
            guidance_path = self.args.guidance_ckpt
            guidance_sd = load_safetensor(guidance_path, map_location="cpu")
            print(self.guidance_model.load_state_dict(guidance_sd, strict=True))   
            
            
    def get_trainable_params(self, is_guidance: bool):
        if not is_guidance:
            return super().get_trainable_params()
        else:
            named_param_list = [(name, param) for name, param in self.guidance_model.named_parameters() if param.requires_grad]
            param_list = [param for _, param in named_param_list]
            return param_list, named_param_list
            
        
    def prepare(self):
        super().prepare()
        self.guidance_model = self.accelerator.prepare(self.guidance_model)
        
    
    def train(self):
        super().train()
        self.guidance_model.train()
        if not self.gan_alone:
            self.accelerator.unwrap_model(self.guidance_model).real_unet.eval()
        
    
    def eval(self):
        super().eval()
        self.guidance_model.eval()
        
        
    @torch.no_grad()
    def prepare_generation_data(self, text_input_ids_one, x_latent, 
                                     unet_added_conditions, uncond_unet_added_conditions):
        text_embedding_output = self.text_encoder(text_input_ids_one)
        text_embedding = text_embedding_output[0].float().detach()
        pooled_text_embedding = text_embedding_output[1].float().detach()
        
        real_train_dict = {
            "images": x_latent,
            "text_input_ids_one": text_input_ids_one,
            "text_embedding": text_embedding,
            "pooled_text_embedding": pooled_text_embedding,
            "unet_added_conditions": unet_added_conditions,
            "uncond_unet_added_conditions": uncond_unet_added_conditions,
        }
        return text_embedding, pooled_text_embedding, real_train_dict
    
    
    def forward(self, 
            image,
            text_ids,
            visual=False,
            compute_generator_gradient=True,
            generator_turn=False,
            guidance_turn=False,
            generator_data_dict=None        # this is also for guidance turn
        ):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn)
        accelerator = self.accelerator
        _, _, H, W = image.shape
        x_pix = image

        if generator_turn:
            x_latent = self.vae_encode_image(x_pix).detach()
            timesteps = torch.ones(x_latent.shape[0], device=accelerator.device, dtype=torch.long) * self.conditioning_timestep
            
            unet_added_conditions = None
            uncond_unet_added_conditions = None
            text_embedding, pooled_text_embedding, real_train_dict =\
                self.prepare_generation_data(text_ids, x_latent, 
                                                  unet_added_conditions, 
                                                  uncond_unet_added_conditions)
            uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)

            # generate image
            if compute_generator_gradient:
                with self.network_context_manager:
                    generated_image, enc_dict = self.forward_codec_unet(
                        x_pix, x_latent, timesteps, 
                        unet_added_conditions,
                    )
                pred_pix_image = self.vae_decode_image(generated_image.float())
            else:
                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()

                with torch.no_grad():
                    with self.network_context_manager:
                        generated_image, enc_dict = self.forward_codec_unet(
                            x_pix, x_latent, timesteps, 
                            unet_added_conditions,
                        )
                    # not decoding image when not update generator or not visualizing.
                    pred_pix_image = self.vae_decode_image(generated_image.float()) if visual else None

                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).enable_gradient_checkpointing()

            generator_data_dict = {
                "original_image": x_latent,
                "image": generated_image,
                "text_embedding": text_embedding,
                "pooled_text_embedding": pooled_text_embedding,
                "uncond_embedding": uncond_embedding,
                "real_train_dict": real_train_dict,
                "unet_added_conditions": unet_added_conditions,
                "uncond_unet_added_conditions": uncond_unet_added_conditions
            } 

            if compute_generator_gradient:
                # avoid any side effects of gradient accumulation, DO NOT use context manager here! as it will handel this.
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {} 
                
            log_dict['pred_pix_image'] = pred_pix_image
            log_dict['generator_data_dict'] = generator_data_dict
            loss_dict['bpp'] = enc_dict['bpp']
            loss_dict['bpp_hard_y'] = enc_dict['bpp_hard_y']

            # generate guidance visual if needed
            image_dict = {}
            if visual:
                image_dict["generated_image"] = pred_pix_image
                image_dict["original_image"] = x_pix
                decode_key = ["dmtrain_pred_real_image", "dmtrain_pred_fake_image"]

                with torch.no_grad():
                    if compute_generator_gradient and (not self.gan_alone):
                        for key in decode_key:
                            image_dict[key+"_decoded"] = self.vae_decode_image(log_dict[key].detach().float()[:self.num_visuals]) 

        elif guidance_turn:
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=generator_data_dict
            )
            image_dict = {}
            
        else:
            raise ValueError("Invalid turn.")
        
        return loss_dict, log_dict, image_dict