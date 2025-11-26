import sys, os, shutil
import time, argparse, logging
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from utils import load_safetensor, instantiate_from_config
from data.evalset import ImageOnlyDataset

from models.sd15_onedc_codec_z_only.model_sd15_with_codec_stage1 import SD15_1step_codec_stage1


logger = get_logger(__name__, log_level="INFO")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 
torch.manual_seed(0)
torch.set_num_threads(1)

        
class Evaluator:
    def __init__(self, args):
        self.args = args

        # accelerate config
        accelerator = Accelerator(mixed_precision="no")
        self.accelerator = accelerator

        # path
        self.output_path = args.output_path
        self.recon_path = os.path.join(self.output_path, "recon")
        self.detail_file = os.path.join(self.output_path, "detail_bpp_caption.xlsx")
        self.summary_file = os.path.join(self.output_path, "summary_bpp.xlsx")
        
        # build folder
        if accelerator.is_main_process:
            if os.path.exists(self.output_path):
                shutil.rmtree(self.output_path)
            os.makedirs(self.output_path, exist_ok=False)
            os.makedirs(self.recon_path, exist_ok=False)
            
        # pandas dataframe
        self.detail_df = pd.DataFrame()
        
        # clean path
        args.use_codeformer = False
        args.unet_ckpt = None
        args.codec_ckpt = None
        args.vae_ckpt = None
        args.control_ckpt = None
        args.unet_ckpt_lora = None
        args.codeformer_ckpt = None
        args.guidance_ckpt = None

        # build model
        self.model = SD15_1step_codec_stage1(args, accelerator)
        self.model.prepare()
        self.model.codec_model.update(force=True)
        self.model.codec_model.debug = False

        self.load(args.checkpoint_path)

        img_dataset = ImageOnlyDataset(paths=args.eval_image_path)
        img_dataloader = torch.utils.data.DataLoader(
            img_dataset, num_workers=1, 
            batch_size=1, shuffle=False,
        )
        self.img_dataloader = accelerator.prepare(img_dataloader)


    def load(self, checkpoint_path):
        feedforward_path = os.path.join(checkpoint_path, "model.safetensors")
        codec_path = os.path.join(checkpoint_path, "model_1.safetensors")
        feedforward_sd = load_safetensor(feedforward_path, map_location="cpu")
        codec_sd = load_safetensor(codec_path, map_location="cpu")
        print(self.model.feedforward_model.load_state_dict(feedforward_sd, strict=True))
        print(self.model.codec_model.load_state_dict(codec_sd, strict=True))


    @torch.no_grad()
    def evaluate(self):
        disable_tqdm = not self.accelerator.is_main_process
        accelerator = self.accelerator
        self.model.eval()
        
        for idx, item in tqdm(enumerate(self.img_dataloader), total=len(self.img_dataloader), disable=disable_tqdm):
            image_name = item['name'][0]
            image = item['image']
            image_norm = image * 0.5 + 0.5
            image_h, image_w = image.shape[2], image.shape[3]
            
            # pad image
            if image.shape[2] % 64 != 0 or image.shape[3] % 64 != 0:
                pad_h = (64 - image.shape[2] % 64) % 64
                pad_w = (64 - image.shape[3] % 64) % 64
                image_pad = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                image_pad = image
                pad_h = pad_w = 0
            img_pad_h, img_pad_w = image_pad.shape[2], image_pad.shape[3]
            patch_h = img_pad_h // 64
            patch_w = img_pad_w // 64
            bits = patch_h * patch_w * 14           # 14 bit per indices
            bpp = bits / (image_h * image_w)
            
            recon_file = os.path.join(self.recon_path, f"{image_name}.png")
            
            enc_dict, recon_pad = self.model.forward(image_pad)
            bpp_hard_y = enc_dict['bpp_hard_y']
            recon_pad_norm = recon_pad.clamp(-1., 1.) * 0.5 + 0.5
            
            # unpad and save recon image
            recon_norm = recon_pad_norm[:, :, :image.shape[2], :image.shape[3]]
            save_image(recon_norm, recon_file)
            
            # record bpp
            bpp_dict = {'image_name': image_name, 'bpp_hard_y': bpp_hard_y, 'bpp_z': bpp}
            bpp_dict['image_name'] = image_name
            bpp_df = pd.DataFrame(bpp_dict, index=[idx])
            self.detail_df = pd.concat([self.detail_df, bpp_df])
            
            self.accelerator.wait_for_everyone()
        
        # save detail data frame
        self.detail_df.to_excel(self.detail_file)
        
        # save average frame
        avg_dict = self.detail_df.mean(numeric_only=True).to_dict()
        pd.DataFrame([avg_dict]).to_excel(self.summary_file)
        print(avg_dict)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--eval_image_path", type=str, required=False)
    args = parser.parse_args()
        
    args_dict = vars(args)
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.load(args.config_path)
    args = OmegaConf.merge(config, args_conf)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.unet_ckpt = None
    return args 

if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(args)
    evaluator.evaluate()