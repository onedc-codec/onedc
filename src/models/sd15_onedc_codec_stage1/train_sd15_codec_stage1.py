import sys, os, shutil, datetime
import time, argparse, logging
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')
import random as rd
 
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.transforms import RandomCrop, Resize
from torchvision.transforms.functional import resize

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from accelerate import DistributedDataParallelKwargs as DDPK

from diffusers.optimization import get_scheduler

from utils import WrappedTensorboard, WrappedWandb
from utils import AvgDict, instantiate_from_config
from data.base import instantiate_datasets
from modules.dmd.utils import cycle

from models.sd15_onedc_codec_stage1.model_sd15_with_codec_stage1 import SD15_1step_codec_stage1
from losses.pixel_loss import SQ_Perceptual_loss

logger = get_logger(__name__, log_level="INFO")


class Trainer:
    def __init__(self, args):
        self.args = args

        # torch config
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        # accelerate config
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            kwargs_handlers=[DDPK(find_unused_parameters=True)]
        )
        set_seed(args.seed + accelerator.process_index)
        self.accelerator = accelerator
        
        # logger config
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        # output path
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.output_path = os.path.join(args.output_path, f"{now}_{args.name}")
        if args.debug:
            self.output_path = os.path.join(args.output_path, f"debug")
        self.save_path = os.path.join(self.output_path, "checkpoints")              # checkpoint path 
        self.save_best_path = os.path.join(self.output_path, "checkpoints_best")    # best checkpoint path
        self.log_path = os.path.join(self.output_path, f"logs")                     # log path
            
        # build folder
        if accelerator.is_main_process:
            if os.path.exists(self.output_path):                # remove previous output folder
                shutil.rmtree(self.output_path)
            os.makedirs(self.output_path, exist_ok=False)       # create new output folder
            os.makedirs(self.save_path, exist_ok=True)          # create checkpoint folder
            os.makedirs(self.save_best_path, exist_ok=True)     # create best checkpoint folder
            os.makedirs(self.log_path, exist_ok=True)           # create log folder
            OmegaConf.save(args, f"{self.output_path}/config.yaml")
        accelerator.wait_for_everyone()
            
        # make logs
        if accelerator.is_main_process:
            if args.use_wandb:
                self.logger = WrappedWandb(self.log_path, args, max_images=args.num_visuals, image_data_range=(-1, 1))
            else:
                self.logger = WrappedTensorboard(self.log_path, args, max_images=args.num_visuals, image_data_range=(-1, 1))
        else:
            self.logger = None

        # build model
        self.model = SD15_1step_codec_stage1(args, accelerator)
        self.model.prepare()
        
        # prepare loss
        self.pix_loss_func = SQ_Perceptual_loss(**args.pix_loss).to(accelerator.device)
        self.use_codeformer = args.use_codeformer
        if self.use_codeformer:
            self.codeformer_loss_weight = args.codeformer_loss_weight
            self.codeformer_mse_weight = args.codeformer_mse_weight

        # training params
        self.prepare_optimizer(args.generator_lr)

        # record best loss
        self.best_loss = 9999.0
        self.monitor_key_lower = args.monitor_key_lower

        # record settings
        self.num_visuals = args.num_visuals
        self.step = 0 
        self.max_grad_norm = args.max_grad_norm
        self.train_iters = args.train_iters
        self.resolution = args.resolution
        self.batch_size = args.batch_size
        self.save_interval = args.save_interval
        self.log_interval = args.log_interval
        self.visual_interval = args.visual_interval
        self.no_save = args.no_save
        self.max_checkpoint = args.max_checkpoint
        self.disable_tqdm = args.disable_tqdm
        self.debug = args.debug

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)
            accelerator.print(f"[INFO] Continue training from step {self.step}")
            
            # override lr
            if args.override_lr is not None:
                self.prepare_optimizer(args.generator_lr)
                accelerator.print(f"[INFO] Overrided lr to {args.generator_lr}")
                
            # override step
            if args.override_step is not None:
                self.step = args.override_step
                accelerator.print(f"[INFO] Overrided step to {args.override_step}")
        else:
            accelerator.print(f"[INFO] Start training from scratch.")

        img_dataset = instantiate_datasets(args.train_dataset)
        img_dataloader = torch.utils.data.DataLoader(
            img_dataset, num_workers=args.num_workers, 
            batch_size=args.batch_size, shuffle=True, 
            drop_last=True
        )
        img_dataloader = accelerator.prepare(img_dataloader)
        self.img_dataloader = cycle(img_dataloader)
        
        # random transform settings
        self.use_random_transform = args.use_random_transform
        self.transforms = []
        self.transforms_prob = []
        self.batch_reduction = []
        if self.use_random_transform:
            for res, prob, batch_factor in zip(args.random_resize_list, args.random_resize_prob, 
                                               args.random_resize_batch_reduction):
                self.transforms.append(Resize(res))
                self.transforms_prob.append(prob)
                self.batch_reduction.append(batch_factor)
            for res, prob, batch_factor in zip(args.random_crop_list, args.random_crop_prob, 
                                               args.random_crop_batch_reduction):
                self.transforms.append(RandomCrop(res))
                self.transforms_prob.append(prob)
                self.batch_reduction.append(batch_factor)
            accelerator.print(f"[INFO] Use random resize for training")
        
        self.use_eval = False
        if args.eval_dataset is not None:
            self.use_eval = True
            eval_dataset = instantiate_datasets(args.eval_dataset)
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, num_workers=args.num_workers, 
                batch_size=args.batch_size, shuffle=False, 
                drop_last=True
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)
            self.eval_dataloader = eval_dataloader
            
        accelerator.wait_for_everyone()


    def prepare_optimizer(self, generator_lr):
        accelerator = self.accelerator
        
        # generator optimizer
        self.optimizer_generator = torch.optim.AdamW(
            self.model.get_trainable_params()[0], 
            lr=generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )
        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )
        self.optimizer_generator = accelerator.prepare(self.optimizer_generator)
        self.scheduler_generator = accelerator.prepare(self.scheduler_generator)


    def load(self, checkpoint_path):
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"[INFO] Loaded checkpoint from {checkpoint_path}")


    def save(self, is_best=False):
        save_path = self.save_best_path if is_best else self.save_path

        # delete previous checkpoints if the number of checkpoints exceed a certain amount 
        checkpoints = sorted(
            [folder for folder in os.listdir(save_path) if folder.startswith("checkpoint_model")]
        )
        if is_best or len(checkpoints) > self.max_checkpoint:
            for folder in checkpoints[:-self.max_checkpoint] if not is_best else checkpoints:
                shutil.rmtree(os.path.join(save_path, folder))
                
        # save the latest checkpoint
        output_path = os.path.join(save_path, f"checkpoint_model_{self.step:06d}")
        self.accelerator.save_state(output_path)


    @torch.no_grad()
    def eval_one_epoch(self):
        accelerator = self.accelerator
        record_dict = AvgDict()
        
        accelerator.wait_for_everyone()
        accelerator.print(f"[INFO] Start evaluation")
        self.model.eval()
        self.pix_loss_func.eval()
        
        # load data from eval dataset
        for idx, sample_dict in enumerate(self.eval_dataloader):
            pix_images = sample_dict['image']
            
            enc_dict, pred_image = self.model(pix_images)
            
            with self.model.network_context_manager:        
                loss, loss_dict = self.pix_loss_func(pix_images, pred_image, enc_dict['bpp'], 
                                                 step=self.step, norm01=True)
            loss_dict['bpp_hard_y'] = enc_dict['bpp_hard_y']
            record_dict.record(loss_dict)
                
            if accelerator.is_main_process:
                if idx == 0:
                    images = {
                        "original": pix_images,
                        "recon": pred_image
                    }
                    self.logger.log_image(images, step=self.step, split="eval")
            
            # for fast end of debugging     
            if self.debug and idx > 10:
                break
        
        accelerator.wait_for_everyone()
        avg_dict = record_dict.avg()
        avg_all_dict = {
            k: accelerator.reduce(torch.tensor(v, device=accelerator.device), reduction="mean").item() 
            for k, v in avg_dict.items()
        }
        self.accelerator.wait_for_everyone()
        
        # log the average loss
        if accelerator.is_main_process:
            self.logger.log_dict(avg_all_dict, step=self.step, split="eval")
        
        return avg_all_dict


    def train_one_step(self):
        self.model.train()
        self.pix_loss_func.train()
        accelerator = self.accelerator

        # load data
        sample_dict = next(self.img_dataloader)
        sample_gt_images = sample_dict['image']
        
        # resize image if need
        if self.use_random_transform:
            trans_idx = rd.choices(range(len(self.transforms)), self.transforms_prob)[0]
            trans_idx = broadcast(
                torch.tensor(trans_idx, device=accelerator.device, dtype=torch.int32),
                from_process=0
            ).item()
            trans = self.transforms[trans_idx]
            sample_gt_images = trans(sample_gt_images)
            bs = sample_gt_images.shape[0]
            bs_reduction_factor = self.batch_reduction[trans_idx]
            bs = int(bs * bs_reduction_factor)
            sample_gt_images = sample_gt_images[:bs]
        
        enc_dict, pred_images = self.model(sample_gt_images)

        with self.model.network_context_manager:        
            loss, loss_dict = self.pix_loss_func(sample_gt_images, pred_images, enc_dict['bpp'], 
                                             step=self.step, norm01=True)
            # codeformer loss
            if self.use_codeformer:
                codeformer_ce_loss = enc_dict['code_ce_loss']
                codeformer_mse_loss = enc_dict['code_mse_loss']
                codeformer_loss = codeformer_ce_loss + codeformer_mse_loss * self.codeformer_mse_weight
                weighted_codeformer_loss = codeformer_loss * self.codeformer_loss_weight
                loss += weighted_codeformer_loss
                loss_dict['codeformer_ce_loss'] = codeformer_ce_loss
                loss_dict['codeformer_mse_loss'] = codeformer_mse_loss
                loss_dict['codeformer_loss'] = codeformer_loss
                loss_dict['weighted_codeformer_loss'] = weighted_codeformer_loss
        
        loss_dict['bpp_hard_y'] = enc_dict['bpp_hard_y']
        
        self.optimizer_generator.zero_grad()
        self.accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = torch.zeros([1], device=accelerator.device)
        self.optimizer_generator.step()
        self.scheduler_generator.step()
        loss_dict['grad_norm'] = grad_norm
        loss_dict['lr'] = self.optimizer_generator.param_groups[0]['lr']
        if self.use_random_transform:
            loss_dict['resolution'] = sample_gt_images.shape[-1]

        if accelerator.is_main_process:
            if self.step % self.log_interval == 0:
                self.logger.log_dict(loss_dict, step=self.step, split="train")

            if self.step % self.visual_interval == 0:
                images_for_log = {
                    "original": resize(sample_gt_images, self.resolution),
                    "recon": resize(pred_images, self.resolution),
                }
                self.logger.log_image(images_for_log, step=self.step, split="train")

        self.accelerator.wait_for_everyone()


    def train(self):
        disable_tqdm = (not self.accelerator.is_main_process) or self.disable_tqdm
        for index in tqdm(range(self.step, self.train_iters), total=self.train_iters, disable=disable_tqdm):                
            self.train_one_step()
            if (not self.no_save)  and self.step % self.save_interval == 0:
                if self.accelerator.is_main_process:
                    self.save()
                    
            if self.use_eval and self.step % self.save_interval == 0:
                eval_dict = self.eval_one_epoch()
                
                if self.accelerator.is_main_process:
                    if eval_dict[self.monitor_key_lower] < self.best_loss:
                        
                        self.best_loss = eval_dict[self.monitor_key_lower]
                        self.accelerator.print(f"[INFO] In Step: {self.step}, best loss updated: {self.best_loss}")
                        
                        if not self.no_save:
                            self.save(is_best=True)

            self.accelerator.wait_for_everyone()
            self.step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--debug", action="store_true", help="don't save ckpt for debugging only")
    args = parser.parse_args()
        
    args_dict = vars(args)
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.load(args.config_path)
    args = OmegaConf.merge(config, args_conf)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"
    return args 


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.train()