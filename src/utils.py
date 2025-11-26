import wandb
import torch
import importlib
from omegaconf import OmegaConf
from safetensors import safe_open
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize
from torch.utils.tensorboard import SummaryWriter


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_safetensor(path, map_location="cpu", force_safetensor=False):
    if force_safetensor or path.endswith("safetensors"):
        tensors = {}
        with safe_open(path, framework="pt", device=map_location) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors
    else:
        return torch.load(path, map_location=map_location)


class AvgDict(dict):
    def __init__(self, *args, **kwargs):
        super(AvgDict, self).__init__(*args, **kwargs)
        self.num = 0
    
    def record(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().mean().item()
            if k not in self:
                self[k] = 0.0
            self[k] += v
        self.num += 1
        
    def avg(self):
        return {k: v / self.num for k, v in self.items()}


class WrappedTensorboard:
    def __init__(self, log_path, args, max_images=4, image_data_range=(-1, 1)):
        self.log_path = log_path
        self.max_images = max_images
        self.image_range = image_data_range
        self.writer = SummaryWriter(log_path)
        OmegaConf.save(args, f"{log_path}/args.yaml")


    def log_dict(self, log_dict, step, split=None):
        for k, v in log_dict.items():
            if split is not None:
                k = f"{split}/{k}"
            if isinstance(v, torch.Tensor):
                v = v.detach().mean().item()
            self.writer.add_scalar(k, v, global_step=step)
            
            
    def log_image(self, log_dict, step, split=None, size=None):
        for k, v in log_dict.items():
            if split is not None:
                k = f"{split}/{k}"
            if size is not None:
                v = resize(v, size)
            v = v[:self.max_images].detach().clamp(*self.image_range)
            v = (v - self.image_range[0]) / (self.image_range[1] - self.image_range[0])
            v = make_grid(v)
            self.writer.add_image(k, v, global_step=step)
        
        
class WrappedWandb:
    def __init__(self, log_path, args,
                 max_images=4, image_data_range=(-1, 1), is_debug=False):
        self.log_path = log_path
        self.max_images = max_images
        self.image_range = image_data_range
        OmegaConf.save(args, f"{log_path}/args.yaml")
        
        wandb_model = "offline" if is_debug else "online"
        run = wandb.init(
            dir=log_path, 
            **{"mode": wandb_model, "entity": args.wandb_entity, "project": args.wandb_project}
        )
        wandb.run.log_code(".")
        wandb.run.name = args.name
        print(f"run dir: {run.dir}")
        self.wandb_folder = run.dir


    def log_dict(self, log_dict, step, split=None):
        dict_new = {}
        for k, v in log_dict.items():
            if split is not None:
                k = f"{split}/{k}"
            if isinstance(v, torch.Tensor):
                v = v.detach().mean().item()
            dict_new[k] = v
        wandb.log(dict_new, step=step)
            
            
    def log_image(self, log_dict, step, split=None, size=None):
        dict_new = {}
        for k, v in log_dict.items():
            if split is not None:
                k = f"{split}/{k}"
            if size is not None:
                v = resize(v, size)
            v = v[:self.max_images].detach().clamp(*self.image_range)
            v = (v - self.image_range[0]) / (self.image_range[1] - self.image_range[0])     # to [0,1]
            v = make_grid(v)
            v = (v * 255.0).permute(1, 2, 0).cpu().numpy().astype("uint8")                  # to [0,255]
            dict_new[k] = wandb.Image(v)
        wandb.log(dict_new, step=step)