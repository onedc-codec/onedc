import torch
from torch import nn
from piq import LPIPS


class Linear_lmbda_scheduler():
    def __init__(self, start_step, end_step, start_value, end_value):
        assert end_step > start_step
        self.start_step = float(start_step)
        self.end_step = float(end_step)
        self.interval = float(end_step - start_step)
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.value_increase = float(end_value - start_value)
    
    def __call__(self, cur_step):
        if cur_step <= self.start_step:
            factor = 0.0
        elif cur_step <= self.end_step:
            factor = (cur_step - self.start_step) / self.interval
        else:
            factor = 1.0
        factor = factor ** 2
        return self.start_value + factor * self.value_increase


class SQ_Perceptual_loss(nn.Module):
    def __init__(self, pix_weight, lpips_weight, lmbda, lmbda_schedule=None,
                 pix_loss_type="l1"):
        super().__init__()
        """pix + LPIPS loss.
        """
        
        # l1 loss
        assert pix_loss_type in ["l1", "mse"]
        self.pix_func = nn.L1Loss(reduction="mean") if pix_loss_type == "l1" else nn.MSELoss(reduction="mean")
        self.pix_weight = pix_weight
        
        # lpips loss
        self.lpips_func = LPIPS(replace_pooling=True, reduction="mean").eval()
        self.lpips_weight = lpips_weight
        
        self.lmbda = lmbda
        if lmbda_schedule:
            self.lmbda_scheduler = Linear_lmbda_scheduler(**lmbda_schedule)
            self.lmbda = self.lmbda_scheduler.end_value
        else:
            self.lmbda_scheduler = None
        
    
    def forward(self, x, x_hat, bpp, step=None, norm01=True):
        ''' If input is in [-1, 1], set norm01=True; if input is in [0, 1], set norm01=False.
        '''
        if norm01:
            x = x * 0.5 + 0.5               # to [0, 1]
            x_hat = x_hat * 0.5 + 0.5       # to [0, 1]
        
        # 1. pix loss
        l_pix = self.pix_func(x, x_hat).float().mean()
        l_weighted_pix = l_pix * self.pix_weight
        
        # 2. lpips loss
        l_lpips = self.lpips_func(x, x_hat).float().mean()
        l_weighted_lpips = l_lpips * self.lpips_weight

        # 4. bpp loss and lambda
        if step and self.lmbda_scheduler and self.training:
            lmbda_this = self.lmbda_scheduler(step)
        else:
            lmbda_this = self.lmbda
        l_bpp = bpp
        l_weighted_bpp = bpp * lmbda_this
        
        # 5. weighted losses
        l_weighted_distortion = l_weighted_pix + l_weighted_lpips
        loss = l_weighted_distortion + l_weighted_bpp

        loss_dict = {
            "pix": l_pix,
            "lpips": l_lpips,
            "bpp": l_bpp,
            "weighted_pix": l_weighted_pix,
            "weighted_lpips": l_weighted_lpips,
            "distortion": l_weighted_distortion,
            "weighted_bpp": l_weighted_bpp,
            "lmbda": lmbda_this,
            "total_loss": loss,
        }
        return loss, loss_dict