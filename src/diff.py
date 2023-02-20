import numpy as np
import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, denoiser, cfg):
        super().__init__()

        self.denoiser = denoiser
        self.cfg = cfg
        if cfg.beta_schedule =='quad':
            self.beta = np.linspace(cfg.beta_start**0.5, cfg.beta_end**0.5, cfg.diff_steps) ** 2
        elif cfg.beta_schedule == 'linear':
            self.beta = np.linspace(cfg.beta_start, cfg.beta_end, cfg.diff_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.FloatTensor(self.alpha)
        print('==============alpha schedule============')
        print(self.alpha ** 0.5)
        print((1-self.alpha)**0.5)
        print('========================================')
        
    def calc_loss(self, prefix, gt):
        L_prefix, B, D = prefix.shape
        L_pred, B, D = gt.shape
        t = torch.randint(0, self.cfg.diff_steps, [B]).to(prefix.device)
        curr_alpha = self.alpha_torch[t][None, :, None].to(prefix.device)

        noise = torch.randn_like(gt)

        diff_sample = (curr_alpha ** 0.5) * gt + (1.0 - curr_alpha) ** 0.5 * noise
        denoise_result = self.denoiser(diff_sample, prefix, t)

        loss = torch.mean((noise - denoise_result)**2)

        return loss


    def sample(self, prefix, pred_len, pose_dim, use_zero=False, add_denoise_process=False):
        num_samples = prefix.shape[1]

        if use_zero:
            curr_sample = torch.zeros(pred_len, num_samples, pose_dim).to(prefix.device)
            
        else:
            curr_sample = torch.randn(pred_len, num_samples, pose_dim).to(prefix.device)

        denoise_process = [curr_sample]
        
        for t in range(self.cfg.diff_steps-1, -1, -1):
            time_vec = torch.LongTensor([t for _ in range(num_samples)]).to(prefix.device)

            diff_result = self.denoiser(curr_sample, prefix, time_vec)

            coeff1 = 1 / self.alpha_hat[t] **0.5 
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            
            curr_sample = coeff1 * (curr_sample - coeff2 * diff_result)

            if t > 0 and not use_zero:
                sigma = (
                    ((1.0 - self.alpha[t-1]) / (1.0 - self.alpha[t])) * self.beta[t]
                ) ** 0.5
                curr_sample += sigma * torch.randn_like(curr_sample)
            
            denoise_process.append(curr_sample)

        if add_denoise_process:
            return curr_sample, denoise_process
        else:
            return curr_sample




