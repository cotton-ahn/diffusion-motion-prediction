import os
import sys
import pickle
import argparse
import os.path as osp
import torch
import numpy as np

sys.path.append(os.getcwd())
from src.utils.general import set_random_seed
from src.utils.euler import define_actions
from src.config import Config
from src.net import Parallel_Denoiser, Series_Denoiser
from src.diff import DDPM
from src.eval.euler import sample_euler_process
from src.viz.euler import video_euler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_euler_parallel_20step')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--sample-num', type=int, default=50)

    args = parser.parse_args()
    cfg = Config(args.cfg)

    set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu_id) if args.gpu_id >= 0 else 'cpu'

    if 'euler' in args.cfg:
        # fixed prefix and pred length
        prefix_len = 50
        pred_len = 25
        pose_dim = 54

        # params for plot
        n_prefix = 8 # fixed
        row = 1
        col = 5

        actions = define_actions("all")

        data = pickle.load(open('./data/h36m_euler.pkl', 'rb'))

        train_set = data['train']
        test_set = data['test']
        data_mean = data['mean']
        data_std = data['std']
        dim_to_ignore = data['dim_to_ignore']
        dim_to_use = data['dim_to_use']

        ## define network 
        if '_parallel_' in args.cfg:
            denoiser = Parallel_Denoiser(pose_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                   prefix_len, pred_len, cfg.diff_steps)
        elif '_series_' in args.cfg:
            denoiser = Series_Denoiser(pose_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                   prefix_len, pred_len, cfg.diff_steps)
        
        ddpm = DDPM(denoiser, cfg, device).to(device)
        ddpm.load_state_dict(torch.load(osp.join(cfg.model_dir, '0500.pth'.format(cfg.max_epoch))))
        ddpm.eval()

        out_path = osp.join('./pred_results/{}.pkl'.format(args.cfg))   
        vid_path = osp.join('./viz_diff/{}'.format(args.cfg))
        os.makedirs('./pred_results', exist_ok=True)
        os.makedirs(vid_path, exist_ok=True)

        print('==========Start Visualization==========')
        sample_euler_process(out_path, ddpm, args.sample_num, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device)
        video_euler(vid_path, out_path, n_prefix, prefix_len, pred_len)
        print('==========Finish Visualization==========')