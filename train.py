import os
import sys
import pickle
import argparse
import os.path as osp
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils.general import set_random_seed
from src.utils.euler import define_actions
from src.config import Config
from src.net import TF2CH_Denoiser, STTF_Denoiser
from src.diff import DDPM
from src.train.euler import train_euler
from src.train.xyz import train_xyz
from src.dataset.xyz import DatasetH36M, DatasetHumanEva


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_euler_STTF_50step')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    
    args = parser.parse_args()
    cfg = Config(args.cfg)

    set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu_id) if args.gpu_id >= 0 else 'cpu'

    f = open('./logs/[TRAIN]{}.txt'.format(args.cfg), 'w')

    if 'euler' in args.cfg:
        # fixed prefix and pred length
        prefix_len = 50
        pred_len = 25
        pose_dim = 54

        actions = define_actions("all")

        data = pickle.load(open('./data/h36m_euler.pkl', 'rb'))

        train_set = data['train']
        test_set = data['test']
        data_mean = data['mean']
        data_std = data['std']
        dim_to_ignore = data['dim_to_ignore']
        dim_to_use = data['dim_to_use']

        ## define network 
        if '_TF2CH_' in args.cfg: # "parallel" model in paper
            denoiser = TF2CH_Denoiser(pose_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                     prefix_len, pred_len, cfg.diff_steps)
        elif '_STTF_' in args.cfg: # "series" model in paper.
            denoiser = STTF_Denoiser(pose_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                     prefix_len, pred_len, cfg.diff_steps)
         
        ddpm = DDPM(denoiser, cfg).to(device)

        optimizer = torch.optim.Adam(ddpm.parameters(), lr=cfg.learning_rate)

        for epoch in tqdm(range(cfg.max_epoch)):
            train_euler(f, epoch, cfg.epoch_iters, ddpm, optimizer, train_set, cfg.batch_size, pose_dim,  
                        prefix_len, pred_len, actions, device)
            if (epoch+1)%cfg.save_epoch == 0:
                torch.save(ddpm.state_dict(), osp.join(cfg.model_dir, '{:04}.pth'.format(epoch+1)))
            torch.save(ddpm.state_dict(), osp.join(cfg.model_dir, 'recent.pth'))
        f.close()
        
    elif 'xyz' in args.cfg:
        # fixed values! do not change.
        prefix_len = 25
        pred_len = 100

        if 'h36m' in args.cfg:
            dataset_cls = DatasetH36M
        elif 'humaneva' in args.cfg:
            dataset_cls = DatasetHumanEva
        dataset = dataset_cls('train', prefix_len, pred_len, actions='all')

        ## define network 
        if '_TF2CH_' in args.cfg: # "parallel" model in paper
            denoiser = TF2CH_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                     prefix_len, pred_len, cfg.diff_steps)
        elif '_STTF_' in args.cfg: # "series" model in paper
            denoiser = STTF_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                     prefix_len, pred_len, cfg.diff_steps)
       

        ddpm = DDPM(denoiser, cfg).to(device)

        optimizer = torch.optim.Adam(ddpm.parameters(), lr=cfg.learning_rate)

        for epoch in tqdm(range(cfg.max_epoch)):
            train_xyz(f, epoch, cfg.epoch_iters, ddpm, optimizer, dataset, cfg.batch_size, prefix_len, device)
            if (epoch+1)%cfg.save_epoch == 0:
                torch.save(ddpm.state_dict(), osp.join(cfg.model_dir, '{:04}.pth'.format(epoch+1)))
            torch.save(ddpm.state_dict(), osp.join(cfg.model_dir, 'recent.pth'))

        f.close()
        