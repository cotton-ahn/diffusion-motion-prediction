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
from src.eval.xyz import compute_stats_xyz, get_multimodal_gt, sample_xyz
from src.eval.euler import compute_stats_euler, sample_euler
from src.dataset.xyz import DatasetH36M, DatasetHumanEva
from src.viz.euler import visualize_euler
from src.viz.xyz import visualize_xyz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_xyz_parallel_20step')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--sample-num', type=int, default=50)
    parser.add_argument('--mode', type=str, default='stats')

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
        ddpm.load_state_dict(torch.load(osp.join(cfg.model_dir, 'recent.pth'.format(cfg.max_epoch))))
        ddpm.eval()

        if args.mode == 'stats':
            '''
            Calculating the metrics for deterministic prediction.
            multi = False, use_zero=True.
            '''
            print('=======Start Calculating Metrics=======')
            f = open('./log/[EVAL]{}.txt'.format(args.cfg), 'w')
            multi = False 
            compute_stats_euler(multi, f, ddpm, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device)
            f.close()
            print('=======Finish Calculating Metrics=======')

        elif args.mode == 'viz':
            '''
            Visualization of motions that are sampled from stochastic prediction.
            '''
            out_path = osp.join('./pred_results/{}.pkl'.format(args.cfg))   
            vid_path = osp.join('./vids/{}'.format(args.cfg))
            os.makedirs('./pred_results', exist_ok=True)
            os.makedirs('./tmp_imgs', exist_ok=True)
            os.makedirs(vid_path, exist_ok=True)

            print('==========Start Visualization==========')
            sample_euler(out_path, ddpm, args.sample_num, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device)
            visualize_euler(vid_path, out_path, n_prefix, row, col, prefix_len, pred_len)
            print('==========Finish Visualization==========')

    elif 'xyz' in args.cfg:
        # fixed!
        prefix_len = 25
        pred_len = 100

        # fixed for poses to visualize
        prefix_num = 100
        row = 1
        col = 5

        if 'h36m' in args.cfg:
            dataset_cls = DatasetH36M
        elif 'humaneva' in args.cfg:
            dataset_cls = DatasetHumanEva
        dataset = dataset_cls('test', prefix_len, pred_len, actions='all')

        ## define network 
        if '_parallel_' in args.cfg:
            denoiser = Parallel_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                          prefix_len, pred_len, cfg.diff_steps)
        elif '_series_' in args.cfg:
            denoiser = Series_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads, 
                                         prefix_len, pred_len, cfg.diff_steps)

        ddpm = DDPM(denoiser, cfg, device).to(device)
        ddpm.load_state_dict(torch.load(osp.join(cfg.model_dir, '0200.pth'.format(cfg.max_epoch))))
        ddpm.eval()

        traj_gt_arr = get_multimodal_gt(dataset, prefix_len, 0.5)
        algos = [args.cfg.split('_')[2]]
        models = {algos[0]: ddpm}
 
        if args.mode == 'stats':
            f = open('./log/[EVAL]{}:nsamp{}.txt'.format(args.cfg, args.sample_num), 'w')
            compute_stats_xyz(f, algos, models, dataset, traj_gt_arr, prefix_len, pred_len, args.sample_num, device)
            f.close()

        elif args.mode == 'viz':
            out_path = osp.join('./pred_results/{}.pkl'.format(args.cfg))   
            vid_path = osp.join('./vids/{}'.format(args.cfg))
            os.makedirs('./pred_results', exist_ok=True)
            os.makedirs('./tmp_imgs', exist_ok=True)
            os.makedirs(vid_path, exist_ok=True)

            print('==========Start Visualization==========')
            sample_xyz(out_path, algos, models, args.sample_num, prefix_num, dataset, prefix_len, pred_len, device )
            visualize_xyz(vid_path, out_path, prefix_num, row, col, prefix_len, pred_len)
            print('==========Finish Visualization==========')
