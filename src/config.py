# Code Reference : [DLow]('https://github.com/Khrylx/DLow')

import yaml
import os
import os.path as osp

class Config:
    def __init__(self, cfg_id):
        self.id = cfg_id
        cfg_name = 'cfg/{}.yml'.format(cfg_id)
        if not osp.exists(cfg_name):
            raise ValueError("Config file {} does not exist".format(cfg_name))
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        self.model_dir = './trained_models/{}'.format(cfg_id)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # general
        self.batch_size = cfg['batch_size']
        self.max_epoch = cfg['max_epoch']
        self.save_epoch = cfg['save_epoch']
        self.epoch_iters = cfg['epoch_iters']
        self.learning_rate = cfg['learning_rate']        
        
        if 'step' in cfg_id:
            self.diff_steps = cfg['diff_steps']
            self.beta_start = cfg['beta_start']
            self.beta_end = cfg['beta_end']
            self.beta_schedule = cfg['beta_schedule']
        if 'TF' in cfg_id:
            self.qkv_dim = cfg['qkv_dim']
            self.num_layers = cfg['num_layers']
            self.num_heads = cfg['num_heads']
        
        if 'GRU' in cfg_id:
            self.hidden_dim = cfg['hidden_dim']
            self.num_layers = cfg['num_layers']
