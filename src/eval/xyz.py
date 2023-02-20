import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from src.eval.metrics import *
import pickle

# code from DLow
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# code from DLow
def get_multimodal_gt(dataset, prefix_len, multimodal_threshold):
    all_data = []
    data_gen = dataset.iter_generator(step=prefix_len)
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, prefix_len - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, prefix_len:, :])
    return traj_gt_arr


# code from DLow
def get_gt(data, prefix_len):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, prefix_len:, :]

# code from DLow
def get_tensor_pose(traj_np, device):
    traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj_ts = torch.Tensor(traj_np).to(device).permute(1, 0, 2)
    
    return traj_ts

def get_prediction(prefix, models, algo, sample_num, pred_len, pose_dim, concat_hist=True):
    prefix = prefix.repeat((1, sample_num, 1))

    pred = models[algo].sample(prefix, pred_len, pose_dim)
    if concat_hist:
        pred = torch.cat([prefix, pred], dim=0)
    pred = pred.permute(1, 0, 2).contiguous().cpu().numpy()
    if pred.shape[0] > 1:
        pred = pred.reshape(-1, sample_num, pred.shape[-2], pred.shape[-1])
    else:
        pred = pred[None, ...] 
    return pred


def sample_xyz(out_path, algos, models, sample_num, prefix_num, dataset, prefix_len, pred_len, device):
    data_gen = dataset.iter_generator(step=prefix_len)
    all_preds = []
    for i, data in enumerate(data_gen):
        if len(all_preds) == prefix_num:
            break

        if np.random.rand() > 0.95: # to random random
            prefix = get_tensor_pose(data, device)[:prefix_len] # T B D

            with torch.no_grad():
                for algo in algos:
                    pred = get_prediction(prefix, models, algo, sample_num, pred_len, dataset.traj_dim, concat_hist=True)
            pred = pred.reshape(1, sample_num, prefix_len+pred_len, -1, 3)
            pred[..., :1, :] = 0
            all_preds.append(pred)
    pickle.dump(np.concatenate(all_preds, axis=0), open(out_path, 'wb'))


def compute_stats_xyz(f, algos, models, dataset, traj_gt_arr,  prefix_len, pred_len, sample_num, device):
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade, 'MDE': compute_mde, 'SDE': compute_sde,
                  'FDE': compute_fde, 'MFDE': compute_mfde, 'SFDE': compute_sfde, 'MMADE': compute_mmade, 'MMMDE': compute_mmmde, 
                  'MMFDE': compute_mmfde, 'MMMFDE': compute_mmmfde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    data_gen = dataset.iter_generator(step=prefix_len)
    num_samples = 0
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data, prefix_len)
        gt_multi = traj_gt_arr[i]

        prefix = get_tensor_pose(data, device)[:prefix_len] # T B D

        save_path = './pred_results/h36m_nsamp50/pred_{}.pkl'.format(i)

        with torch.no_grad():
            for algo in algos:
                pred = get_prediction(prefix, models, algo, sample_num, pred_len, dataset.traj_dim, concat_hist=False)
                saver = {'raw': data[0], 'pred':pred[0], 'gt':gt[0]}
                for stats in stats_names:
                    val = 0
                    for pred_i in pred:
                        val += stats_func[stats](pred_i, gt, gt_multi)
                    stats_meter[stats][algo].update(val)
        print('-' * 80, file=f)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats, file=f)
        f.flush()
        pickle.dump(saver, open(save_path, 'wb'))
        
    print('=' * 80, file=f)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        print(str_stats, file=f)
    print('=' * 80, file=f)
    f.flush()
