## do test
from src.dataset.euler import get_batch_srnn, get_srnn_gts
from src.utils.euler import unNormalizeDataChunk, rotmat2euler, expmap2rotmat, rotmat2expmap
import torch
import numpy as np
import pickle

def sample_euler_supervised_inf(out_path, model, num_iters, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device):
    samples = {}
    with torch.no_grad():
        for action in actions:
            val_prefix, _ = get_batch_srnn(test_set, pose_dim, prefix_len, pred_len, action)
            val_prefix = torch.FloatTensor(val_prefix).to(device).permute(1, 0, 2) # L B D

            val_prefix_expmap = unNormalizeDataChunk(val_prefix.permute(1, 0, 2).detach().cpu(),
                                                        data_mean, data_std, dim_to_ignore, action)                
            expmap_results = [val_prefix_expmap]

            for i in range(num_iters):
                # val_prefix = 50 8 99, pred : 25 8 99
                last_pose = val_prefix[-1][None, :].repeat(pred_len, 1, 1)
                window = torch.cat([val_prefix, last_pose], dim=0)

                pred_result = model(window)

                srnn_pred_expmap = unNormalizeDataChunk(pred_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, actions)

                expmap_results.append(srnn_pred_expmap)

                val_prefix = torch.cat([val_prefix[(prefix_len-pred_len):], pred_result], dim=0)

            samples[action] = np.concatenate(expmap_results, axis=1).reshape(1, -1, prefix_len + pred_len*num_iters, 99)

    pickle.dump(samples, open(out_path, 'wb'))


def sample_euler_gt(out_path, model, num_samples, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device):
    samples = {}
    deters = {}
    gts = {}
    with torch.no_grad():
        for action in actions:
            val_prefix, val_gt = get_batch_srnn(test_set, pose_dim, prefix_len, pred_len, action)
            val_total_gt = np.concatenate([val_prefix, val_gt], axis=1)

            val_prefix = torch.FloatTensor(val_prefix).to(device).permute(1, 0, 2) # L B D
            val_gt = torch.FloatTensor(val_gt).repeat(num_samples, 1, 1)
            
            val_prefix = val_prefix.repeat(1, num_samples, 1)
            
            pred_result = model.sample(val_prefix, pred_len, pose_dim, use_zero=False)
            deter_result = model.sample(val_prefix, pred_len, pose_dim, use_zero=True)

            val_total_gt_expmap = unNormalizeDataChunk(val_total_gt, data_mean, data_std, dim_to_ignore, action)
            val_prefix_expmap = unNormalizeDataChunk(val_prefix.permute(1, 0, 2).detach().cpu(),
                                                    data_mean, data_std, dim_to_ignore, action)
            srnn_pred_expmap = unNormalizeDataChunk(pred_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, actions)
            srnn_deter_expmap = unNormalizeDataChunk(deter_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, actions)

            samples[action] = np.concatenate([val_prefix_expmap, srnn_pred_expmap], axis=1).reshape(num_samples, -1, prefix_len+pred_len, 99)
            deters[action] = np.concatenate([val_prefix_expmap, srnn_deter_expmap], axis=1).reshape(num_samples, -1, prefix_len+pred_len, 99)
            gts[action] = val_total_gt_expmap
    result = {'samples':samples, 'deters': deters, 'gts':gts}
    pickle.dump(result, open(out_path, 'wb'))
    
def sample_euler(out_path, model, num_samples, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device):
    samples = {}
    with torch.no_grad():
        for action in actions:
            val_prefix, _ = get_batch_srnn(test_set, pose_dim, prefix_len, pred_len, action)
            val_prefix = torch.FloatTensor(val_prefix).to(device).permute(1, 0, 2) # L B D

            val_prefix = val_prefix.repeat(1, num_samples, 1)
            
            pred_result = model.sample(val_prefix, pred_len, pose_dim, use_zero=False)

            val_prefix_expmap = unNormalizeDataChunk(val_prefix.permute(1, 0, 2).detach().cpu(),
                                                    data_mean, data_std, dim_to_ignore, action)
            srnn_pred_expmap = unNormalizeDataChunk(pred_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, actions)

            samples[action] = np.concatenate([val_prefix_expmap, srnn_pred_expmap], axis=1).reshape(num_samples, -1, prefix_len+pred_len, 99)
    pickle.dump(samples, open(out_path, 'wb'))


def sample_euler_inf(out_path, model, num_samples, num_iters, actions, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, pose_dim, prefix_len, pred_len, device):
    samples = {}
    with torch.no_grad():
        for action in actions:
            val_prefix, _ = get_batch_srnn(test_set, pose_dim, prefix_len, pred_len, action)
            val_prefix = torch.FloatTensor(val_prefix).to(device).permute(1, 0, 2) # L B D

            val_prefix = val_prefix.repeat(1, num_samples, 1)

            val_prefix_expmap = unNormalizeDataChunk(val_prefix.permute(1, 0, 2).detach().cpu(),
                                                        data_mean, data_std, dim_to_ignore, action)                
            expmap_results = [val_prefix_expmap]

            for i in range(num_iters):
                # val_prefix = 50 8 99, pred : 25 8 99
                pred_result = model.sample(val_prefix, pred_len, pose_dim, use_zero=True)

                srnn_pred_expmap = unNormalizeDataChunk(pred_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, actions)

                # new_pred_expmap = np.zeros(srnn_pred_expmap.shape)

                # for i in range(new_pred_expmap.shape[0]):
                #     for ii in range(new_pred_expmap.shape[1]):
                #         for k in range(3, 97, 3):
                #             new_pred_expmap[i, ii, k:k+3] = rotmat2expmap(expmap2rotmat(srnn_pred_expmap[i, ii, k:k+3]))

                # pred_result = torch.Tensor(np.divide(new_pred_expmap-data_mean, data_std)[:, :, dim_to_use]).permute(1, 0, 2).to(device)

                expmap_results.append(srnn_pred_expmap)

                val_prefix = torch.cat([val_prefix[(prefix_len-pred_len):], pred_result], dim=0)

            samples[action] = np.concatenate(expmap_results, axis=1).reshape(num_samples, -1, prefix_len + pred_len*num_iters, 99)

    pickle.dump(samples, open(out_path, 'wb'))


def sample_euler_process(out_path, model, num_samples, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device):
    data = {'sample': {}, 'denoise_process': {}}
    with torch.no_grad():
        for action in actions:
            val_prefix, _ = get_batch_srnn(test_set, pose_dim, prefix_len, pred_len, action)
            val_prefix = torch.FloatTensor(val_prefix).to(device).permute(1, 0, 2) # L B D

            val_prefix = val_prefix.repeat(1, num_samples, 1)
            
            pred_result, denoise_process = model.sample(val_prefix, pred_len, pose_dim, use_zero=False, add_denoise_process=True)

            denoise_process_expmap = []
            for dp in denoise_process:
                dp_expmap = unNormalizeDataChunk(dp.permute(1, 0, 2).detach().cpu(), data_mean, data_std, dim_to_ignore, action)
                dp_expmap = dp_expmap.reshape(num_samples, -1, pred_len, 99)                
                denoise_process_expmap.append(dp_expmap)                

            val_prefix_expmap = unNormalizeDataChunk(val_prefix.permute(1, 0, 2).detach().cpu(),
                                                    data_mean, data_std, dim_to_ignore, action)
            srnn_pred_expmap = unNormalizeDataChunk(pred_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, action)

            data['sample'][action] = np.concatenate([val_prefix_expmap, srnn_pred_expmap], axis=1).reshape(num_samples, -1, prefix_len+pred_len, 99)
            data['denoise_process'][action] = denoise_process_expmap
    pickle.dump(data, open(out_path, 'wb'))


def compute_stats_euler(multi, f, model, actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, device ):
    print('', file=f)
    print("{0: <16} |".format("milliseconds"), end="", file=f)
    for ms in [80, 160, 320, 400, 560, 1000]:
        print(" {0:5d} |".format(ms), end="", file=f)
    print('', file=f)

    avg_error = 0.0

    srnn_gts_euler = get_srnn_gts(actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len )

    with torch.no_grad():
        for action in actions:
            # get axis angle prefix and gt_out
            val_prefix, val_gt_out = get_batch_srnn(test_set, pose_dim, prefix_len, pred_len, action)

            val_prefix = torch.FloatTensor(val_prefix).to(device).permute(1, 0, 2) # L B D

            # get prediction result
            if multi:
                num_samples = 50
                # extract metrics when doing stochastic prediction.
                val_prefix = val_prefix.repeat(1, num_samples, 1)
                pred_result = model.sample(val_prefix, pred_len, pose_dim, use_zero=False)
            else:
                # use_zero=True makes the model not use stochastic process. it is only for deterministic prediction
                pred_result = model.sample(val_prefix, pred_len, pose_dim, use_zero=True)

            # denormalize the output
            srnn_pred_expmap = unNormalizeDataChunk(pred_result.permute(1, 0, 2).cpu(), data_mean, data_std, dim_to_ignore, actions)
            if multi:
                srnn_pred_expmap = srnn_pred_expmap.reshape(-1, num_samples, srnn_pred_expmap.shape[-2], srnn_pred_expmap.shape[-1])

            # save the errors
            mean_errors = np.zeros((srnn_pred_expmap.shape[0], srnn_pred_expmap.shape[2 if multi else 1])) #  B L

            N_SEQUENCE_TEST = 8
            for n in range(N_SEQUENCE_TEST):
                eulerchannels_pred = srnn_pred_expmap[n]

                if multi:
                    for ii in range(50):
                        for j in range(eulerchannels_pred.shape[1]):
                            for k in range(3, 97, 3):
                                eulerchannels_pred[ii, j, k:k+3] = rotmat2euler( expmap2rotmat( eulerchannels_pred[ii, j, k:k+3] ))

                else:
                    for j in range(eulerchannels_pred.shape[0]):
                        for k in range(3, 97, 3):
                            eulerchannels_pred[j, k:k+3] = rotmat2euler( expmap2rotmat( eulerchannels_pred[j, k:k+3] ))

                gt_i = np.copy(srnn_gts_euler[action][n])
                gt_i[:, 0:6] = 0

                # Now compute the l2 error. The following is numpy port of the error
                # function provided by Ashesh Jain (in matlab), available at
                # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54         
                idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4)[0]

                if multi:
                    euc_error = np.power( gt_i[:, idx_to_use][None, :, :] - eulerchannels_pred[:, :, idx_to_use], 2)
                    euc_error = np.sqrt(np.sum(euc_error, -1))
                    euc_error = np.min(euc_error, 0)
                    
                else:
                    euc_error = np.power( gt_i[:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
                    euc_error = np.sqrt( np.sum(euc_error, 1) )
                mean_errors[n, :] = euc_error

            mean_mean_errors = np.mean( mean_errors, 0 )
            avg_error += mean_mean_errors / len(actions)

            # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
            print("{0: <16} |".format(action), end="", file=f)
            for ms in [1,3,7,9,13,24]:
                print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="", file=f)
            print('', file=f)
    print("{0: <16} |".format('average'), end="", file=f)
    for ms in [1,3,7,9,13,24]:
        print(" {0:.3f} |".format( avg_error[ms] ), end="", file=f)
    print('', file=f)
    print('', file=f)
    f.flush()
    