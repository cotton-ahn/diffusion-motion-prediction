import torch
import time

# code from DLow
def get_tensor_pose(traj_np, device):
    traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj_ts = torch.Tensor(traj_np).permute(1, 0, 2).to(device) # T B D
    
    return traj_ts

def train_xyz(f, epoch, epoch_iters, model, optimizer, dataset, batch_size, prefix_len, device):
    t_s = time.time()

    generator = dataset.sampling_generator(num_samples=epoch_iters*batch_size, batch_size=batch_size)

    cnt = 0
    train_losses = 0

    for traj_np in generator:
        traj_ts = get_tensor_pose(traj_np, device)

        prefix = traj_ts[:prefix_len] # T B D
        gt = traj_ts[prefix_len:] # T B D

        loss = model.calc_loss(prefix, gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        train_losses += loss.item()
        cnt += 1
    dt = time.time() - t_s
    train_losses /= cnt
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(['DIFF'], [train_losses])])
    print('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr), file=f)

    f.flush()
        
    