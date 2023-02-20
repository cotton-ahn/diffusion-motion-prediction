import torch

from src.dataset.euler import get_batch

def train_euler(f, epoch, epoch_iters, model, optimizer, 
                train_set, batch_size, pose_dim, 
                prefix_len, pred_len, actions, device):
    model.train()
    
    epoch_loss = 0.0    
    for it in range(epoch_iters):
        prefix, gt_out = get_batch(train_set, batch_size, pose_dim, prefix_len, pred_len, actions)
        prefix = torch.FloatTensor(prefix).to(device).permute(1, 0, 2)
        gt_out = torch.FloatTensor(gt_out).to(device).permute(1, 0, 2)

        loss = model.calc_loss(prefix, gt_out)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item() / epoch_iters

    print('[EPOCH {}] train average loss: {}'.format(epoch, epoch_loss), file=f)

    f.flush()


def train_supervised(f, epoch, epoch_iters, model, optimizer, 
                     train_set, batch_size, pose_dim, 
                     prefix_len, pred_len, actions, device):
    model.train()
    
    epoch_loss = 0.0    
    for it in range(epoch_iters):
        prefix, gt_out = get_batch(train_set, batch_size, pose_dim, prefix_len, pred_len, actions)
        prefix = torch.FloatTensor(prefix).to(device).permute(1, 0, 2)
        last_pose = prefix[-1][None, :].repeat(pred_len, 1, 1)
        window = torch.cat([prefix, last_pose], dim=0)

        gt_out = torch.FloatTensor(gt_out).to(device).permute(1, 0, 2)

        loss = torch.mean((model(window) - gt_out)**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item() / epoch_iters

    print('[EPOCH {}] train average loss: {}'.format(epoch, epoch_loss), file=f)

    f.flush()
