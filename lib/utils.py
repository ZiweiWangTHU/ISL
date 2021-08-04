import random
import numpy as np

import torch


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


def cal_aff_num(aff_mat):
    """Calculate accuracy ratio for found neighbours."""
    aff_num = 0
    for i in range(len(aff_mat)):
        if len(aff_mat[i]) <= 1:
            continue
        aff = list(aff_mat[i])
        aff.remove(i)
        aff_num += len(aff)
    return aff_num


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_grad_norm(model):
    """Show the max gradient in a step of all the model's parameters."""
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            module_norm = p.grad.norm()
            total_norm += module_norm ** 2
    total_norm = torch.sqrt(total_norm).item()
    return total_norm


def set_train(model):
    for p in model.parameters():
        p.requires_grad = True
    model.train()


def set_eval(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def stop_train_GAN(losses, is_round0=False):
    if is_round0 and len(losses) < 10:  # round0 should train longer time
        return False
    min_loss = np.array(losses).min()
    if min_loss > 1. or (is_round0 and losses[-1] > 1.):  # experiments show that G will converge to loss < 1.
        return False
    if np.min(losses[-3:]) == losses[-1]:  # loss still decreasing
        return False
    if min_loss not in losses[-3:]:
        return True
    return False


def stop_mining(aff_mat, cur_round, ndata, per):
    aff_num = cal_aff_num(aff_mat)
    if (float(aff_num) / float(ndata)) > (cur_round + 1) * per:
        return True
    return False
