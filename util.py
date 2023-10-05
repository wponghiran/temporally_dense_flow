import os
import numpy as np
import shutil
import torch


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, batch_mean, batch_size=1):
        self.sum += batch_mean * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f}'.format(self.avg)


class StatTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.total_sq = 0
        self.count = 0

    def update(self, val):
        self.total = self.total + torch.sum(val)
        self.total_sq = self.total_sq + torch.sum(val ** 2)
        self.count = self.count + torch.numel(val)

    def __repr__(self):
        if self.count != 0:
            avg = self.total / self.count
            std = (self.total_sq / self.count - avg ** 2) ** 0.5
            return '{:.3e},{:.3e}'.format(avg, std)
        else:
            return ''


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

