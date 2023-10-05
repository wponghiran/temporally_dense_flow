
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from tqdm import tqdm

import os
import random
import queue
import math
import cv2


def flow_viz_np(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    gt_flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return gt_flow_rgb


"""Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow. x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement."""
def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False & bool(scale_factor)
    y_mask[flow_y_interp == 0] = False & bool(scale_factor)

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
    return


# Custom dataset class for training DSEC dataset in a supervised manner with continuous inference option
class AugmentedVoxelDSECDataset(Dataset):
    def __init__(self, dataset_dir, dt, n_split, transform, random_flip=False,
                 mode='train'):

        # Transformation used for data augmentation purpose
        self.transform = transform
        # Random flip for optical flow is not simply swapping pixels values like random flip for images,
        # so we take a seperate flag to
        self.random_flip = random_flip

        assert mode in ['train', 'test_w_reset'], '{} is invalid mode for dataset'.format(mode)
        self.mode = mode
        self.dt = dt  # Index difference between grayscale images
        # dt=1 meaning that events between 2 consecutive grayscale frames are used to construct an event representation
        # dt=4 meaning that events between grayscale image k and grayscale iamge k+4
        #   are used to construct an event representation
        # dt=1 is known to give the best training result as phometric loss is well estimated in a short interval
        #   while dt=4 is widely used to minimize a training time
        self.n_split = n_split  # Number of splits between events happened between two grayscale images
        # Since we construct an event represenation by counting number of postive and negative events
        #   and puting the value into 2 seperate channels,
        # n_split=1 is equivalent to summing all positive and negaitve events between grayscale frames.
        # n_split=2 is equivalent to dividing total events into 2 bins
        #   where each bin has approximately the same number of events.
        # Positive and negative events in each bin are then translated to values of 2 channels.
        # For this reason, this custom dataset object will output an event representation
        #   as a tensor of shape (n_split, n_polarity=2, width, height)
        self.dataset_dir = os.path.join(dataset_dir, f'dt{self.dt}_tsplit{self.n_split}')
        self.n_prefix_event_repr = 1 + 1 # Get additional prefix for computing intermediate target
            # Also get additional prefix for computing ERAFT model
        # This parameter is only used for training sequential optical flow prediction

        # Environment pool includes 18 sequences in DSEC dataset that have optical flows recorded
        # Since test sequences for the DSEC dataset are not publicly released,
        #   we split part of the training sequence for evaluation
        # Note that each sequence below has been pre-processed to reduce processing time during training and testing
        train_ratio = 0.8
        env_name_pool = ['thun_00_a', 'zurich_city_01_a', 'zurich_city_02_a', 'zurich_city_02_c', 'zurich_city_02_d',
                         'zurich_city_02_e', 'zurich_city_03_a', 'zurich_city_05_a', 'zurich_city_05_b',
                         'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a', 'zurich_city_09_a',
                         'zurich_city_10_a', 'zurich_city_10_b', 'zurich_city_11_a', 'zurich_city_11_b',
                         'zurich_city_11_c']
        # env_name_pool = ['zurich_city_11_b', 'zurich_city_02_c']

        # Creating a list of datapath to each sample from each environment mapping file
        # The mapping file is generated during dataset pre-processing step
        self.seqs_of_sample_paths = []
        mapping_file_paths = [path for path in os.listdir(self.dataset_dir) if path.endswith('_mapping.txt')]
        for file_idx, mapping_file_path in enumerate(mapping_file_paths):
            # Since there can be more than one segment of optical flow recording in each sequence,
            # We get both the environment name and segment/chunk index.
            env_name = mapping_file_path[:-19]
            chunk_idx = int(mapping_file_path[-13:-12])

            # Process only with environments declared in the pool
            if not env_name in env_name_pool:
                continue

            num_flow = sum(1 for line in open(os.path.join(self.dataset_dir, mapping_file_path), 'r')) - 1
            # There are (num_flow - n_prefix) samples from each sequence
            # since the first (n_prefix) events in the sequence don't have sufficient prior events to form a sample
            # We first compute location to split training and testing set
            split_idx = int((num_flow - self.n_prefix_event_repr) * train_ratio)

            seqs_of_sample_paths = []
            queue_of_sample_paths = queue.Queue()  # Utilize queue to save sequence of paths
            if mode == 'train':
                start_idx = 0
                end_idx = split_idx + self.n_prefix_event_repr
            elif mode == 'test_w_reset':
                start_idx = split_idx
                end_idx = num_flow

            for event_repr_idx in range(start_idx, end_idx):
                # Add new path to queue
                queue_of_sample_paths.put(os.path.join(self.dataset_dir,
                                                       '{}_chunk{:01d}_{:06d}.pt'.format(env_name,
                                                                                         chunk_idx, event_repr_idx)))

                # Skip if queue doesn't have enough events
                if queue_of_sample_paths.qsize() != (self.n_prefix_event_repr + 1):
                    continue
                # Otherwise, extend sequence of paths to a dataset
                else:
                    seqs_of_sample_paths.append(list(queue_of_sample_paths.queue))
                    # Maintain queue size at desired size - 1 for adding the next path
                    queue_of_sample_paths.get()
            # Save sequences of paths
            self.seqs_of_sample_paths.extend(seqs_of_sample_paths)

        self.len = len(self.n_split*self.seqs_of_sample_paths)
        print('{}ing set has {} samples'.format(mode, self.len))

    # Number of prefix may be changed during training in case of incremental length training
    # Given that initial value of n_prefix is always set to be the maximum value possible during training
    def change_n_prefix_event_repr(self, n_prefix_event_repr):
        self.n_prefix_event_repr = n_prefix_event_repr

    def __getitem__(self, idx):
        # For training, we can access any sequence in random because we are allowed some reset to happen
        # and expect that training in such a way help inference without reset
        # For testing, batch size is limited to 1 since we aim to perform inference without reset

        # For training and testing with reset, load individual sample in the same way based on a given seq_idx and offset_idx

        # Load from -(n_prefix+1) event reprs instead of all event reprs
        #   since n_prefix may be changed during training in case of incremental length training
        # Initial value of n_prefix is always set to be the maximum value possible during the training
        seq_idx = idx // self.n_split
        offset_idx = (idx % self.n_split) + 1
        ratio_latter_gt = offset_idx / self.n_split
        seq_of_sample_paths = self.seqs_of_sample_paths[seq_idx][-(self.n_prefix_event_repr+1):]
        # event_repr has a dimension (num_bins*(1+n_prefix_event_repr), num_polarities, height, width)
        event_repr = torch.cat([torch.load(sample_path)['event_repr'] for sample_path in seq_of_sample_paths])
        # event_repr = torch.cat(
        #     [torch.from_numpy(np.load(sample_path.replace(f'dt{self.dt}_tsplit{self.n_split}', f'np_voxel_dt{self.dt}_tsplit{self.n_split}').replace('.pt', '.npz'))['voxel']) for sample_path in
        #      seq_of_sample_paths])

        # Shorten length of event repr since we load more than we need corresponding to augmented flow
        if self.mode == 'test_w_reset':
            ori_event_repr = torch.clone(event_repr)
        event_repr = event_repr[:, 0] - event_repr[:, 1]
      
        event_repr = event_repr[offset_idx:(offset_idx + (self.n_split * self.n_prefix_event_repr))]

        # Normalize voxel representation
        mask = torch.nonzero(event_repr, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = event_repr[mask].mean()
            std = event_repr[mask].std()
            if std > 0:
                event_repr[mask] = (event_repr[mask] - mean) / std
            else:
                event_repr[mask] = event_repr[mask] - mean

        data = torch.load(seq_of_sample_paths[-1])
        flow = data['flow'] # flow has a dimension (direction, height, width)
        flow_mask = data['flow_mask']   # flow_mask has a dimension (height, width)
        # data = np.load(seq_of_sample_paths[-1])
        # flow = torch.from_numpy(data['flow']) # flow has a dimension (direction, height, width)
        # flow_mask = torch.from_numpy(data['flow_mask'])  # flow_mask has a dimension (height, width)

        flow = flow.masked_fill((~flow_mask).unsqueeze(0).repeat(2, 1, 1), 0)
        x_flow = np.squeeze(flow[0].numpy())
        y_flow = np.squeeze(flow[1].numpy())

        # Load previous flow and flow_mask in case of training to compute augmented flow
        prev_data = torch.load(seq_of_sample_paths[-2])
        prev_flow = prev_data['flow']
        prev_flow_mask = prev_data['flow_mask']
        # prev_data = np.load(seq_of_sample_paths[-2])
        # prev_flow = torch.from_numpy(prev_data['flow'])
        # prev_flow_mask = torch.from_numpy(prev_data['flow_mask'])

        prev_flow = prev_flow.masked_fill((~prev_flow_mask.bool()).unsqueeze(0).repeat(2, 1, 1), 0)
        prev_x_flow = np.squeeze(prev_flow[0].numpy())
        prev_y_flow = np.squeeze(prev_flow[1].numpy())

        # Compute flow and flow_mask here
        # The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we need to propagate the ground truth flow over the time between two images.
        # This function assumes that the ground truth flow is in terms of pixel displacement, not velocity. Pseudo code for this process is as follows:
        # x_orig = range(cols)      y_orig = range(rows)
        # x_prop = x_orig           y_prop = y_orig
        # Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
        # for all of these flows:
        #   x_prop = x_prop + gt_flow_x(x_prop, y_prop)
        #   y_prop = y_prop + gt_flow_y(x_prop, y_prop)
        # The final flow, then, is x_prop - x-orig, y_prop - y_orig.
        # Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.
        # Inputs:
        #   x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at each timestamp.
        #   gt_timestamps - timestamp for each flow array.  start_time, end_time - gt flow will be estimated between start_time and end time.
        x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))
        x_indices = x_indices.astype(np.float32)
        y_indices = y_indices.astype(np.float32)

        orig_x_indices = np.copy(x_indices)
        orig_y_indices = np.copy(y_indices)

        # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
        x_mask = np.ones(x_indices.shape, dtype=bool)
        y_mask = np.ones(y_indices.shape, dtype=bool)

        prop_flow(prev_x_flow, prev_y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=(1 - ratio_latter_gt))

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=ratio_latter_gt)

        x_shift = x_indices - orig_x_indices
        y_shift = y_indices - orig_y_indices
        x_shift[~x_mask] = 0
        y_shift[~y_mask] = 0

        int_flow = torch.from_numpy(np.stack((x_shift, y_shift), axis=0)).float()
        int_flow_mask = (int_flow[0].bool() | int_flow[1].bool()).float()

        if self.mode == 'test_w_reset':
            transformed_ori_event_repr = torch.zeros(list(ori_event_repr.shape[0:2]) + [288, 384])
        transformed_event_repr = torch.zeros([event_repr.shape[0], 288, 384])
        transformed_flow = torch.zeros([int_flow.shape[0], 288, 384])

        # For randomized transformation, setting random seed guarantee the uniform operation
        #   between events and grayscale images
        # For static transformation, setting random seed has no effect
        seed = np.random.randint(2147483647)

        # Random or central crop for event_repr
        if self.mode == 'test_w_reset':
            for bin_idx in range(ori_event_repr.shape[0]):
                for pol_idx in range(ori_event_repr.shape[1]):
                    random.seed(seed)
                    torch.manual_seed(seed)
                    transformed_ori_event_repr[bin_idx, pol_idx] = self.transform(ori_event_repr[bin_idx, pol_idx])
        for bin_idx in range(event_repr.shape[0]):
            random.seed(seed)
            torch.manual_seed(seed)
            transformed_event_repr[bin_idx] = self.transform(event_repr[bin_idx])

        # Random or central crop for flow
        for direction_idx in range(transformed_flow.shape[0]):
            random.seed(seed)
            torch.manual_seed(seed)
            transformed_flow[direction_idx] = self.transform(int_flow[direction_idx])

        # Random or central crop for flow mask
        random.seed(seed)
        torch.manual_seed(seed)
        transformed_flow_mask = self.transform(int_flow_mask)

        # Horizontal flipping for event_repr, flow, flow_mask
        random.seed(seed)
        torch.manual_seed(seed)
        if self.random_flip and torch.rand(1) < 0.5:
            if self.mode == 'test_w_reset':
                transformed_ori_event_repr = transformed_ori_event_repr.flip(-1)
            transformed_event_repr = transformed_event_repr.flip(-1)
            transformed_flow = transformed_flow.flip(-1)
            transformed_flow[0] = -transformed_flow[0]
            transformed_flow_mask = transformed_flow_mask.flip(-1)

        # Vertical flipping for event_repr, flow, flow_mask
        random.seed(seed)
        torch.manual_seed(seed)
        if self.random_flip and torch.rand(1) < 0.5:
            if self.mode == 'test_w_reset':
                transformed_ori_event_repr = transformed_ori_event_repr.flip(-2)
            transformed_event_repr = transformed_event_repr.flip(-2)
            transformed_flow = transformed_flow.flip(-2)
            transformed_flow[1] = -transformed_flow[1]
            transformed_flow_mask = transformed_flow_mask.flip(-2)

        # Assign appropriate type to mask
        transformed_flow_mask = transformed_flow_mask.bool()

        if self.mode == 'test_w_reset':
            return transformed_event_repr[:self.n_split], transformed_event_repr[self.n_split:], transformed_ori_event_repr, transformed_flow, transformed_flow_mask
        else:
            return transformed_event_repr[:self.n_split], transformed_event_repr[self.n_split:], transformed_flow, transformed_flow_mask

    def __len__(self):
        return self.len


if __name__ == '__main__':
    from datasets import DSECDatasetSupervisedContinuous

    # Test DSEC for supervised training with continous inference option
    print('--- Test test_w_reset ---')
    test_2_w_reset_dataset = AugmentedVoxelDSECDataset(
        dataset_dir='/local/a/datasets/dsec-preprocessed/', dt=1, n_split=10,
        transform=transforms.Compose([transforms.CenterCrop((288, 384))]),
        random_flip=False, mode='test_w_reset')
    test_2_w_reset_loader = DataLoader(dataset=test_2_w_reset_dataset, batch_size=16, shuffle=False, num_workers=8)
    for idx, (prev_event_repr, cur_event_repr, ori_event_repr, flow, flow_mask) in tqdm(enumerate(test_2_w_reset_loader), total=len(test_2_w_reset_loader)):
        pass
#         if idx == 1000:
#             break
        # np_flow = flow.numpy()
        # # masked_gt_flow_rgb = flow_viz_np(np_flow[0, 0, :, :], np_flow[0, 1, :, :])
        # np_flow_mask = flow_mask.float().numpy()
        # masked_gt_flow_rgb = flow_viz_np(np_flow[0, 0, :, :] * np_flow_mask[0],
        #                                  np_flow[0, 1, :, :] * np_flow_mask[0])
        # cv2.imshow('Visualize test_w_reset', cv2.cvtColor(masked_gt_flow_rgb, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(1)  # wait time in millisecond unit

    print('--- Test train ---')
    # train_dataset_1 = DSECDatasetSupervisedContinuous(
    #     dataset_dir='/local/a/datasets/dsec-preprocessed/', dt=1, n_split=10,
    #     transform=transforms.Compose([transforms.CenterCrop((288, 384))]),
    #     random_flip=False, mode='train', n_prefix_event_repr=n_prefix_event_repr+1)
    # train_loader_1 = DataLoader(dataset=train_dataset_1, batch_size=1, shuffle=False, num_workers=0)
    # it_1 = iter(train_loader_1)
    train_dataset_2 = AugmentedVoxelDSECDataset(
        dataset_dir='/local/a/datasets/dsec-preprocessed/', dt=1, n_split=10,
        transform=transforms.Compose([transforms.CenterCrop((288, 384))]),
        random_flip=False, mode='train')
    train_loader_2 = DataLoader(dataset=train_dataset_2, batch_size=16, shuffle=False, num_workers=8)

    for idx, (prev_event_repr, event_repr_curr, flow, flow_mask) in tqdm(enumerate(train_loader_2), total=len(train_loader_2)):
        pass
#         if idx == 10000:
#             break
        # np_flow = flow.numpy()
        # np_flow_mask = flow_mask.float().numpy()
        # masked_gt_flow_rgb = flow_viz_np(np_flow[0, 0, :, :] * np_flow_mask[0], np_flow[0, 1, :, :] * np_flow_mask[0])
        # # masked_gt_flow_rgb = flow_viz_np(np_flow[0, 0, :, :], np_flow[0, 1, :, :])
        # cv2.imshow('Visualize train', cv2.cvtColor(masked_gt_flow_rgb, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(1)  # wait time in millisecond unit
        # # video_file.write(masked_gt_flow_rgb)
        # # cv2.imwrite(path, cv2.cvtColor(masked_gt_flow_rgb, cv2.COLOR_BGR2RGB))
    # video_file.release()
