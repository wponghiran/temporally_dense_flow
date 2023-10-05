
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from models.eraft import ERAFT
from augmented_voxel_datasets import AugmentedVoxelDSECDataset
from multiscaleloss import flow_error_dsec_supervised
from util import StatTracker
import os
import sys

import cv2
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import ast
from copy import deepcopy


def train(train_loader, model, optim, epoch, log_file, grad_scalar):
    loss_stat = StatTracker() 

    pbar = tqdm(train_loader)
    for batch_idx, data in enumerate(pbar, 0):
        # event_repr has a dimension (batch_size, num_bins, num_polarities, height, width)
        # flow has a dimension (batch_size, direction, height, width)
        # flow_mask has a dimension (batch_size, height, width)
        prev_event_reprs, cur_event_reprs, gt_flows, gt_flow_masks = data
        gt_flows = gt_flows.cuda()

        # Switch model to training modargs
        model.train()
        # Process output based on the network type
        outp_len = 12
        _, outps = model(prev_event_reprs, cur_event_reprs)
        assert len(outps) == outp_len
        pred_flows = outps[outp_len - 1]

        gt_flow_masks = gt_flow_masks.unsqueeze(dim=1).expand(gt_flows.shape).cuda()
        all_pixel_errors = (gt_flows - pred_flows)**2
        valid_pixel_errors = all_pixel_errors[gt_flow_masks]
        avg_loss = torch.mean(valid_pixel_errors)

        # compute gradient and do optimization step
        optim.zero_grad()
        # avg_loss.backward()
        # optim.step()
        grad_scalar.scale(avg_loss).backward()
        grad_scalar.step(optim)
        grad_scalar.update()

        loss_stat.update(valid_pixel_errors.detach())

        if batch_idx % 20 == 0:
            pbar.set_description("Training - Epoch: {} Loss: {}".format(epoch+1, loss_stat))
    log_file.write('Epoch: {} Loss: {}\n'.format(epoch, loss_stat))

    return str(loss_stat)


def validate(test_loader, model, visualize, save_visualization_dir):

    # Switch model to the testing mode
    model.eval()

    error_stat = StatTracker()
    masked_error_stat = StatTracker() 
    gt_flow_stat = StatTracker()

    total_errors = 0
    total_pe1 = 0
    total_pe2 = 0
    total_pe3 = 0
    total_pe4 = 0
    total_pe5 = 0

    outp_len = 12

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, data in enumerate(pbar, 0):
            # event_repr has a dimension (batch_size, num_bins, num_polarities, height, width)
            # flow has a dimension (batch_size, direction, height, width)
            prev_event_reprs, cur_event_reprs, ori_event_reprs, gt_flows, gt_flow_masks = data

            # Conditional check for running model
            _, outps = model(prev_event_reprs, cur_event_reprs)

            # Add evaluation results to the stat tracking
            pred_flows = outps[outp_len - 1]
            # print('w_reset', pred_flows.shape, gt_flows.shape, gt_flow_masks.shape)
            valid_pixel_errors, n_errors, \
            n_pe1, n_pe2, n_pe3, n_pe4, n_pe5 = \
                flow_error_dsec_supervised(gt_flows, gt_flow_masks, pred_flows, ori_event_reprs,
                                           error_stat, masked_error_stat, gt_flow_stat)
            total_errors += n_errors
            total_pe1 += n_pe1
            total_pe2 += n_pe2
            total_pe3 += n_pe3
            total_pe4 += n_pe4
            total_pe5 += n_pe5

    print(error_stat)
    print(masked_error_stat)
    print(gt_flow_stat)
    print('---- NP ----- ')
    print(f'{total_errors:.3e}, '
          f'{total_pe1 / total_errors:.3e}, '
          f'{total_pe2 / total_errors:.3e}, '
          f'{total_pe3 / total_errors:.3e}, '
          f'{total_pe4 / total_errors:.3e}, '
          f'{total_pe5 / total_errors:.3e}')


if __name__ == '__main__':
    # Parser for setting hyperparameters during training and testing 
    parser = argparse.ArgumentParser(description='Optical flow prediction training script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Name of the result directory')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--n-epochs', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--dataset-dir', default=None, help='Path to pre-processed dataset')

    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'], help='Solver algorithms')
    parser.add_argument('--save-interval', default=1, type=int, help='Save interval')
    parser.add_argument('--mode', default='test_w_reset', type=str, help='Evaluate model on validation set')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite exisiting file?')
    parser.add_argument('--model-path', default=None, help='Path to saved model')

    parser.add_argument('--use-scheduler', action='store_true', help='Use learning rate scheduler?')
    parser.add_argument('--milestones', default=[5, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 170],
                        nargs='*', help='epochs at which learning rate is divided by 2')

    parser.add_argument('--dt', type=int, default=1, help='Frame difference for computing a photometric loss')
    parser.add_argument('--n-split', type=int, default=10, help='Number of bins for events representation')
    parser.add_argument('--model-options', default='', type=str,
                        help='Number of epoch to be trained with an incremental sequence length')

    parser.add_argument('--visualize', action='store_true', help='Visualize grouth truth flow and prediction')

    args = parser.parse_args()
    assert args.mode in ['train', 'test_w_reset'], '{} is invalid mode'.format(args.mode)

    # Dataset and groud truth paths 
    dataset_dir = os.path.abspath(args.dataset_dir)

    save_dir = os.path.join(args.save_dir, 'dt{},tsplit{},ERAFT,{},e{},bs{},lr{:.0e}{}'.format(
        args.dt, args.n_split, args.solver,
        args.n_epochs, args.bs,
        args.lr,
        '' if args.model_options == '' else ','+args.model_options.replace('"', '').replace('\'', '').replace(':', '-')))

    if args.mode == 'train':
        # There is no need to convert sample in tensor format to PIL image for transformation
        # Be careful with ToPILImage() if need to be used since it normalizes inputs by default
        #   Tensor with value above 1 is cliped to 1
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # Default random flip function cannot be used for DSEC dataset
            # since optical flows have both quantity and direction unlike image intensities
            # We create an option to randomly flip the sample as a part of dataset
            transforms.RandomCrop((288, 384))
        ])
        train_dataset = AugmentedVoxelDSECDataset(dataset_dir, dt=args.dt, n_split=args.n_split,
                                                        transform=train_transform, random_flip=True, mode='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                  pin_memory=False)
    elif 'test' in args.mode:
        test_transform = transforms.Compose([
            transforms.CenterCrop((288, 384))
        ])
        test_dataset = AugmentedVoxelDSECDataset(dataset_dir, dt=args.dt, n_split=args.n_split,
                                                       transform=test_transform, random_flip=False, mode=args.mode)
        print('event_repr.shape:', test_dataset[0][0].shape)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs,
                                 shuffle=False, num_workers=args.workers,
                                 pin_memory=False)

    model_options = ast.literal_eval('{'+args.model_options+'}')
    for key, default_val in {'subtype': 'standard', 'n_first_channels': 10}.items():
        if key not in model_options:
            model_options[key] = default_val

    model = ERAFT(**model_options)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)['model'])
        print("=> Restore the pre-trained model 'ERAFT' from {}".format(args.model_path))

    model = torch.nn.DataParallel(model).cuda()

    if 'test' in args.mode:
        validate(test_loader=test_loader, model=model, visualize=args.visualize,
                 save_visualization_dir=save_dir + '_visualization')

    elif args.mode == 'train':
        if os.path.isfile(save_dir+'.log') and not args.overwrite:
            print('File {} exists!'.format(save_dir+'.log'))
            sys.exit(0)
        log_file = open(save_dir+'.log', 'w', buffering=1)

        # Record input argument
        print('=> Input command ' + ' '.join(sys.argv))
        log_file.write(' '.join(sys.argv) + '\n')

        # Make a directory to save files if it doesn't exist 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('=> Results will be saved to {}'.format(save_dir))

        assert (args.solver in ['adam', 'sgd'])
        print('=> Setting {} solver'.format(args.solver))
        optim = None
        if args.solver == 'adam':
            optim = torch.optim.Adam(model.parameters(), args.lr)
        elif args.solver == 'sgd':
            optim = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)

        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=args.milestones, gamma=0.1)
            print("=> Using learning rate scheduler")

        grad_scalar = torch.cuda.amp.GradScaler()

        for epoch in range(args.n_epochs):
            train_loss = train(train_loader=train_loader, model=model, optim=optim, epoch=epoch,
                               log_file=log_file, grad_scalar=grad_scalar)

            if args.use_scheduler:
                scheduler.step()

            # Save the model for testing every N epochs
            if (epoch + 1) % args.save_interval == 0:
                torch.save({'epoch': epoch+1, 'arch': 'ERAFT', 'optim': deepcopy(optim.state_dict()),
                            'model': deepcopy(model.module.state_dict())},
                           os.path.join(save_dir, 'checkpoint_ep{}.pt'.format(epoch+1)))

        log_file.close()
