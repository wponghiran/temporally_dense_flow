import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn


"""
Calculates per pixel flow error between pred_flow and gt_flow. event_repr is used to mask out any pixels without events
"""
# def flow_error_dsec_supervised(gt_flows, gt_flow_masks, pred_flows, event_reprs):
def flow_error_dsec_supervised(gt_flows, gt_flow_masks, pred_flows, event_reprs, error_stat, masked_error_stat, gt_flow_stat):
    # event_rerp has a dimension of (batch_size, num_bins, num_polarities, height, width)
    # pred_flow, gt_flow has a dimension of (batch_size, direction, height, width)
    # gt_flow_masks has a dimension of (batch_size, height, width)
    pred_flows = pred_flows.cpu()
    nonzero_event_masks = (torch.sum(torch.sum(event_reprs, dim=2), dim=1) > 0)

    # Compute discrepancy in each direction and squared norm to compute epe according to https://arxiv.org/pdf/1612.02590.pdf
    all_pixel_errors = torch.linalg.norm(gt_flows - pred_flows, dim=1)
    valid_pixel_errors = all_pixel_errors[gt_flow_masks]
    masked_valid_pixel_errors = all_pixel_errors[gt_flow_masks & nonzero_event_masks]
    valid_pixel_gt_flows = torch.linalg.norm(gt_flows, dim=1)[gt_flow_masks]

    error_stat.update(valid_pixel_errors)
    masked_error_stat.update(masked_valid_pixel_errors)
    gt_flow_stat.update(valid_pixel_gt_flows)

    n_errors = torch.numel(masked_valid_pixel_errors)
    n_pe1 = torch.numel(masked_valid_pixel_errors[masked_valid_pixel_errors > 1])
    n_pe2 = torch.numel(masked_valid_pixel_errors[masked_valid_pixel_errors > 2])
    n_pe3 = torch.numel(masked_valid_pixel_errors[masked_valid_pixel_errors > 3])
    n_pe4 = torch.numel(masked_valid_pixel_errors[masked_valid_pixel_errors > 4])
    n_pe5 = torch.numel(masked_valid_pixel_errors[masked_valid_pixel_errors > 5])

    return torch.mean(valid_pixel_errors), n_errors, n_pe1, n_pe2, n_pe3, n_pe4, n_pe5

