
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

import math
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )


def conv_s(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        )


def predict_flow(batchNorm, in_planes):
    if batchNorm:
        return nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Conv2d(in_planes,2,kernel_size=1,stride=1,padding=0,bias=False),
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )


def deconv(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))


def crop_like(inp, target):
    if inp.size()[2:] == target.size()[2:]:
        return inp
    else:
        return inp[:, :, :target.size(2), :target.size(3)]


class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, inp):
        self.save_for_backward(inp)
        return inp.gt(1e-5).float()

    @staticmethod
    def backward(self, grad_output):
        inp, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inp <= 1e-5] = 0
        return grad_input


def IF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    out = out.detach() + (1/threshold)*out - (1/threshold)*out.detach()

    return membrane_potential, out


class SpikeFlowNet(BaseModel):
    def __init__(self, model_options):
        super().__init__()
        self.num_pols = model_options['num_pols']
        self.batchNorm = model_options['batch_norm']
        self.threshold = model_options['threshold']
        self.n_split = model_options['n_split']

        self.conv1 = conv_s(self.batchNorm, self.num_pols*2, 64, kernel_size=3, stride=2)
        self.conv2 = conv_s(self.batchNorm,  64,  128, kernel_size=3, stride=2)
        self.conv3 = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=2)
        self.conv4 = conv_s(self.batchNorm, 256,  512, kernel_size=3, stride=2)

        self.conv_r11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        self.deconv3 = deconv(self.batchNorm, 512, 128)
        self.deconv2 = deconv(self.batchNorm, 384+2, 64)
        self.deconv1 = deconv(self.batchNorm, 192+2, 4)

        self.predict_flow4 = predict_flow(self.batchNorm, 32)
        self.predict_flow3 = predict_flow(self.batchNorm, 32)
        self.predict_flow2 = predict_flow(self.batchNorm, 32)
        self.predict_flow1 = predict_flow(self.batchNorm, 32)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=192+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=68+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, inp):

        batch_size, num_ts, num_c, inp_h, inp_w = inp.shape
        device = inp.device

        mem_1 = torch.zeros(batch_size, 64,  int(inp_h/2),  int(inp_w/2) ).to(device)
        mem_2 = torch.zeros(batch_size, 128, int(inp_h/4),  int(inp_w/4) ).to(device)
        mem_3 = torch.zeros(batch_size, 256, int(inp_h/8),  int(inp_w/8) ).to(device)
        mem_4 = torch.zeros(batch_size, 512, int(inp_h/16), int(inp_w/16)).to(device)

        mem_1_total = torch.zeros(batch_size, 64,  int(inp_h/2),  int(inp_w/2) ).to(device)
        mem_2_total = torch.zeros(batch_size, 128, int(inp_h/4),  int(inp_w/4) ).to(device)
        mem_3_total = torch.zeros(batch_size, 256, int(inp_h/8),  int(inp_w/8) ).to(device)
        mem_4_total = torch.zeros(batch_size, 512, int(inp_h/16), int(inp_w/16)).to(device)

        for t in range(num_ts//2):
            current_1 = self.conv1(torch.cat((inp[:, t], inp[:, self.n_split//2+t]), dim=1))
            mem_1 = mem_1 + current_1
            mem_1_total = mem_1_total + current_1
            mem_1, out_conv1 = IF_Neuron(mem_1, self.threshold)

            current_2 = self.conv2(out_conv1)
            mem_2 = mem_2 + current_2
            mem_2_total = mem_2_total + current_2
            mem_2, out_conv2 = IF_Neuron(mem_2, self.threshold)

            current_3 = self.conv3(out_conv2)
            mem_3 = mem_3 + current_3
            mem_3_total = mem_3_total + current_3
            mem_3, out_conv3 = IF_Neuron(mem_3, self.threshold)

            current_4 = self.conv4(out_conv3)
            mem_4 = mem_4 + current_4
            mem_4_total = mem_4_total + current_4
            mem_4, out_conv4 = IF_Neuron(mem_4, self.threshold)

        mem_4_residual = 0
        mem_3_residual = 0
        mem_2_residual = 0

        out_conv4 = mem_4_total + mem_4_residual
        out_conv3 = mem_3_total + mem_3_residual
        out_conv2 = mem_2_total + mem_2_residual
        out_conv1 = mem_1_total

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        flow4 = self.predict_flow4(self.upsampled_flow4_to_3(out_rconv22))
        flow4_up = crop_like(flow4, out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3 = self.predict_flow3(self.upsampled_flow3_to_2(concat3))
        flow3_up = crop_like(flow3, out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(self.upsampled_flow2_to_1(concat2))
        flow2_up = crop_like(flow2, out_conv1)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)

        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        flow1 = self.predict_flow1(self.upsampled_flow1_to_0(concat1))

        return flow4, flow3, flow2, flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


if __name__ == '__main__':
    _seed_ = 16146
    import random

    random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)

    print('--- Test Model ---')
    # (batch_size, num_bins, num_polarities, height, width)
    inps = torch.rand([1, 20, 2, 288, 384])  # (B, T, C, H, W)
    
    model_options = {'num_pols': 2, 'batch_norm': False, 'threshold': 0.75, 'n_split': 20}
    model = SpikeFlowNet(model_options)
    print(model)
    outps = model(inps)
    print('len(outps) =', len(outps)) # num_encoders = 4 + 2*num_encoders = 8 + 2*num_decoders = 8

