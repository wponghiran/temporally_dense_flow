
import torch
import torch.nn as nn
import numpy as np

import copy

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


class ATan(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    """
    alpha = 10.0
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = (x > 0).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 1/(1 + (ATan.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None


class LIF(nn.Module):
    def __init__(self,
                 threshold: float = 1.0,
                 leak: float = 1.0,
                 learn_threshold: bool = False,
                 learn_leak: bool = False,
                 activation=None,
                 reset_mechanism: str = "soft",
                 accumulate: bool = False,
                 per_channel: int = 0,
                 rec=False):
        super(LIF, self).__init__()
        self.u = None
        self.spike_k = None
        self.u_reset = None
        self.per_channel = per_channel
        self.learn_threshold = learn_threshold
        self.learn_leak = learn_leak
        if per_channel == 0:
            self.leak = nn.Parameter(torch.tensor(leak), requires_grad=learn_leak)
            self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=learn_threshold)
        else:
            self.leak = nn.Parameter(torch.ones([1, per_channel, 1, 1]) * torch.tensor(leak),
                                     requires_grad=learn_leak)
            self.threshold = nn.Parameter(torch.ones([1, per_channel, 1, 1]) * torch.tensor(threshold),
                                          requires_grad=learn_threshold)
        self.reset_mechanism = reset_mechanism
        self.accumulate = accumulate
        if activation is None:
            self.activation = ATan.apply
        else:
            self.activation = activation.apply

    def reset_state(self):
        self.u = copy.deepcopy(self.u_reset)

    def _init_states(self, x):
        if self.u is None:
            self.u = torch.zeros_like(x).to(x.device)
            self.u_reset = torch.zeros_like(x).to(x.device)
        elif self.u.shape[0] != x.shape[0]:
            self.u = torch.zeros_like(x).to(x.device)
            self.u_reset = torch.zeros_like(x).to(x.device)

    def forward(self, x: torch.tensor):
        self._init_states(x)
        if not self.accumulate:
            self.u = self.leak*self.u + x
            u_thr = self.u - self.threshold.clamp(min=0.01)
            out = self.activation(u_thr)
            rst = out.detach()
            if self.reset_mechanism == "hard":
                self.u = self.u * (1 - rst)
            else:
                self.u = self.u - self.threshold.clamp(min=0.01) * rst  # self.spikes*self.threshold
        else:
            self.u = self.u + x
            out = self.u

        return out

    def extra_repr(self) -> str:
        return 'threshold={}, leak={}, reset={}, accumulate={}, learn_threshold={}, learn_leak={}, per_channel={}'.format(
            self.threshold.max(),
            self.leak.max(), 
            self.reset_mechanism, 
            self.accumulate,
            self.learn_threshold,
            self.learn_leak,
            self.per_channel
        )


def general_conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, batchNorm=False, accumulate=False, model_options=None):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], 
                learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], accumulate=accumulate, per_channel=model_options['per_channel'])
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], 
                learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], accumulate=accumulate, per_channel=model_options['per_channel'])
        )


def pred_flow(in_channels, out_channels, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels)
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1,padding=0,bias=False)
        )


class AdaptiveFlowNet(BaseModel):
    def __init__(self, model_options):
        super().__init__()
        self.model_options = model_options

        self.num_pols = model_options['num_pols']
        self._BASE_CHANNELS = model_options['base_channels']
        self.batchNorm = model_options['batchNorm']

        self.encoder1 = general_conv2d(in_channels=self.num_pols*2, out_channels=self._BASE_CHANNELS, kernel_size=3, stride=2, padding=1, 
                                        batchNorm=self.batchNorm, model_options=model_options) 
        self.encoder2 = general_conv2d(in_channels=self._BASE_CHANNELS, out_channels=2*self._BASE_CHANNELS, kernel_size=3, stride=2, padding=1, 
                                        batchNorm=self.batchNorm, model_options=model_options)
        self.encoder3 = general_conv2d(in_channels=2*self._BASE_CHANNELS, out_channels=4*self._BASE_CHANNELS, kernel_size=3, stride=2, padding=1, 
                                        batchNorm=self.batchNorm, model_options=model_options)
        self.encoder4 = general_conv2d(in_channels=4*self._BASE_CHANNELS, out_channels=8*self._BASE_CHANNELS, kernel_size=3, stride=2, padding=1,  
                                        batchNorm=self.batchNorm, model_options=model_options)

        self.residual11 = general_conv2d(in_channels=8*self._BASE_CHANNELS, out_channels=8*self._BASE_CHANNELS, kernel_size=3, stride=1, padding=1,  
                                        batchNorm=self.batchNorm, model_options=model_options)
        self.residual12 = general_conv2d(in_channels=8*self._BASE_CHANNELS, out_channels=8*self._BASE_CHANNELS, kernel_size=3, stride=1, padding=1,  
                                        batchNorm=self.batchNorm, model_options=model_options)
        self.residual21 = general_conv2d(in_channels=8*self._BASE_CHANNELS, out_channels=8*self._BASE_CHANNELS, kernel_size=3, stride=1, padding=1,  
                                        batchNorm=self.batchNorm, model_options=model_options)                                
        self.residual22 = general_conv2d(in_channels=8*self._BASE_CHANNELS, out_channels=8*self._BASE_CHANNELS, kernel_size=3, stride=1, padding=1,  
                                        batchNorm=self.batchNorm, model_options=model_options)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(8*self._BASE_CHANNELS, 2*self._BASE_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(4*self._BASE_CHANNELS + 2*self._BASE_CHANNELS + 2, self._BASE_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2*self._BASE_CHANNELS + 1*self._BASE_CHANNELS + 2, 4, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )


        self.predict_flow4_acc = LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                                    reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'], accumulate=True)
        self.predict_flow3_acc = LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                                    reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'], accumulate=True)
        self.predict_flow2_acc = LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                                    reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'], accumulate=True)
        self.predict_flow1_acc = LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                                    reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'], accumulate=True)
        self.predict_flow4 = pred_flow(32, 2)
        self.predict_flow3 = pred_flow(32, 2)
        self.predict_flow2 = pred_flow(32, 2)
        self.predict_flow1 = pred_flow(32, 2)

        self.upsampled_flow4_to_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8*self._BASE_CHANNELS, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )
        self.upsampled_flow3_to_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4*self._BASE_CHANNELS + 2*self._BASE_CHANNELS + 2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )
        self.upsampled_flow2_to_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*self._BASE_CHANNELS + 1*self._BASE_CHANNELS + 2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )
        self.upsampled_flow1_to_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1*self._BASE_CHANNELS + 4 + 2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIF(leak=model_options['ileak'], threshold=model_options['ithresh'], learn_leak=model_options['learn_leak'], learn_threshold=model_options['learn_thresh'],
                reset_mechanism=model_options['reset_mechanism'], per_channel=model_options['per_channel'])
        )

    def reset_vmems(self):
        # Initialize hidden states and outputs at t=0 ###|SNN without BN|###
        self.encoder1[1].reset_state()
        self.encoder2[1].reset_state()
        self.encoder3[1].reset_state()
        self.encoder4[1].reset_state()

        self.residual11[1].reset_state()
        self.residual12[1].reset_state()
        self.residual21[1].reset_state()
        self.residual22[1].reset_state()

        self.predict_flow4_acc.reset_state()
        self.predict_flow3_acc.reset_state()
        self.predict_flow2_acc.reset_state()
        self.predict_flow1_acc.reset_state()

        self.deconv1[1].reset_state()
        self.deconv2[1].reset_state()
        self.deconv3[1].reset_state()

        self.upsampled_flow1_to_0[1].reset_state()
        self.upsampled_flow2_to_1[1].reset_state()
        self.upsampled_flow3_to_2[1].reset_state()
        self.upsampled_flow4_to_3[1].reset_state()

    def forward(self, event_voxel):
        # event_repr has a dimension (batch_size, num_bins, num_polarities, height, width)
        event_reprs = event_voxel.permute(0,2,3,4,1)

        new_event_reprs = torch.zeros(event_reprs.size(0), 4, event_reprs.size(2), event_reprs.size(3), 5).float().to(event_reprs.device)

        new_event_reprs[:, 0, :, :, :] = event_reprs[:,0,:,:,0:5] #former_inputs_on
        new_event_reprs[:, 1, :, :, :] = event_reprs[:,1,:,:,0:5] #former_inputs_off
        new_event_reprs[:, 2, :, :, :] = event_reprs[:,0,:,:,5:10] #latter_inputs_on
        new_event_reprs[:, 3, :, :, :] = event_reprs[:,1,:,:,5:10] #latter_inputs_off
        event_reprs = new_event_reprs

        skip_connections = {}

        #Reset
        self.reset_vmems()
        flow1_acc, flow2_acc, flow3_acc, flow4_acc = 0, 0, 0, 0

        for i in range(event_reprs.size(4)):
            x = event_reprs[:, :, :, :, i]

            #ENCODER
            out_enc1 = self.encoder1(x)
            out_enc2 = self.encoder2(out_enc1)
            out_enc3 = self.encoder3(out_enc2)
            out_enc4 = self.encoder4(out_enc3)

            #skip connections
            skip_connections['skip1'] = out_enc1.clone()
            skip_connections['skip2'] = out_enc2.clone()
            skip_connections['skip3'] = out_enc3.clone()
            skip_connections['skip4'] = out_enc4.clone()

            #RESIDUAL
            #Block-1
            out_r11 = self.residual11(out_enc4)
            out_r12 = self.residual12(out_r11) + out_enc4
            
            #Block-2
            out_r21 = self.residual21(out_r12)
            out_r22 = self.residual22(out_r21) + out_r12

            #DECODER
            flow4 = self.predict_flow4(self.upsampled_flow4_to_3(out_r22))
            flow4_acc = self.predict_flow4_acc(flow4)
            out_dec3 = self.deconv3(out_r22)

            concat3 = torch.cat([skip_connections['skip3'], out_dec3, flow4], dim=1)
            flow3 = self.predict_flow3(self.upsampled_flow3_to_2(concat3))
            flow3_acc = self.predict_flow3_acc(flow3)
            out_dec2 = self.deconv2(concat3)
            
            concat2 = torch.cat([skip_connections['skip2'], out_dec2, flow3], dim=1)
            flow2 = self.predict_flow2(self.upsampled_flow2_to_1(concat2))
            flow2_acc = self.predict_flow2_acc(flow2)
            out_dec1 = self.deconv1(concat2)

            concat1 = torch.cat([skip_connections['skip1'], out_dec1, flow2], dim=1)
            flow1 = self.predict_flow1(self.upsampled_flow1_to_0(concat1))
            flow1_acc = self.predict_flow1_acc(flow1)

        return flow4_acc, flow3_acc, flow2_acc, flow1_acc

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
    inps = torch.rand([1, 20, 2, 288, 384])  # (B, T, C, H, W)

    model_options = {"num_pols": 2, "base_channels": 64, "batchNorm": False, "learn_thresh": True, "learn_leak": True,
                     "ithresh": 1.0, "ileak": 1.0, "reset_mechanism": "soft", "per_channel": 0}
    model = AdaptiveFlowNet(model_options)

    print(model)
    outps = model(inps)
    print('len(outps) =', len(outps)) # num_encoders = 4 + 2*num_encoders = 8 + 2*num_decoders = 8
