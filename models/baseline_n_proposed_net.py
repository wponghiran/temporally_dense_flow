
import torch
import torch.nn as nn

import math
import numpy as np
import models.spiking_util as spiking


class AverageTracker(object):
    """Computes and stores the average value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.n_sample = 0

    def update(self, val):
        new_n_sample = self.n_sample + torch.numel(val)
        self.avg = self.avg * (self.n_sample / new_n_sample) + (torch.sum(val).cpu() / new_n_sample)
        self.n_sample = new_n_sample

    def __repr__(self):
        return '{:.3e}'.format(self.avg)


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


class LIF(nn.Module):
    """
    Spking LIF cell.

    Design choices:
    - Arctan surrogate grad (Fang et al. 2021)
    - Hard reset (Ledinauskas et al. 2020) - questionable
    - Detach reset (Zenke et al. 2021)
    - Multiply previous voltage with leak; incoming current with (1 - leak) (Fang et al. 2020) - questionable
    - Make leak numerically stable with sigmoid (Fang et al. 2020)
    - Learnable threshold instead of bias
    - Per-channel leaks normally distributed (Yin et al. 2021)
    - Residual added to spikes (Fang et al. 2021)
    """

    def __init__(self,
                 param_dim=[32, 1, 1],  # [C, 1, 1] for having different leak & thresh for different channels
                 # or [1] for having the same leak & thresh across spatial dimension
                 activation='arctanspike',
                 act_width=10.0,
                 init_leak_param=(0.0, 0.0),  # (-4.0, 0.1) = -4*N(0,1)+0.1 or (0, 0) for default at tau=2.0
                 learn_leak=True,
                 init_thresh_param=(0.0, 1.0),  # (0.8, 0) = -0.8*N(0,1)+0 or (0, 1) for default at 1.0
                 learn_thresh=True,
                 hard_reset=True,
                 detach_reset=True,
                 ):
        super(LIF, self).__init__()

        self.learn_leak = learn_leak
        if learn_leak:
            self.leak_proxy = nn.Parameter(torch.randn(param_dim) * init_leak_param[1] + init_leak_param[0])
        else:
            self.register_buffer("leak_proxy", torch.randn(param_dim) * init_leak_param[1] + init_leak_param[0])

        self.learn_thresh = learn_thresh
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(param_dim) * init_thresh_param[1] + init_thresh_param[0])
        else:
            self.register_buffer("thresh", torch.randn(param_dim) * init_thresh_param[1] + init_thresh_param[0])

        # spiking and reset mechanics
        assert isinstance(activation, str), "See models/spiking_util.py for valid activation choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset

    def extra_repr(self):
        s = ('act={spike_fn.__name__}, hard_reset={hard_reset}, detach_reset={detach_reset}'
             ', learn_leak={learn_leak}, learn_thresh={learn_thresh}')
        return s.format(**self.__dict__)

    def forward(self, inp, prev_v=None, prev_z=None):
        # generate empty prev_states, if None is provided
        if (prev_v is None) or (prev_z is None):
            prev_v = torch.zeros_like(inp)
            prev_z = torch.zeros_like(inp)

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leak
        leak = torch.sigmoid(self.leak_proxy)

        # detach reset
        if self.detach_reset:
            prev_z = prev_z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            new_v = prev_v * leak * (1 - prev_z) + inp
        else:
            new_v = prev_v * leak + inp - prev_z * thresh

        # compute output
        new_z = self.spike_fn(new_v, thresh, self.act_width)

        return new_v, new_z


class SpikingBasicBlock(nn.Module):
    def __init__(self, num_inp_c, num_outp_c, kernel_size, stride, padding, norm, neuron_options, conn_class=nn.Conv2d):
        super().__init__()
        self.norm = norm
        assert norm in ['none', 'bn-all-ts'], '{} is an invalid normalization'.format(norm)

        self.conn_1 = conn_class(in_channels=num_inp_c, out_channels=num_outp_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False if self.norm else True)
        self.lif_1 = LIF(param_dim=[num_outp_c, 1, 1], **neuron_options)

        w_scale = math.sqrt(1/num_inp_c)
        nn.init.uniform_(self.conn_1.weight, -w_scale, w_scale)

        if self.norm == 'bn-all-ts':
            self.bn_1 = nn.BatchNorm2d(num_outp_c)

    def forward(self, inp, prev_v1, prev_z1):
        # Apply 1st convolutional layer
        v1 = self.conn_1(inp)
        # Apply 1st batch normalization before passing to neuron
        if self.norm == 'bn-all-ts':
            v1 = self.bn_1(v1)
        new_v1, new_z1 = self.lif_1(v1, prev_v1, prev_z1)

        return new_v1, new_z1


class SpikingResidualBlock(nn.Module):
    def __init__(self, num_c, kernel_size, stride, padding, norm, neuron_options, conn_class=nn.Conv2d):
        super().__init__()
        self.norm = norm
        assert norm in ['none', 'bn-all-ts'], '{} is an invalid normalization'.format(norm)

        self.conn_1 = conn_class(in_channels=num_c, out_channels=num_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False if self.norm else True)
        self.lif_1 = LIF(param_dim=[num_c, 1, 1], **neuron_options)

        self.conn_2 = conn_class(in_channels=num_c, out_channels=num_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False if self.norm else True)
        self.lif_2 = LIF(param_dim=[num_c, 1, 1], **neuron_options)

        w_scale = math.sqrt(1/num_c)
        nn.init.uniform_(self.conn_1.weight, -w_scale, w_scale)
        nn.init.uniform_(self.conn_2.weight, -w_scale, w_scale)

        if self.norm == 'bn-all-ts':
            self.bn_1 = nn.BatchNorm2d(num_c)
            self.bn_2 = nn.BatchNorm2d(num_c)

    def forward(self, inp, prev_v1, prev_z1, prev_v2, prev_z2):
        # Apply 1st convolutional layer
        v1 = self.conn_1(inp)
        # Apply 1st batch normalization before passing to neuron
        if self.norm == 'bn-all-ts':
            v1 = self.bn_1(v1)
        new_v1, new_z1 = self.lif_1(v1, prev_v1, prev_z1)

        # Apply 1st convolutional layer
        v2 = self.conn_2(new_z1)
        # Apply 1st batch normalization before passing to neuron
        if self.norm == 'bn-all-ts':
            v2 = self.bn_1(v2)
        new_v2, new_z2 = self.lif_2(v2, prev_v2, prev_z2)

        return new_z2 + inp, new_v1, new_z1, new_v2, new_z2


class EfficientSpikeFlowNet(BaseModel):
    def __init__(self, model_options, neuron_options={}):
        super().__init__()

        self.num_pols = model_options['num_pols']
        self.num_encoders = model_options['num_encoders']
        self.build_res_blocks = (model_options['num_res_blocks'] != 0)
        self.num_res_blocks = model_options['num_res_blocks']
        self.norm = model_options['norm']
        self.c_mul = model_options['c_mul'] # 2
        self.num_base_c = model_options['num_base_c']   # 32

        encoder_input_sizes = [int(self.num_base_c * pow(self.c_mul, i)) for i in range(self.num_encoders)]
        encoder_output_sizes = [int(self.num_base_c * pow(self.c_mul, i + 1)) for i in range(self.num_encoders)]

        # Creating encoder blocks
        encoders = []
        for i, (input_size, output_size) in enumerate(zip(encoder_input_sizes, encoder_output_sizes)):
            encoders.append(
                SpikingBasicBlock(num_inp_c=self.num_pols if i == 0 else input_size, num_outp_c=output_size,
                                  kernel_size=3, stride=2, padding=1, norm=self.norm, neuron_options=neuron_options))
        self.encoders = nn.ModuleList(encoders)

        # Creating residual blocks
        if self.build_res_blocks:
            res_blocks = []
            for i in range(self.num_res_blocks):
                res_blocks.append(
                    SpikingResidualBlock(num_c=output_size, kernel_size=3, stride=1, padding=1, norm=self.norm,
                                         neuron_options=neuron_options)
                )
            self.res_blocks = nn.ModuleList(res_blocks)

        # Creating decoder blocks
        # Spike-FlowNet utilizes only 3 up-sampling blocks to be decoder
        # while EV-FlowNet, conventional U-Net architecture uses 4 up-sampling blocks
        decoder_input_sizes = reversed(encoder_output_sizes)
        decoder_output_sizes = reversed(encoder_input_sizes)
        decoders = []
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            num_skip_c = 0 if ((not self.build_res_blocks) and (i == 0)) else input_size
            num_xtra_c = 0 if i == 0 else 2
            decoders.append(
                SpikingBasicBlock(num_inp_c=input_size + num_xtra_c + num_skip_c, num_outp_c=output_size, kernel_size=4,
                                  stride=2, padding=1, norm=self.norm, neuron_options=neuron_options,
                                  conn_class=nn.ConvTranspose2d))
        self.decoders = nn.ModuleList(decoders)

        # Creating prediction blocks
        # Spike-FlowNet uses casecaded convolutional transpose and convolutional for flow prediction
        # while EV-FlowNet, EventFlow use only convolution block
        pred_blocks = []
        for output_size in reversed(encoder_input_sizes):
            pred_blocks.append(
                nn.Conv2d(in_channels=output_size, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.pred_blocks = nn.ModuleList(pred_blocks)

        # Initialize operation counter
        self.n_nonzero_inp_trackers = [AverageTracker() for _ in range(self.num_encoders * 3 + self.num_res_blocks)]
        self.n_inp_trackers = [AverageTracker() for _ in range(self.num_encoders * 3 + self.num_res_blocks)]

    def forward(self, inps, prev_v1s=None, prev_z1s=None):
        # inps.shape = (batch_size, num_bins, num_polarities, height, width)
        if (prev_v1s is None) or (prev_z1s is None):
            prev_v1s = [None for _ in range(self.num_encoders * 2 + self.num_res_blocks * 2)]
            prev_z1s = [None for _ in range(self.num_encoders * 2 + self.num_res_blocks * 2)]

        all_ts_predictions = []
        for t in range(inps.size(1)):
            # encoder
            blocks = []
            offset = 0
            x = inps[:, t]
            for i, encoder in enumerate(self.encoders):
                self.n_nonzero_inp_trackers[i].update(torch.sum((x != 0), dim=[1, 2, 3]).cpu())
                self.n_inp_trackers[i].update(torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())

                prev_v1s[i], x = encoder(x, prev_v1s[i], prev_z1s[i])
                prev_z1s[i] = x
                blocks.append(x)

            # residual blocks
            offset += self.num_encoders
            if self.build_res_blocks:
                for i, res_block in enumerate(self.res_blocks):
                    self.n_nonzero_inp_trackers[self.num_encoders + i].update(torch.sum((x != 0), dim=[1, 2, 3]).cpu())
                    self.n_inp_trackers[self.num_encoders + i].update(
                        torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())
                    x, prev_v1s[offset + (2 * i)], prev_z1s[offset + (2 * i)], \
                    prev_v1s[offset + (2 * i) + 1], prev_z1s[offset + (2 * i) + 1] = \
                        res_block(x, prev_v1s[offset + (2 * i)], prev_z1s[offset + (2 * i)],
                                  prev_v1s[offset + (2 * i) + 1], prev_z1s[offset + (2 * i) + 1])

            # decoder and multires predictions
            predictions = []
            offset += 2 * self.num_res_blocks if self.build_res_blocks else 0
            for i, (decoder, pred_block) in enumerate(zip(self.decoders, self.pred_blocks)):
                if i == 0:
                    if self.build_res_blocks:
                        x = torch.cat([x, blocks[self.num_encoders - i - 1]], dim=1)
                else:
                    x = torch.cat([x, blocks[self.num_encoders - i - 1], predictions[-1]], dim=1)
                self.n_nonzero_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i].update(
                    torch.sum((x != 0), dim=[1, 2, 3]).cpu())
                self.n_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i].update(
                    torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())

                prev_v1s[offset + i], x = decoder(x, prev_v1s[offset + i], prev_z1s[offset + i])
                prev_z1s[offset + i] = x

                self.n_nonzero_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i + 1].update(
                    torch.sum((x != 0), dim=[1, 2, 3]).cpu())
                self.n_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i + 1].update(
                    torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())

                predictions.append(pred_block(x))
            all_ts_predictions.append(predictions[-1])

        return all_ts_predictions + prev_v1s + prev_z1s


class LSTMBasicBlock(nn.Module):
    def __init__(self, num_inp_c, num_outp_c, kernel_size, stride, padding, norm, conn_class=nn.Conv2d):
        super().__init__()
        self.norm = norm
        assert norm in ['none', 'bn-all-ts'], '{} is an invalid normalization'.format(norm)

        self.conv_x = conn_class(in_channels=num_inp_c, out_channels=4 * num_outp_c, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=False if self.norm else True)
        self.conv_h = conn_class(in_channels=num_outp_c, out_channels=4 * num_outp_c, kernel_size=3, stride=1,
                                 padding=1, bias=False if self.norm else True)

        if self.norm == 'bn-all-ts':
            self.bn_wfx = nn.BatchNorm2d(num_outp_c)
            self.bn_wix = nn.BatchNorm2d(num_outp_c)
            self.bn_wox = nn.BatchNorm2d(num_outp_c)
            self.bn_wcx = nn.BatchNorm2d(num_outp_c)

        # Initialize inp -> outp connections with kaiming normalize
        wx_f = torch.empty(num_outp_c if conn_class is nn.Conv2d else num_inp_c,
                           num_inp_c if conn_class is nn.Conv2d else num_outp_c, kernel_size, kernel_size)
        nn.init.kaiming_normal_(wx_f, nonlinearity='sigmoid')
        wx_i = torch.empty(num_outp_c if conn_class is nn.Conv2d else num_inp_c,
                           num_inp_c if conn_class is nn.Conv2d else num_outp_c, kernel_size, kernel_size)
        nn.init.kaiming_normal_(wx_i, nonlinearity='sigmoid')
        wx_o = torch.empty(num_outp_c if conn_class is nn.Conv2d else num_inp_c,
                           num_inp_c if conn_class is nn.Conv2d else num_outp_c, kernel_size, kernel_size)
        nn.init.kaiming_normal_(wx_o, nonlinearity='sigmoid')
        wx_c = torch.empty(num_outp_c if conn_class is nn.Conv2d else num_inp_c,
                           num_inp_c if conn_class is nn.Conv2d else num_outp_c, kernel_size, kernel_size)
        nn.init.kaiming_normal_(wx_c, nonlinearity='tanh')
        self.conv_x.weight.data.copy_(torch.cat((wx_f, wx_i, wx_o, wx_c), dim=(0 if conn_class is nn.Conv2d else 1)))
        # Initialize prev_outp -> outp connections with orthogonal
        wh_f = torch.empty(num_outp_c, num_outp_c, 3, 3)
        nn.init.orthogonal_(wh_f, gain=torch.nn.init.calculate_gain('sigmoid'))
        wh_i = torch.empty(num_outp_c, num_outp_c, 3, 3)
        nn.init.orthogonal_(wh_i, gain=torch.nn.init.calculate_gain('sigmoid'))
        wh_o = torch.empty(num_outp_c, num_outp_c, 3, 3)
        nn.init.orthogonal_(wh_o, gain=torch.nn.init.calculate_gain('sigmoid'))
        wh_c = torch.empty(num_outp_c, num_outp_c, 3, 3)
        nn.init.orthogonal_(wh_c, gain=torch.nn.init.calculate_gain('tanh'))
        self.conv_h.weight.data.copy_(torch.cat((wh_f, wh_i, wh_o, wh_c), dim=(0 if conn_class is nn.Conv2d else 1)))

    def forward(self, inp, prev_c=None, prev_h=None):
        wfx, wix, wox, wcx = (self.conv_x(inp)).chunk(4, 1)

        if self.norm == 'bn-all-ts':
            wfx = self.bn_wfx(wfx)
            wix = self.bn_wix(wix)
            wox = self.bn_wox(wox)
            wcx = self.bn_wcx(wcx)

        if (prev_c is None) or (prev_h is None):
            prev_c = torch.zeros_like(wfx)
            prev_h = torch.zeros_like(wfx)

        ufh, uih, uoh, uch = (self.conv_h(prev_h)).chunk(4, 1)
        f = torch.sigmoid(wfx + ufh)
        i = torch.sigmoid(wix + uih)
        o = torch.sigmoid(wox + uoh)
        next_c = i * torch.tanh(wcx + uch) + f * prev_c
        next_h = o * torch.tanh(next_c)

        return next_c, next_h


class LSTMEVFlowNet(BaseModel):
    def __init__(self, model_options):
        super().__init__()

        self.num_pols = model_options['num_pols']
        self.num_encoders = model_options['num_encoders']
        self.build_res_blocks = (model_options['num_res_blocks'] != 0)
        self.num_res_blocks = model_options['num_res_blocks']
        self.norm = model_options['norm']
        self.c_mul = model_options['c_mul'] # 2
        self.num_base_c = model_options['num_base_c']   # 32

        encoder_input_sizes = [int(self.num_base_c * pow(self.c_mul, i)) for i in range(self.num_encoders)]
        encoder_output_sizes = [int(self.num_base_c * pow(self.c_mul, i + 1)) for i in range(self.num_encoders)]

        # Creating encoder blocks
        encoders = []
        for i, (input_size, output_size) in enumerate(zip(encoder_input_sizes, encoder_output_sizes)):
            encoders.append(
                LSTMBasicBlock(num_inp_c=self.num_pols if i == 0 else input_size, num_outp_c=output_size,
                               kernel_size=3, stride=2, padding=1, norm=self.norm))
        self.encoders = nn.ModuleList(encoders)

        # Creating residual blocks
        if self.build_res_blocks:
            res_blocks = []
            for i in range(self.num_res_blocks):
                res_blocks.append(
                    ResidualBlock(num_c=output_size, kernel_size=3, stride=1, padding=1, norm=self.norm)
                )
            self.res_blocks = nn.ModuleList(res_blocks)

        # Creating decoder blocks
        decoder_input_sizes = reversed(encoder_output_sizes)
        decoder_output_sizes = reversed(encoder_input_sizes)
        decoders = []
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            num_skip_c = 0 if ((not self.build_res_blocks) and (i == 0)) else input_size
            num_xtra_c = 0 if i == 0 else 2
            decoders.append(
                LSTMBasicBlock(num_inp_c=input_size + num_xtra_c + num_skip_c, num_outp_c=output_size, kernel_size=4,
                               stride=2, padding=1, norm=self.norm, conn_class=nn.ConvTranspose2d))
        self.decoders = nn.ModuleList(decoders)

        # Creating prediction blocks
        pred_blocks = []
        for output_size in reversed(encoder_input_sizes):
            pred_blocks.append(
                nn.Conv2d(in_channels=output_size, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.pred_blocks = nn.ModuleList(pred_blocks)

    def forward(self, inps, prev_c1s=None, prev_h1s=None):
        if (prev_c1s is None) or (prev_h1s is None):
            prev_c1s = [None for _ in range(self.num_encoders * 2)]
            prev_h1s = [None for _ in range(self.num_encoders * 2)]

        all_ts_predictions = []
        for t in range(inps.size(1)):
            # encoder
            blocks = []
            offset = 0
            x = inps[:, t]
            for i, encoder in enumerate(self.encoders):
                prev_c1s[i], x = encoder(x, prev_c1s[i], prev_h1s[i])
                prev_h1s[i] = x
                blocks.append(x)

            # residual blocks
            if self.build_res_blocks:
                for i, res_block in enumerate(self.res_blocks):
                    x = res_block(x)

            # decoder and multires predictions
            predictions = []
            offset += self.num_encoders
            for i, (decoder, pred_block) in enumerate(zip(self.decoders, self.pred_blocks)):
                if i == 0:
                    if self.build_res_blocks:
                        x = torch.cat([x, blocks[self.num_encoders - i - 1]], dim=1)
                else:
                    x = torch.cat([x, blocks[self.num_encoders - i - 1], predictions[-1]], dim=1)
                prev_c1s[offset + i], x = decoder(x, prev_c1s[offset + i], prev_h1s[offset + i])
                prev_h1s[offset + i] = x
                predictions.append(pred_block(x))
            all_ts_predictions.append(predictions[-1])

        # return predictions + prev_c1s + prev_h1s
        return all_ts_predictions + prev_c1s + prev_h1s


class BasicBlock(nn.Module):
    def __init__(self, num_inp_c, num_outp_c, kernel_size, stride, padding, norm, conn_class=nn.Conv2d):
        super().__init__()
        self.norm = norm
        assert norm in ['none', 'bn-all-ts'], '{} is an invalid normalization'.format(norm)

        self.conn_1 = conn_class(in_channels=num_inp_c, out_channels=num_outp_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False if self.norm else True)
        self.act = nn.ReLU()

        w_scale = math.sqrt(1/num_inp_c)
        nn.init.uniform_(self.conn_1.weight, -w_scale, w_scale)

        if self.norm == 'bn-all-ts':
            self.bn_1 = nn.BatchNorm2d(num_outp_c)

    def forward(self, inp):
        # Apply 1st convolutional layer
        x = self.conn_1(inp)
        # Apply 1st batch normalization before passing to neuron
        if self.norm == 'bn-all-ts':
            x = self.bn_1(x)

        return self.act(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_c, kernel_size, stride, padding, norm, conn_class=nn.Conv2d):
        super().__init__()
        self.norm = norm
        assert norm in ['none', 'bn-all-ts'], '{} is an invalid normalization'.format(norm)

        self.conn_1 = conn_class(in_channels=num_c, out_channels=num_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False if self.norm else True)
        self.act_1 = nn.ReLU()
        self.conn_2 = conn_class(in_channels=num_c, out_channels=num_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False if self.norm else True)
        self.act_2 = nn.ReLU()

        w_scale = math.sqrt(1/num_c)
        nn.init.uniform_(self.conn_1.weight, -w_scale, w_scale)
        nn.init.uniform_(self.conn_2.weight, -w_scale, w_scale)

        if self.norm == 'bn-all-ts':
            self.bn_1 = nn.BatchNorm2d(num_c)
            self.bn_2 = nn.BatchNorm2d(num_c)

    def forward(self, inp):
        # Apply 1st convolutional layer
        x = self.conn_1(inp)
        # Apply 1st batch normalization before passing to activation function
        if self.norm == 'bn-all-ts':
            x = self.bn_1(x)
        x = self.act_1(x)

        # Apply 2nd convolutional layer
        x = self.conn_2(x)
        # Apply 2nd batch normalization before passing to activation function
        if self.norm == 'bn-all-ts':
            x = self.bn_2(x)
        x = self.act_2(x + inp)

        return x


class NonSpikingEVFlowNet(BaseModel):
    def __init__(self, model_options):
        super().__init__()

        self.num_pols = model_options['num_pols']
        self.num_encoders = model_options['num_encoders']
        self.build_res_blocks = (model_options['num_res_blocks'] != 0)
        self.num_res_blocks = model_options['num_res_blocks']
        self.norm = model_options['norm']
        self.c_mul = model_options['c_mul'] # 2
        self.num_base_c = model_options['num_base_c']   # 32

        encoder_input_sizes = [int(self.num_base_c * pow(self.c_mul, i)) for i in range(self.num_encoders)]
        encoder_output_sizes = [int(self.num_base_c * pow(self.c_mul, i + 1)) for i in range(self.num_encoders)]

        # Creating encoder blocks
        encoders = []
        for i, (input_size, output_size) in enumerate(zip(encoder_input_sizes, encoder_output_sizes)):
            encoders.append(
                BasicBlock(num_inp_c=self.num_pols if i == 0 else input_size, num_outp_c=output_size, kernel_size=3,
                           stride=2, padding=1, norm=self.norm))
        self.encoders = nn.ModuleList(encoders)

        # Creating residual blocks
        if self.build_res_blocks:
            res_blocks = []
            for i in range(self.num_res_blocks):
                res_blocks.append(
                    ResidualBlock(num_c=output_size, kernel_size=3, stride=1, padding=1, norm=self.norm)
                )
            self.res_blocks = nn.ModuleList(res_blocks)

        # Creating decoder blocks
        # Spike-FlowNet utilizes only 3 up-sampling blocks to be decoder
        # while EV-FlowNet, conventional U-Net architecture uses 4 up-sampling blocks
        decoder_input_sizes = reversed(encoder_output_sizes)
        decoder_output_sizes = reversed(encoder_input_sizes)
        decoders = []
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            num_skip_c = 0 if ((not self.build_res_blocks) and (i == 0)) else input_size
            num_xtra_c = 0 if i == 0 else 2
            decoders.append(
                BasicBlock(num_inp_c=input_size + num_xtra_c + num_skip_c, num_outp_c=output_size, kernel_size=4,
                           stride=2, padding=1, norm=self.norm, conn_class=nn.ConvTranspose2d))
        self.decoders = nn.ModuleList(decoders)

        # Creating prediction blocks
        # Spike-FlowNet uses casecaded convolutional transpose and convolutional for flow prediction
        # while EV-FlowNet, EventFlow use only convolution block
        pred_blocks = []
        for output_size in reversed(encoder_input_sizes):
            pred_blocks.append(
                nn.Conv2d(in_channels=output_size, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.pred_blocks = nn.ModuleList(pred_blocks)

        # # Initialize operation counter
        # self.n_nonzero_inp_trackers = [AverageTracker() for _ in range(self.num_encoders * 3 + self.num_res_blocks)]
        # self.n_inp_trackers = [AverageTracker() for _ in range(self.num_encoders * 3 + self.num_res_blocks)]

    def forward(self, inps):
        # Reshape the input since the feed-forward model have no notion of time-step
        inps = inps.view([inps.shape[0], -1]+list(inps.shape[3:]))

        # encoder
        blocks = []
        x = inps
        for i, encoder in enumerate(self.encoders):
            # self.n_nonzero_inp_trackers[i].update(torch.sum((x != 0), dim=[1, 2, 3]).cpu())
            # self.n_inp_trackers[i].update(torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        if self.build_res_blocks:
            for i, res_block in enumerate(self.res_blocks):
                # self.n_nonzero_inp_trackers[self.num_encoders+i].update(torch.sum((x != 0), dim=[1, 2, 3]).cpu())
                # self.n_inp_trackers[self.num_encoders + i].update(
                #     torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())
                x = res_block(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred_block) in enumerate(zip(self.decoders, self.pred_blocks)):
            if i == 0:
                if self.build_res_blocks:
                    x = torch.cat([x, blocks[self.num_encoders - i - 1]], dim=1)
            else:
                x = torch.cat([x, blocks[self.num_encoders - i - 1], predictions[-1]], dim=1)

            # self.n_nonzero_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i].update(
            #     torch.sum((x != 0), dim=[1, 2, 3]).cpu())
            # self.n_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i].update(
            #     torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())

            x = decoder(x)

            # self.n_nonzero_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i + 1].update(
            #     torch.sum((x != 0), dim=[1, 2, 3]).cpu())
            # self.n_inp_trackers[self.num_encoders + self.num_res_blocks + 2 * i + 1].update(
            #     torch.sum(torch.ones_like(x, dtype=torch.bool), dim=[1, 2, 3]).cpu())

            predictions.append(pred_block(x))

        return predictions


if __name__ == '__main__':
    _seed_ = 16146
    import random

    random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)

    # print('--- Test LIF() ---')
    # inp = torch.rand([4, 32, 128, 128]) # (B, C, H, W)
    # print('inp.shape =', inp.shape)

    # # Test neuron with layer normalization
    # neuron_options = {'param_dim': [32, 1, 1], 'layer_norm': True, 'norm_dim': [128, 128]}
    # lif_1 = LIF(**neuron_options)
    # outp, states = lif_1(inp)
    # print('outp.shape =', outp.shape)

    # print('--- Test BasicBlock() ---')
    # inp = torch.rand([4, 2, 128, 128]) # (B, C, H, W)
    # print('inp.shape =', inp.shape)
    #
    # # Test neuron with layer normalization
    # block = LSTMBasicBlock(num_inp_c=2, num_outp_c=4, kernel_size=3,
    #                        stride=2, padding=1, norm='bn-all-ts')
    # outp, states = block(inp)
    # print('outp.shape =', outp.shape)

    print('--- Test Model ---')
    # (batch_size, num_bins, num_polarities, height, width)
    inps = torch.rand([1, 20, 2, 288, 384])  # (B, T, C, H, W)

    # # Test block with batch normalization
    # # model_options = {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 0, 'norm': 'bn-all-ts', 'c_mul': 2,
    # #                  'num_base_c': 32}
    # model_options = {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 2, 'norm': 'bn-all-ts', 'c_mul': 2,
    #                  'num_base_c': 32}
    # neuron_options = {'act_width': 4.0}
    # model = EfficientSpikeFlowNet(model_options, neuron_options)

    # model_options = {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 0, 'norm': 'bn-all-ts', 'c_mul': 2,
    #                  'num_base_c': 32}
    model_options = {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 2, 'norm': 'bn-all-ts', 'c_mul': 2,
                     'num_base_c': 32}
    model = LSTMEVFlowNet(model_options)

    # # model_options = {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 0, 'norm': 'bn-all-ts', 'c_mul': 2,
    # #                  'num_base_c': 32}
    # model_options = {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 2, 'norm': 'bn-all-ts', 'c_mul': 2,
    #                  'num_base_c': 32}
    # model = NonSpikingEVFlowNet(model_options)

    print(model)
    outps = model(inps)
    print('len(outps) =', len(outps)) # num_encoders = 4 + 2*num_encoders = 8 + 2*num_decoders = 8

