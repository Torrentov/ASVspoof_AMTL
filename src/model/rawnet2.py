import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base_model import BaseModel


class SincConvFast(nn.Module):  # from seminar
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=0,
        min_band_hz=0,
        scale_type="s1",
    ):
        super(SincConvFast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        if scale_type == "s1":
            mel = np.linspace(
                self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
            )
            hz = self.to_hz(mel)
        elif scale_type == "s2":
            mel = np.linspace(
                self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
            )
            hz = self.to_hz(mel)
            hz = np.abs(hz[::-1] - 1)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = torch.Tensor(hz[:-1]).view(-1, 1)

        # filter frequency band (out_channels, 1)
        self.band_hz_ = torch.Tensor(np.diff(hz)).view(-1, 1)

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.low_hz_ = self.low_hz_.to(waveforms.device)
        self.band_hz_ = self.band_hz_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class FMS(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).view(x.size(0), -1)
        y = self.linear(y)
        y = self.sigmoid(y).view(x.size(0), x.size(1), -1)

        return x * y + y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.first = first

        if not first:
            self.bn_start = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.bn_mid = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.resize_residual = in_channels != out_channels
        if self.resize_residual:
            self.conv_resize = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

        self.max_pool = nn.MaxPool1d(3)

        self.fms = FMS(out_channels)

    def forward(self, x):
        if not self.first:
            out = self.bn_start(x)
            out = F.leaky_relu(out, negative_slope=0.3)
        else:
            out = x

        out = self.conv1(out)

        out = self.bn_mid(out)
        out = F.leaky_relu(out, negative_slope=0.3)

        out = self.conv2(out)

        if self.resize_residual:
            out = out + self.conv_resize(x)
        else:
            out = out + x

        out = self.max_pool(out)

        out = self.fms(out)

        return out


class RawNet2(BaseModel):
    def __init__(
        self,
        sinc_out_channels,
        sinc_kernel_size,
        resblocks_channels,
        gru_size,
        gru_num_layers,
        fc_size,
        num_classes,
        sinc_sample_rate=16000,
        sinc_in_channels=1,
        sinc_stride=1,
        sinc_padding=0,
        sinc_dilation=1,
        sinc_bias=False,
        sinc_groups=1,
        sinc_min_low_hz=0,
        sinc_min_band_hz=0,
        sinc_scale_type="s1",
        **batch
    ):
        super().__init__(**batch)
        self.sinc_conv = SincConvFast(
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size,
            sample_rate=sinc_sample_rate,
            in_channels=sinc_in_channels,
            stride=sinc_stride,
            padding=sinc_padding,
            dilation=sinc_dilation,
            bias=sinc_bias,
            groups=sinc_groups,
            min_low_hz=sinc_min_low_hz,
            min_band_hz=sinc_min_band_hz,
            scale_type=sinc_scale_type,
        )

        self.after_sinc_conv = nn.Sequential(
            nn.MaxPool1d(3), nn.BatchNorm1d(resblocks_channels[0][0]), nn.SELU()
        )

        self.res_stack = nn.Sequential(
            ResBlock(*resblocks_channels[0], first=True),
            *[
                ResBlock(*resblocks_channels[i])
                for i in range(1, len(resblocks_channels))
            ]
        )

        self.pre_gru = nn.Sequential(
            nn.BatchNorm1d(resblocks_channels[-1][-1]), nn.LeakyReLU(negative_slope=0.3)
        )

        self.gru = nn.GRU(
            input_size=resblocks_channels[-1][-1],
            hidden_size=gru_size,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        self.fc_out = nn.Sequential(
            nn.Linear(gru_size, fc_size), nn.Linear(fc_size, num_classes)
        )

    def forward(self, audio_gt, **batch):
        x = audio_gt
        x = self.sinc_conv(x)
        # x = torch.abs(x)
        x = self.after_sinc_conv(x)

        x = self.res_stack(x)

        x = self.pre_gru(x)
        x = x.transpose(1, 2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]

        x = self.fc_out(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
