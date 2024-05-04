import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

import torchaudio

from src.utils.mel_spectrogram import MelSpectrogram


class CrossEntropyLossWrapper(nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        if weight is None:
            super().__init__()
        else:
            super().__init__(weight=torch.tensor(weight))

    def forward(self, logits, gt_label, **batch) -> torch.Tensor:
        return super().forward(
            input=logits,
            target=gt_label,
        )


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)


class MultiTaskStftLoss(nn.Module):
    def __init__(self, lr, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0, weight_loss=1.0, lambda_r=0.04, lambda_c=1.0, lambda_m=0.01, delta=1.0):
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(
            win_length=1024,
            hop_length=256,
            n_fft=1024,
            power=1,
            pad=(1024 - 256) // 2,
            center=False
        )
        
        self.weight_loss = weight_loss
        self.lambda_r = lambda_r
        self.lambda_c = lambda_c
        self.lambda_m = lambda_m
        self.delta = delta

        self.criterion = CrossEntropyLossWrapper()

        self.audio_loss = OCSoftmax(feat_dim, r_real, r_fake, alpha)
        self.audio_loss.train()
        self.audio_loss_optimizer = torch.optim.SGD(self.audio_loss.parameters(), lr=lr)
    
    def reconstruction_loss(self, audio1, audio2):
        spec1 = self.spectrogram(audio1).squeeze(1)
        spec2 = self.spectrogram(audio2).squeeze(1)
        if spec1.size(2) > spec2.size(2):
            padding_size = spec1.size(2) - spec2.size(2)
            spec2 = F.pad(spec2, (0, padding_size))
        if spec2.size(2) > spec1.size(2):
            padding_size = spec2.size(2) - spec1.size(2)
            spec1 = F.pad(spec1, (0, padding_size))
        return F.l1_loss(spec1, spec2)
    