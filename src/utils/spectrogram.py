from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa


@dataclass
class SpectrogramConfig:
    win_length: int = 304
    hop_length: int = 152
    n_fft: int = 304


class Spectrogram(nn.Module):

    def __init__(self, config=None):
        super(Spectrogram, self).__init__()

        if config is None:
            config = SpectrogramConfig()
        self.config = config

        self.spectrogram = torchaudio.transforms.Spectrogram(
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            power=None
        )


    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_fft // 2 + 1, T']
        """

        spec = self.spectrogram(audio)

        magnitude = torch.abs(spec)
        phase = torch.angle(spec)

        return magnitude, phase


class InverseSpectrogram(nn.Module):

    def __init__(self, config=None):
        super(InverseSpectrogram, self).__init__()

        if config is None:
            config = SpectrogramConfig()
        self.config = config

        self.inverse_spectrogram = torchaudio.transforms.InverseSpectrogram(
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft
        )


    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        :param magnitude: Expected shape is [B, n_fft // 2 + 1, T]
        :param phase: Expected shape is [B, n_fft // 2 + 1, T]
        :return: Shape is [B, T']
        """

        audio = self.inverse_spectrogram(torch.polar(magnitude, phase))

        return audio


if __name__ == "__main__":

    config = SpectrogramConfig()

    config.hop_length = 152
    config.win_length = config.hop_length * 2
    config.n_fft = config.hop_length * 2
    
    spec = Spectrogram(config)
    inv = InverseSpectrogram(config)

    waveform = torch.randn(1, 64600)

    print(spec(waveform)[0].shape)

    print(inv(*spec(waveform)).shape)

