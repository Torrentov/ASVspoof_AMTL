from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in tqdm(data):
            assert "path" in entry
            assert Path(entry["path"]).exists(), f"Path {entry['path']} doesn't exist"
            entry["path"] = str(Path(entry["path"]).absolute().resolve())
            # t_info = torchaudio.info(entry["path"], format='flac')
            # entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)
