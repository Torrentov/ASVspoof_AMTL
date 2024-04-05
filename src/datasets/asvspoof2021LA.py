from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.custom_audio import CustomAudioDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVSpoof2021LADataset(CustomAudioDataset):
    def __init__(self, audio_dir, part, *args, **kwargs):
        audio_dir = Path(audio_dir)
        if str(audio_dir)[0] != "/" and str(audio_dir)[0] != "\\":
            audio_dir = ROOT_PATH / audio_dir
        if part == "train":
            part_protocol = (
                audio_dir
                / "ASVspoof2019_LA_cm_protocols"
                / "ASVspoof2019.LA.cm.train.trn.txt"
            )
            part_dir = audio_dir / "ASVspoof2019_LA_train" / "flac"
        elif part == "dev":
            part_protocol = (
                audio_dir
                / "ASVspoof2019_LA_cm_protocols"
                / "ASVspoof2019.LA.cm.dev.trl.txt"
            )
            part_dir = audio_dir / "ASVspoof2019_LA_dev" / "flac"
        elif part == "eval":
            part_protocol = (
                audio_dir / "ASVspoof2021_LA_eval" / "ASVspoof2021.LA.cm.eval.trl.txt"
            )
            part_dir = audio_dir / "ASVspoof2021_LA_eval" / "flac"
        else:
            raise ValueError("Unknown part")
        data = []
        with open(part_protocol, "r") as f:
            for line in tqdm(f):
                entry = {}
                line = line.strip("\n")
                speaker_id, audio_file_name, _, system_id, label = line.split()
                entry["speaker_id"] = speaker_id
                entry["system_id"] = system_id
                entry["gt_label"] = int(label == "bonafide")
                entry["path"] = str(part_dir / (audio_file_name + ".flac"))
                if len(entry) > 0:
                    data.append(entry)
        super().__init__(data, *args, **kwargs)
