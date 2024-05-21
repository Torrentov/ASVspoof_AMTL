from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.custom_audio import CustomAudioDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVSpoof2021LADataset(CustomAudioDataset):
    def __init__(self, audio_dir, part, limit=None, *args, **kwargs):
        self.part = part
        self.speaker_embedding = {
            'LA_0079': 0, 'LA_0080': 1, 'LA_0081': 2, 'LA_0082': 3, 'LA_0083': 4, 'LA_0084': 5, 'LA_0085': 6,
            'LA_0086': 7, 'LA_0087': 8,
            'LA_0088': 9, 'LA_0089': 10, 'LA_0090': 11, 'LA_0091': 12, 'LA_0092': 13, 'LA_0093': 14, 'LA_0094': 15,
            'LA_0095': 16, 'LA_0096': 17,
            'LA_0097': 18, 'LA_0098': 19
        }
        self.system_id_embedding = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                                   "A19": 19}
        audio_dir = Path(audio_dir)
        if str(audio_dir)[0] != "/" and str(audio_dir)[0] != "\\":
            audio_dir = ROOT_PATH / audio_dir
        if part == "train":
            part_protocol = (
                audio_dir
                / "ASVspoof2019_LA_cm_protocols"
                / "ASVspoof2019.LA.cm.train.trl.txt"
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
        elif part == "eval_2019":
            part_protocol = (
                audio_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
            )
            part_dir = audio_dir / "ASVspoof2019_LA_eval" / "flac"
        else:
            raise ValueError("Unknown part")
        data = []
        with open(part_protocol, "r") as f:
            for line in tqdm(f):
                entry = {}
                line = line.strip("\n")
                if part != "eval":
                    speaker_id, audio_file_name, _, system_id, label = line.split()
                else:
                    speaker_id, audio_file_name, _, _, system_id, label, _, _ = line.split()
                entry["speaker_id"] = speaker_id
                entry["system_id"] = system_id
                entry["gt_label"] = int(label == "spoof")
                entry["audio_file_name"] = audio_file_name
                entry["path"] = str(part_dir / (audio_file_name + ".flac"))
                if len(entry) > 0:
                    data.append(entry)
        super().__init__(data, *args, **kwargs)
    
    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object, sr = self.load_object(data_path)
        data_object = self.process_object(data_object)
        gt_label = data_dict.get("gt_label", -1)
        if self.part == "train":
            speaker_id = self.speaker_embedding[data_dict.get("speaker_id", "")]
        else:
            speaker_id = -1
        system_id = self.system_id_embedding.get(data_dict.get("system_id", "-"), 0)
        audio_file_name = int(data_dict.get("audio_file_name", "LA_E_-1")[5:])
        return {
            "audio": data_object,
            "duration": data_object.size(1) / sr,
            "audio_path": data_path,
            "gt_label": gt_label,
            "speaker_id": speaker_id,
            "system_id": system_id,
            "audio_file_name": audio_file_name,
        }