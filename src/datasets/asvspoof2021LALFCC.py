from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
import pickle

from src.datasets.custom_audio import CustomAudioDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)


def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


class ASVSpoof2021LALFCCDataset(CustomAudioDataset):
    def __init__(self, audio_dir, part, limit=None, feat_len=750, padding="repeat", *args, **kwargs):
        self.feat_len = feat_len
        self.padding=padding
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
            part_dir = audio_dir / "Features" / "train"
            part_prefix = ""
        elif part == "dev":
            part_protocol = (
                audio_dir
                / "ASVspoof2019_LA_cm_protocols"
                / "ASVspoof2019.LA.cm.dev.trl.txt"
            )
            part_dir = audio_dir / "Features" / "dev"
            part_prefix = ""
        elif part == "eval":
            part_protocol = (
                audio_dir / "ASVspoof2021_LA_eval" / "ASVspoof2021.LA.cm.eval.trl.txt"
            )
            part_dir = audio_dir / "Features" / "eval_2021"
            part_prefix = "2021_"
        elif part == "eval_2019":
            part_protocol = (
                audio_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
            )
            part_dir = audio_dir / "Features" / "eval_2019"
            part_prefix = "2019_"
        else:
            raise ValueError("Unknown part")
        data = []
        with open(part_protocol, "r") as f:
            counter = 0
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
                entry["path"] = str(part_dir / (part_prefix + audio_file_name + "LFCC.pkl"))
                if len(entry) > 0:
                    data.append(entry)
                    counter += 1
                    if limit is not None and counter >= limit:
                        break
        super().__init__(data, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
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
            "audio_path": data_path,
            "gt_label": gt_label,
            "speaker_id": speaker_id,
            "system_id": system_id,
            "audio_file_name": audio_file_name,
        }

    def __len__(self):
        return len(self._index)

    def load_object(self, path):
        with open(path, 'rb') as object_file:
            feat_mat = pickle.load(object_file)
        feat_mat = torch.from_numpy(feat_mat)
        return feat_mat
    
    def process_object(self, data_object):
        object_len = data_object.shape[1]
        if object_len > self.feat_len:
            startp = np.random.randint(object_len - self.feat_len)
            data_object = data_object[:, startp:startp + self.feat_len]
        if object_len < self.feat_len:
            if self.padding == 'zero':
                data_object = padding(data_object, self.feat_len)
            elif self.padding == 'repeat':
                data_object = repeat_padding(data_object, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')
        return data_object.unsqueeze(0).float()
        