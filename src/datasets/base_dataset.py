import logging
import random
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, index, limit=None, shuffle=True, instance_transforms=None):
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle)
        # index = self._sort_index(index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object, sr = self.load_object(data_path)
        data_object = self.process_object(data_object)
        gt_label = data_dict.get("gt_label", -1)
        speaker_id = data_dict.get("speaker_id", "")
        system_id = data_dict.get("system_id", "")
        return {
            "audio": data_object,
            "duration": data_object.size(1) / sr,
            "audio_path": data_path,
            "gt_label": gt_label,
            "speaker_id": speaker_id,
            "system_id": system_id,
        }

    def __len__(self):
        return len(self._index)

    def load_object(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        return audio_tensor, sr

    def process_object(self, data_object):
        if self.instance_transforms is not None:
            data_object = self.instance_transforms(data_object)
        return data_object

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        # TODO Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle):
        if shuffle:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
