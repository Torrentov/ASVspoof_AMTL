import logging
from math import ceil
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}

    result_batch["audio"] = []
    result_batch["audio_path"] = []
    result_batch['speaker_id'] = []
    result_batch['audio_file_name'] = []
    result_batch['system_id'] = []
    result_batch["gt_label"] = []

    for elem in dataset_items:
        if elem["audio"].shape != torch.Size([1, 60, 750]):
            wave_current_length = elem['audio'].shape[1]
            if wave_current_length >= 64600:
                wave_padded = elem['audio'][:, :64600]
            else:
                wave_repeats = ceil(64600 / wave_current_length)
                wave_padded = torch.tile(elem['audio'], (1, wave_repeats))[:, :64600]
            result_batch["audio"].append(wave_padded)
        else:
            result_batch["audio"].append(elem["audio"])
        result_batch["audio_path"].append(elem["audio_path"])
        result_batch['speaker_id'].append(elem['speaker_id'])
        result_batch['audio_file_name'].append(elem['audio_file_name'])
        result_batch['system_id'].append(elem['system_id'])
        result_batch["gt_label"].append(elem["gt_label"])

    result_batch["audio"] = torch.stack(result_batch["audio"])
    result_batch['speaker_id'] = torch.LongTensor(result_batch['speaker_id'])
    result_batch['audio_file_name'] = torch.LongTensor(result_batch['audio_file_name'])
    result_batch['system_id'] = torch.LongTensor(result_batch['system_id'])
    result_batch["gt_label"] = torch.LongTensor(result_batch["gt_label"])

    return result_batch
