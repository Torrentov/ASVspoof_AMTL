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

    result_batch["audio_gt"] = []
    result_batch["audio_path"] = []
    # result_batch['speaker_id'] = []
    # result_batch['system_id'] = []
    result_batch["gt_label"] = []

    for elem in dataset_items:
        wave_current_length = elem["audio"].shape[1]
        if wave_current_length >= 64000:
            wave_padded = elem["audio"][:, :64000]
        else:
            wave_repeats = ceil(64000 / wave_current_length)
            wave_padded = torch.tile(elem["audio"], (1, wave_repeats))[:, :64000]
        result_batch["audio_gt"].append(wave_padded)
        result_batch["audio_path"].append(elem["audio_path"])
        # result_batch['speaker_id'].append(elem['speaker_id'])
        # result_batch['system_id'].append(elem['system_id'])
        result_batch["gt_label"].append(elem["gt_label"])

    result_batch["audio_gt"] = torch.stack(result_batch["audio_gt"])
    # result_batch['speaker_id'] = torch.Tensor(result_batch['speaker_id'])
    # result_batch['system_id'] = torch.Tensor(result_batch['system_id'])
    result_batch["gt_label"] = torch.LongTensor(result_batch["gt_label"])

    return result_batch
