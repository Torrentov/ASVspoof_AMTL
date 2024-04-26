import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, all_probs, all_targets, **kwargs) -> (float, float):
        eer, thr = compute_eer(
            bonafide_scores=all_probs[all_targets == 0],
            other_scores=all_probs[all_targets == 1],
        )
        return eer, thr
