import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWrapper(CrossEntropyLoss):
    def __init__(self, weight=None):
        if weight is None:
            weight = [1.0, 9.0]
        super().__init__(weight=torch.tensor(weight))

    def forward(self, logits, gt_label, **batch) -> Tensor:
        return super().forward(
            input=logits,
            target=gt_label,
        )
