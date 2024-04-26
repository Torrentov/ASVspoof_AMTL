import random
import warnings
import os
import subprocess

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.trainer import Trainer
from src.utils.data_utils import get_dataloaders
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def move_batch_to_device(batch, config):
    """
    Move all necessary tensors to the device
    """
    for tensor_for_device in config.trainer.device_tensors:
        batch[tensor_for_device] = batch[tensor_for_device].to(device)
    return batch

def transform_batch(batch):
    transform_type = "inference"
    transforms = batch_transforms.get(transform_type)
    if transforms is not None:
        print(transforms.keys())
        for transform_name in transforms.keys():
            batch[transform_name] = transforms[transform_name](
                batch[transform_name]
            )
    return batch

def generate_score_file(model, loss, dataloader, score_file_name, config):
    """
    Validate after training an epoch

    :return: A log that contains information about validation
    """
    model.audio_model.eval()
    all_probs = []
    all_targets = []
    with open(score_file_name, 'w') as cm_score_file:
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            ):
                batch = move_batch_to_device(batch, config)
                batch = transform_batch(batch)

                feats, audio_outputs = model.audio_model(batch["audio"])
                _, batch["score"] = loss.audio_loss(feats, batch["gt_label"])

                # print("SCORE:", batch["score"].shape)
                # print("AUDIO_FILE_NAME:", batch["audio_file_name"].shape)

                for i in range(len(batch["score"])):
                    print(f"LA_E_{batch['audio_file_name'][i]} {batch['score'][i]}", file=cm_score_file)


@hydra.main(version_base=None, config_path="src/configs", config_name="resnet_initial_params")
def main(config):
    global batch_transforms, device

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config)
    evaluation_dataloader = None
    for k, v in dataloaders.items():
        if k == "eval":
            evaluation_dataloader = v

    # get function handles of loss and metrics
    loss = instantiate(config.loss).to(device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    checkpoint_path = config.test.checkpoint_path
    checkpoint_name = config.test.checkpoint_name
    checkpoint = torch.load(checkpoint_path + checkpoint_name, device)

    if checkpoint.get("state_dict") is not None:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError("Invalid checkpoint structure")

    if checkpoint.get("loss") is not None:
        loss.load_state_dict(checkpoint["loss"])
    else:
        raise ValueError("Invalid checkpoint structure")

    cm_score_file_name = checkpoint_path + "score_" + checkpoint_name.replace('.pth', '.txt')

    generate_score_file(model, loss, evaluation_dataloader, cm_score_file_name, config)

    subprocess.run(["python", "2021/eval-package/main.py", "--cm-score-file", cm_score_file_name,
                    "--track", "LA", "--subset", "eval", "--metadata", "2021/eval-package/keys/"])


if __name__ == "__main__":
    main()
