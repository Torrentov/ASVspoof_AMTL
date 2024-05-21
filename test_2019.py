import random
import warnings
import os
import subprocess

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm
import torch.nn.functional as F

from src.trainer import Trainer
from src.utils.data_utils import get_dataloaders
from src.utils.init_utils import setup_saving_and_logging
from src.metrics.utils import calculate_tDCF_EER

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
                if loss is None:
                    batch["score"] = F.softmax(audio_outputs)[:, 0]
                else:
                    _, batch["score"] = loss.audio_loss(feats, batch["gt_label"])

                for i in range(len(batch["score"])):
                    system_id = str(batch["system_id"][i].item())
                    if len(system_id) == 1:
                        system_id = '0' + system_id
                    print(f"LA_E_{batch['audio_file_name'][i]} A{system_id} {'spoof' if batch['gt_label'][i].data.cpu().numpy() else 'bonafide'} {batch['score'][i]}", file=cm_score_file)


def test_custom_scores(cm_score_file_name, output_file):
    asv_score_file_name = "data/ASVspoof2021LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    calculate_tDCF_EER(cm_score_file_name, asv_score_file_name, output_file)


@hydra.main(version_base=None, config_path="src/configs", config_name="resnet_initial_params")
def main(config):
    global batch_transforms, device

    print(config_name)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config)
    evaluation_dataloader = None
    for k, v in dataloaders.items():
        if k == "eval_2019":
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
    # checkpoint_path = "saved/CheckpointSoftmax/"
    # checkpoint_name = "model_best.pth"
    # checkpoint = torch.load(checkpoint_path + checkpoint_name, device)
    # model.audio_model.load_state_dict(checkpoint["state_dict"])
    # loss = None

    cm_score_file_name = checkpoint_path + "score_2019_" + checkpoint_name.replace('.pth', '.txt')
    asv_score_file_name = "data/ASVspoof2021LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    output_file = "2019_res.txt"

    generate_score_file(model, loss, evaluation_dataloader, cm_score_file_name, config)

    calculate_tDCF_EER(cm_score_file_name, asv_score_file_name, output_file)


if __name__ == "__main__":
    main()
    # test_custom_scores("/home/asbekyan/Multi-Task-Learning-Improves-Synthetic-Speech-Detection/models/ocsoftmax_class_00005_recon_04_conver_0003/checkpoint_cm_score.txt", "resnet_paper_2019_res.txt")
