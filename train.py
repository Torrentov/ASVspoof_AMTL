import random
import warnings
import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate

from src.trainer import Trainer
from src.utils.data_utils import get_dataloaders
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="src/configs", config_name="resnet_initial_params")
def main(config):
    set_random_seed(config.trainer.seed)

    logger = setup_saving_and_logging(config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # audio_optimizer = instantiate(config.audio_optimizer, params=[{'params': model.audio_model.parameters()},
    #                                                        {'params': model.reconstruction_autoencoder.parameters()}])
    audio_optimizer = torch.optim.Adam(params=[{'params': model.audio_model.parameters()},
                                               {'params': model.reconstruction_autoencoder.parameters()}], lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)

    ca_optimizer = instantiate(config.ca_optimizer, params=model.conversion_autoencoder.parameters())
    sc_optimizer = instantiate(config.sc_optimizer, params=model.speaker_classifier.parameters())

    # lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        loss=loss,
        metrics=metrics,
        audio_optimizer=audio_optimizer,
        ca_optimizer=ca_optimizer,
        sc_optimizer=sc_optimizer,
        # lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        use_mtl=config.trainer.get("use_mtl", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
