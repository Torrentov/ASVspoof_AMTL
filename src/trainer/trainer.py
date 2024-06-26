import torch
import os
from tqdm import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.data_utils import inf_loop

import torch.nn.functional as F

from src.baseline.eval_package.main import evaluation_API


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.is_train = True
        self.model.audio_model.train()
        self.train_metrics.reset()
        all_probs = []
        all_targets = []
        self.writer.add_scalar("epoch", epoch)
        if self.config.adjust_lr.use_adjust_lr:
            self.adjust_learning_rate(self.audio_optimizer, epoch)
            self.adjust_learning_rate(self.loss.audio_loss_optimizer, epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics
                )
                all_probs.append(batch["scores"])
                all_targets.append(batch["gt_label"])
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} OCSoftmax: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["ocsoftmax"].item()
                    )
                )
                # TODO: add proper learning rate logging
                # self.writer.add_scalar(
                #     "learning rate", self.lr_scheduler.get_last_lr()[0]
                # )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        all_probs = torch.cat(all_probs, 0).data.cpu().numpy()
        all_targets = torch.cat(all_targets, 0).data.cpu().numpy()
        log = last_train_metrics

        log["train_eer"] = self._log_eer(all_probs, all_targets)

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log, val_eer, val_min_tdcf = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
            log["dev_eer"] = val_eer
            log["dev_min_tDCF"] = val_min_tdcf

        return log

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train:
            if self.use_mtl:
                feats, _ = self.model.audio_model(batch["audio"].squeeze(1))

                index = (batch["gt_label"] == 1).nonzero().squeeze()
                feature_select = feats.index_select(0, index)
                label_select_for_classify = batch["speaker_id"].index_select(0,index)
                self.sc_optimizer.zero_grad()
                classify_result = self.model.speaker_classifier(feature_select)
                loss_speaker_norm = self.loss.criterion(classify_result, label_select_for_classify)
                loss_speaker_norm = loss_speaker_norm * self.loss.lambda_m

                index = (batch["gt_label"] == 1).nonzero().squeeze()
                feats_select = feats.index_select(0, index)
                audio_re = self.model.reconstruction_autoencoder(feats_select)
                audio_ = batch["audio"].index_select(0, index)
                loss2 = self.loss.reconstruction_loss(audio_, audio_re) / len(index) * len(batch["audio"]) * self.loss.lambda_r

                index = (batch["gt_label"] == 0).nonzero().squeeze()
                audio_select = batch["audio"].index_select(0, index)
                audio_loss = 0
                if audio_select.shape[0] > 0:
                    re_ad = self.model.conversion_autoencoder(audio_select)
                    feats_ad, _ = self.model.audio_model(re_ad.squeeze(1))
                    ocsoftmaxloss_adv, _ = self.loss.audio_loss(feats_ad, batch["gt_label"].index_select(0, index))
                    audio_loss = ocsoftmaxloss_adv * self.loss.weight_loss * self.loss.lambda_c

                audio_loss = audio_loss + loss_speaker_norm

                ocsoftmaxloss, batch["scores"] = self.loss.audio_loss(feats, batch["gt_label"])
                loss1 = ocsoftmaxloss * self.loss.weight_loss
                audio_loss = loss1 + loss2 + audio_loss
                self.audio_optimizer.zero_grad()
                self.loss.audio_loss_optimizer.zero_grad()
                batch["ocsoftmax"] = ocsoftmaxloss
                audio_loss.backward()
                self.audio_optimizer.step()
                self.loss.audio_loss_optimizer.step()
                self.sc_optimizer.step()

                index = (batch["gt_label"] == 0).nonzero().squeeze()
                audio_select = batch["audio"].index_select(0, index)
                if audio_select.shape[0] > 0:
                    re_ad = self.model.conversion_autoencoder(audio_select)
                    feats_ad, _ = self.model.audio_model(re_ad.squeeze(1))
                    loss_recon = self.loss.reconstruction_loss(re_ad, audio_select)
                    loss_ad, _ = self.loss.audio_loss(feats_ad, batch["gt_label"].index_select(0, index).fill_(1))
                    loss_for_G = self.loss.delta * loss_ad + loss_recon
                    self.ca_optimizer.zero_grad()
                    loss_for_G.backward()
                    self.ca_optimizer.step()
            else:
                feats, _ = self.model.audio_model(batch["audio"].squeeze(1))
                ocsoftmaxloss, batch["scores"] = self.loss.audio_loss(feats, batch["gt_label"])
                loss1 = ocsoftmaxloss * self.loss.weight_loss
                self.audio_optimizer.zero_grad()
                self.loss.audio_loss_optimizer.zero_grad()
                batch["ocsoftmax"] = ocsoftmaxloss
                loss1.backward()
                self.audio_optimizer.step()
                self.loss.audio_loss_optimizer.step()
        else:
            feats, _ = self.model.audio_model(batch["audio"].squeeze(1))
            batch["ocsoftmax"], batch["scores"] = self.loss.audio_loss(feats, batch["gt_label"])

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch
    
    def adjust_learning_rate(self, optimizer, epoch_num):
        lr = self.config.adjust_lr.lr * (self.config.adjust_lr.lr_decay ** (epoch_num // self.config.adjust_lr.interval))
        for param_group in optimizer.param_groups:
            if param_group.get("name", "") == 'audio_model':
                if self.config.adjust_lr.adjust_audio_optimizer:
                    param_group['lr'] = lr
            else:
                param_group['lr'] = lr

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.is_train = False
        self.model.audio_model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            
            with open(str(self.checkpoint_dir / "tmp_scores.txt"), "w") as cm_score_file:
                for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
                ):
                    batch = self.process_batch(
                        batch,
                        metrics=self.evaluation_metrics,
                    )
                    for i in range(len(batch["scores"])):
                        print(f"LA_D_{batch['audio_file_name'][i]} {batch['scores'][i]}", file=cm_score_file)
                self.writer.set_step(epoch * self.epoch_len, part)
                self._log_scalars(self.evaluation_metrics)

        min_tdcfs, eers = evaluation_API(
            str(self.checkpoint_dir / "tmp_scores.txt"),
            "LA_dev", 
            "dev", 
            "baseline/eval_package/keys/",
            False,
            "baseline/eval_package/keys/LA_dev/ASV/score.txt",
            "baseline/eval_package/keys/LA_dev/LA-c012.npy")
        min_tdcf, eer = min_tdcfs[-1][-1], eers[-1][-1]
        self._log_eer_tdcf(eer, min_tdcf)
        return self.evaluation_metrics.result(), eer, min_tdcf
