import time
import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import tqdm

from ipu.base_trainer import BaseTrainerIPU
from trainer import verbose, format_nested_metrics_for_writer
from model.model import sim_matrix
from utils import inf_loop
import poptorch
from ipu import options


class TrainerIPU(BaseTrainerIPU):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(model, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min(len(x) for x in data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.total_batch_sum = sum(x.batch_size for x in self.data_loader)
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

        # TODO--
        self.model = self.model.half()
        opts = options.get_options()
        layers_on_ipu = [0,0,1,1,1,1]
        for index, layer in enumerate(self.model.text_model.transformer.layer):
            # ipu = layer_ipu[index]
            # layer = RecomputationCheckpoint(layer) if config.recompute_checkpoint_every_layer else layer
            self.model.text_model.transformer.layer[index] = poptorch.BeginBlock(layer, f"text_encoder{index}", ipu_id=layers_on_ipu[index])
            print(f"text_encoder {index:<2} --> IPU {layers_on_ipu[index]}")
        self.model.txt_proj = poptorch.BeginBlock(self.model.txt_proj,"txt_proj",ipu_id=1)

        layers_on_ipu = [2,2,3,3,4,4,5,5,6,6,7,7]
        for index, layer in enumerate(self.model.video_model.blocks):
            # ipu = layer_ipu[index]
            # layer = RecomputationCheckpoint(layer) if config.recompute_checkpoint_every_layer else layer
            self.model.video_model.blocks[index] = poptorch.BeginBlock(layer, f"video_encoder{index}", ipu_id=layers_on_ipu[index])
            print(f"video_encoder {index:<2} --> IPU {layers_on_ipu[index]}")
        self.model.vid_proj = poptorch.BeginBlock(self.model.vid_proj,"vid_proj",ipu_id=7)

        self.model.loss = loss
        self.loss = loss
        print(self.model.loss)

        self.training_model = poptorch.trainingModel(self.model,
                                        options=opts,
                                        optimizer=optimizer)
        # Compile model
        # log.logger.info("---------- Compilation Started ---------")
        start_compile = time.perf_counter()
        datum = next(iter(self.data_loader[0]))
        text = self.tokenizer(datum['text'], return_tensors='pt', padding=True, truncation=True)
        datum = {'input_ids':text['input_ids'], 'attention_mask':text['attention_mask'], 'video':datum['video']}
        self.training_model.compile(**datum)
        duration_compilation = time.perf_counter() - start_compile
        # log.logger.info(f"Compiled model in {duration_compilation} secs")
        print((f"Compiled training model in {duration_compilation} secs"))
        # log.logger.info("---------------------------------------")
        
        inf_opts = poptorch.Options()
        inf_opts.deviceIterations(4)
        # self.inference_model = self.model.eval()
        self.inference_model = poptorch.inferenceModel(self.model.eval(),options=inf_opts)
        # # Compile inference_model
        # # log.logger.info("---------- Compilation Started ---------")
        # start_compile = time.perf_counter()
        # datum = next(iter(self.valid_data_loader[0]))
        # text = self.tokenizer(datum['text'], return_tensors='pt', padding=True, truncation=True)
        # datum = {'input_ids':text['input_ids'], 'attention_mask':text['attention_mask'], 'video':datum['video']}
        # self.inference_model.compile(**datum)
        # duration_compilation = time.perf_counter() - start_compile
        # # log.logger.info(f"Compiled model in {duration_compilation} secs")
        # print((f"Compiled inference model in {duration_compilation} secs"))
        # # log.logger.info("---------------------------------------")

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_iterations = self.max_samples_per_epoch // self.total_batch_sum + 1
        with tqdm(zip(*self.data_loader), desc=f"Training epoch {epoch}", total=total_iterations) as progress:
            for batch_idx, data_li in enumerate(progress):
                if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                    break
                for dl_idx, data in enumerate(data_li):
                    # then assume we must tokenize the input, e.g. its a string
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                      truncation=True)

                    # self.optimizer.zero_grad()
                    # text_embeds, video_embeds = self.model(data)
                    _, loss = self.training_model(data['text']['input_ids'], data['text']['attention_mask'], data['video'])
                    
                    # loss.backward()
                    # self.optimizer.step()

                    detached_loss = loss.detach().item()

                    if self.writer is not None:
                        self.writer.log_scalar(f'loss_train_{dl_idx}', detached_loss)

                    total_loss[dl_idx] += detached_loss

                    progress.set_postfix({"dl": dl_idx, "loss": detached_loss})

                    self.optimizer.zero_grad()

                if batch_idx == self.len_epoch:
                    break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        self.inference_model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                # for video, text, meta in tqdm(dl, desc=f"Validating dl{dl_idx}"):
                for data in tqdm(dl, desc=f"Validating dl{dl_idx}"):
                    # meta_arr[dl_idx].append(meta)
                    meta_arr[dl_idx].append(data['meta'])

                    if self.tokenizer is not None:
                        text = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    results = self.inference_model(text['input_ids'], text['attention_mask'], data['video'])
                    text_embed, vid_embed = results
                    text_embed_arr[dl_idx].append(text_embed.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed.cpu())
                    sims_batch = sim_matrix(text_embed, vid_embed)
                    loss = self.loss(sims_batch)
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                        mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


