import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import os.path
import faiss
import GPUtil
import pickle
from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import CheckpointIO

from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)


from metrics import MetricsProcessor
from models.rag.rag_model_blip import RagModelForBlip
from utils.dirs import *


import logging
logger = logging.getLogger(__name__)


@register_executor
class RagBlipExecutor(BaseExecutor, MetricsProcessor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        global_config=None,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, global_config=global_config, use_data_node=use_data_node, *args, **kwargs)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        

    
    def _init_model(self, model_config): 
        """Initialize self.model

        Args:
            model_config (dict): contains key-values for model configuration
        """
        # super()._init_model(model_config) # alternatively, use the default implementation in super()._init_model()

        ModelClass = globals()[self.config.model_config.ModelClass]
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)
        self.model = ModelClass(self.config, self.prepared_data)
        
        
    def prepare_data(self):
        super().prepare_data()
        
    
    def setup(self, stage):
        super().setup(stage)
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)
        
        print(len(self.prepared_data.vqa_data.get('lookup', {})))
        if len(self.prepared_data.vqa_data.get('lookup', {})) == 0:
            self.prepared_data.vqa_data.lookup = {}
            print("Loading lookup table...")
            for data_split in ['train', 'valid', 'test']:
                if data_split not in self.prepared_data.vqa_data:
                    continue
                ds_split = self.prepared_data.vqa_data[data_split]
                lookup_dict = ds_split.to_pandas().set_index("question_id", drop=False).to_dict(orient="index")
                self.prepared_data.vqa_data.lookup.update(lookup_dict)
            # print(f"Rank {torch.distributed.get_rank()} Done loading lookup table.")
        
        logger.info(f"Preparing passages...")

        test_ds = self.prepared_data.valid_passages if self.config.mode == 'train' else self.prepared_data.test_passages
        self.prepared_data.passages = EasyDict({
            'dataset': test_ds,
            'id2doc': {},
        })
        
        # find duplicated rows
        df = test_ds.to_pandas()
        duplicated_rows = df[df.duplicated(subset=['passage_id'], keep=False)]
        print("find duplicates:", duplicated_rows)
        logger.error(f"Find {len(duplicated_rows)} duplicated rows in the passages dataset.")
        
        self.prepared_data.passages.id2doc = df.drop_duplicates(subset=["passage_id"]).set_index("passage_id").to_dict(orient="index")
        for k, v in tqdm(self.prepared_data.passages.id2doc.items(), desc="formatting the passages"):
            self.prepared_data.passages.id2doc[k] = v['passage_content']

        self.model.passage_id2doc = self.prepared_data.passages.id2doc
        logger.info(f"passages prepared.")
        
        self.data_loaders = self.prepared_data['data_loaders']

        self.train_dataloaders = list(self.data_loaders['train'].values())
        self.valid_dataloaders = list(self.data_loaders['valid'].values())
        self.test_dataloaders = list(self.data_loaders['test'].values())

        self.tokenizers = self.prepared_data['tokenizers']

        self.tokenizer = self.tokenizers['tokenizer']
        self.decoder_tokenizer = self.tokenizers['decoder_tokenizer']

        checkpoint_to_load = self.global_config.train.get('load_model_path', '')

        if not checkpoint_to_load or checkpoint_to_load == '':
            logger.warning("No checkpoint found. First time to train...")
        else:
            # We manually load the state dict
            logger.info(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)['state_dict']
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict_from_ckpt.items() if k in model_dict}
            logger.info(f"Load the following parameters from the given checkpoint: {pretrained_dict.keys()}")
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)
        
        
        

    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """
        optimizer_name = self.optimizer_config['optimizer_name']
        optimizer_params = self.optimizer_config.get('optimizer_params', {})
        retriever_lr = self.optimizer_config.get('retriever_lr', 1e-4)

        optimization_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()],
                'lr': optimizer_params.lr,
                'initial_lr': optimizer_params.lr,
            },
        ]
        
        
        def get_parameter_names(model, forbidden_layer_types):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(model._parameters.keys())
            return result
        

        if 'freeze_question_encoder' in self.config.model_config.modules:
            # Freeze retriever
            print('freeze retriever!')
            for n, p in self.model.named_parameters():
                if 'generator' not in n:
                    p.requires_grad = False
        
        weight_decay = self.config.train.get('weight_decay', 0)
        if weight_decay == 0:
            optimization_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() if 'generator' in n and p.requires_grad],
                    'lr': optimizer_params.lr,
                    'initial_lr': optimizer_params.lr,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if 'generator' not in n and p.requires_grad],
                    'lr': retriever_lr,
                    'initial_lr': retriever_lr,
                },
            ]
        else:
            # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimization_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'generator' in n and p.requires_grad],
                    "weight_decay": weight_decay,
                    'lr': optimizer_params.lr,
                    'initial_lr': optimizer_params.lr,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if n in decay_parameters and 'generator' not in n and p.requires_grad],
                    'lr': retriever_lr,
                    'initial_lr': retriever_lr,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and 'generator' in n and p.requires_grad],
                    "weight_decay": 0.0,
                    'lr': optimizer_params.lr,
                    'initial_lr': optimizer_params.lr,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if n not in decay_parameters and 'generator' not in n and p.requires_grad],
                    'lr': retriever_lr,
                    'initial_lr': retriever_lr,
                },
            ]

        for group in optimization_parameters:
            logger.info('#params: {}   lr: {}'.format(len(group['params']), group['lr']))
        

        """define optimizer"""
        
        if optimizer_name == 'AdamW':
            from transformers import AdamW
            self.optimizer = AdamW(optimization_parameters, **optimizer_params)
        elif optimizer_name == 'Adafactor':
            from transformers import Adafactor
            self.optimizer = Adafactor(optimization_parameters, **optimizer_params)
        elif optimizer_name == 'Adam':
            from torch.optim import Adam
            self.optimizer = Adam(optimization_parameters, **optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {optimizer_name}")
        
        num_warmup_steps = self.optimizer_config.get('scheduler_params', {}).get('num_warmup_steps', 0)
        if self.optimizer_config.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.optimizer_config.scheduler == 'cosine':
            t_total = self.training_config.trainer_paras.max_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                            t_total, eta_min=1e-5, last_epoch=-1, verbose=False)
        else:
            from transformers import get_constant_schedule_with_warmup
            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=self.global_step,
            )
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                # REQUIRED: The scheduler instance
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
        }

    def training_step(self, sample_batched, batch_idx):
        train_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            'labels': sample_batched['labels'].to(self.device),
            'input_text_sequences': sample_batched['input_text_sequences'],
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'training': True,
        })

        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            train_batch['pixel_values'] = pixel_values.to(self.device)
        
        decoder_pixel_values = sample_batched.get('decoder_pixel_values', None)
        if decoder_pixel_values is not None:
            train_batch['decoder_pixel_values'] = decoder_pixel_values.to(self.device)
        
        
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            train_batch['image_features'] = image_features.to(self.device)
        
        pos_item_ids = sample_batched.get('pos_item_ids', None)
        if pos_item_ids is not None:
            train_batch['pos_item_ids'] = pos_item_ids
        
        forward_results = self.model(**train_batch)
        batch_loss = forward_results.loss

        loss_dict = forward_results.loss_dict
        for loss_name, loss_value in loss_dict.items():
            # print(f"loss_name: {loss_name}, loss_value: {loss_value}")
            self.log(f"train/{loss_name}", loss_value.to(dtype=torch.float32), on_step=True, logger=True, sync_dist=True)

        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(f"train/lr[{index}]", current_lr, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", batch_loss.to(dtype=torch.float32), prog_bar=True, on_step=True, logger=True, sync_dist=True)

        data_to_return = {
            'loss': batch_loss,
        }
        return data_to_return

    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        # print(f'batch_idx {batch_idx}  dataloader_idx {dataloader_idx}')
        pred = self._generative_step(sample_batched, batch_idx)
        self.validation_step_outputs.append(pred)
        return pred

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        if len(validation_step_outputs) == 0:
            return None
        for i in range(len(self.val_dataloader())):
            if len(self.val_dataloader()) == 1:
                validation_step_output = validation_step_outputs
            else:
                validation_step_output = validation_step_outputs[i]
            
            log_dict = self.evaluate_outputs(validation_step_output)
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        
        self.validation_step_outputs.clear()

        return None

    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred = self._generative_step(sample_batched, batch_idx)
        self.test_step_outputs.append(pred)
        return pred

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        logger.info("reading global step of the checkpoint...")
        if self.trainer.ckpt_path is not None:
            self.ckpt_global_step = torch.load(self.trainer.ckpt_path, map_location=torch.device('cpu'))['global_step']
        elif self.global_config.train.get('load_model_path', '') != "":
            self.ckpt_global_step = torch.load(self.global_config.train.load_model_path, map_location=torch.device('cpu'))['global_step']
        else:
            self.ckpt_global_step = self.global_step
        
        for i in range(len(self.test_dataloader())):
            if len(self.test_dataloader()) == 1:
                test_step_output = test_step_outputs
            else:
                test_step_output = test_step_outputs[i]
            
            log_dict = self.evaluate_outputs(test_step_output)
            self.logging_results(log_dict, prefix=f"{self.config.test_suffix}_{self.test_dataloader_names[i]}")
        
        self.test_step_outputs.clear()
        return None

    def _generative_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        predictions = []
        table_entries = []

        labels = sample_batched['labels']
        test_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            'labels': sample_batched['labels'].to(self.device),
            'input_text_sequences': sample_batched['input_text_sequences'],
            'question_ids': sample_batched['question_ids'],
        })

        # add pixel values
        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            test_batch['pixel_values'] = pixel_values.to(self.device)
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            test_batch['image_features'] = image_features.to(self.device)
        decoder_pixel_values = sample_batched.get('decoder_pixel_values', None)
        if decoder_pixel_values is not None:
            test_batch['decoder_pixel_values'] = decoder_pixel_values.to(self.device)
        

        generation_outputs = self.model.generate(**test_batch)
        outputs = generation_outputs.outputs
        retrieved_docs = generation_outputs.retrieved_docs
        generation_outputs_for_docs = generation_outputs.generation_outputs_for_docs
        loss_with_doc_scores = generation_outputs.loss_with_doc_scores
        

        bos_token_id = self.decoder_tokenizer.tokenizer.bos_token_id
        for index, i in enumerate(labels):

            cleaned_i = [label if label!=-100 else self.decoder_tokenizer.tokenizer.pad_token_id for label in i]
            cleaned_i = torch.LongTensor(cleaned_i)
            decoded_label = self.decoder_tokenizer.tokenizer.decode(cleaned_i, skip_special_tokens=True)
            # print(self.tokenizer.decode(cleaned_i, skip_special_tokens=True))
            
            output_sequence = outputs[index]
            # print('output_sequence', output_sequence)
            cleaned_i = [label if label!=-100 else self.decoder_tokenizer.tokenizer.pad_token_id for label in output_sequence]
            cleaned_i = torch.LongTensor(cleaned_i)

            output_sequence = output_sequence.cpu().numpy().tolist()

            if bos_token_id in output_sequence:
                output_sequence = output_sequence[output_sequence.index(bos_token_id):]

            # print('output_sequence after', output_sequence)
            decoded_output = self.decoder_tokenizer.tokenizer.decode(output_sequence, skip_special_tokens=True)
            actual_output = self.decoder_tokenizer.tokenizer.decode(output_sequence, skip_special_tokens=False)
            # print(self.tokenizer.decode(cleaned_i, skip_special_tokens=True))
            
            if batch_idx < 10:
                print(decoded_label, '<--->', decoded_output, '   ({})'.format(actual_output))
            
            question_id = sample_batched['question_ids'][index]
            predictions.append({
                'question_id': question_id,
                'answer': decoded_output,
            })

            item = self.prepared_data.vqa_data.lookup[str(question_id)]
            table_entry = [
                question_id,
                item['img_path'],
                item['question'],
                # item['img_caption']['caption'],
                str(item['answers']),
                item['gold_answer'],
                decoded_output,
                generation_outputs_for_docs[index],
            ]
            
            for doc, doc_prediction in zip(retrieved_docs[index], generation_outputs_for_docs[index]):
                table_entry+=['[' + doc['passage_id'] + ']:' + doc['content'], doc_prediction]
            table_entries.append(table_entry)

        data_to_return = {
            'predictions': predictions,
            'outputs': outputs,
            'retrieved_docs': retrieved_docs,
            'generation_outputs_for_docs': generation_outputs_for_docs,
            'loss_with_doc_scores': loss_with_doc_scores,
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'numeric_ranges': sample_batched.get('wikidata_ranges', None),
            'table_entries': table_entries,
        }

        return data_to_return
    
    


    def evaluate_outputs(self, step_outputs, mode='test'):
        # Batching every validation step outputs
        batch_predictions = []
        batch_answers = []
        batch_retrieved_docs = []
        batch_generation_outputs_for_docs = []
        batch_loss_with_doc_scores = []
        batch_question_ids = []
        batch_numeric_ranges = []

        n_docs = self.config.model_config.num_knowledge_passages
        columns=["question_id", "img_path", "question", "answers", "gold_answer", "prediction", "doc_predictions"]
        for i in range(n_docs):
            columns += ['p_{}'.format(i), 'a_{}'.format(i)]
        test_table = wandb.Table(columns=columns)

        for step_output in step_outputs:
            batch_predictions += step_output['predictions']
            batch_answers += step_output['answers']
            if step_output.get("numeric_ranges", None) is not None:
                batch_numeric_ranges += step_output['numeric_ranges']
            else:
                batch_numeric_ranges += [None] * len(step_output['predictions'])
            batch_retrieved_docs += step_output['retrieved_docs']
            batch_generation_outputs_for_docs += step_output['generation_outputs_for_docs']
            batch_loss_with_doc_scores.append(step_output['loss_with_doc_scores'])
            batch_question_ids += step_output['question_ids']
            for table_entry in step_output['table_entries']:
                test_table.add_data(*table_entry)
        
        # concatenate all tensors
        batch_loss_with_doc_scores = np.concatenate(batch_loss_with_doc_scores, axis=0)

        
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_question_ids=batch_question_ids,
            batch_predictions=batch_predictions,
            batch_answers=batch_answers,
            batch_retrieved_docs=batch_retrieved_docs,
            batch_generation_outputs_for_docs=batch_generation_outputs_for_docs,
            batch_loss_with_doc_scores=batch_loss_with_doc_scores,
            batch_numeric_ranges=batch_numeric_ranges,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)
        log_dict.artifacts.test_table = test_table

        return log_dict

    def logging_results(self, log_dict, prefix='test'):
        
        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            if metric in ['precision', 'recall', 'gold_precision', 'gold_recall']:
                metrics_to_log[f'{prefix}/{metric}_at_{log_dict.metrics["n_retrieved_docs"]}'] = value
            else:
                metrics_to_log[f'{prefix}/{metric}'] = value
        
        # include other artifacts / metadata
        metrics_to_log[f'{prefix}/epoch'] = self.current_epoch
        wandb_artifacts_to_log.update({
            f"predictions/step_{self.global_step}_MODE({self.config.mode})_SET({prefix})_rank({self.global_rank})": log_dict.artifacts['test_table']
        })
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        logger.info(f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}")
        
        if self.trainer.state.stage in ['sanity_check']:
            logging.warning('Sanity check mode, not saving to loggers.')
            return
        
        # Add to loggers
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(metric, float(value), logger=True, sync_dist=True)
            else:
                logger.info(f'{metric} is not a type that can be logged, skippped.')
        
        # Call wandb to log artifacts; remember to use commit=False so that the data will be logged
        #       with other metrics later.
        if self.config.args.log_prediction_tables:
            self.wandb_logger.experiment.log(wandb_artifacts_to_log, commit=False)
            local_table_path = os.path.join(self.config.test_dir, f"predictions_rank_{self.global_rank}.csv")
            df = log_dict.artifacts['test_table'].get_dataframe()
            df.to_csv(local_table_path, sep='\t')
        
        
    def forward(self, **kwargs):
        return self.model(**kwargs)