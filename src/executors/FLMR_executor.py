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
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)


from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import CheckpointIO
from transformers import AdamW, Adafactor, get_scheduler
from torch.optim import Adam

# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from functools import partial

from colbert.data import Queries
from colbert import Searcher

from metrics import MetricsProcessor
from models.retriever.FLMR import *
from models.retriever.retriever_dpr import RetrieverDPR
from utils.dirs import *
import faiss
import wandb
import GPUtil
import pickle

import logging
logger = logging.getLogger(__name__)

import datasets

def get_world_size():
    return dist.get_world_size()

@register_executor
class FLMRExecutor(BaseExecutor, MetricsProcessor):
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
        only_encode_item_once=False,
        load_only_vision_projection_weights=False,
        perform_zero_shot_eval=False,
        validation_indexing_source=None,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, global_config=global_config, use_data_node=use_data_node, *args, **kwargs)
        
        self.tmp_index = defaultdict(None)
        self.multimodal_docs = model_config.get("multimodal_docs", False)
        if self.multimodal_docs:
            from models.retriever.colbert_utils import MultiModalIndexer
            self.IndexerClass = MultiModalIndexer
        else:
            from colbert import Indexer
            self.IndexerClass = Indexer
        
        # When this flag is set to True, we only encode the item once in non- sanity check mode
        self.only_encode_item_once = only_encode_item_once

        # When this flag is set to true, only parameters in vision_projection will be loaded
        self.load_only_vision_projection_weights = load_only_vision_projection_weights

        # When this flag is set to true, we will perform zero-shot evaluation at sanity check
        self.perform_zero_shot_eval = perform_zero_shot_eval
        
        # When a list of names are provided, the indexing process will be run multiple times
        # this allows for evaluating the validation sets on different corpora
        self.validation_indexing_source = validation_indexing_source

        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

    
    def _init_model(self, model_config): 
        """Initialize self.model

        Args:
            model_config (dict): contains key-values for model configuration
        """
        # super()._init_model(model_config) # alternatively, use the default implementation in super()._init_model()

        ModelClass = globals()[self.model_config.ModelClass]

        colbert_config = ColBERTConfig(
            bsize=None,
            use_ib_negatives=True,
            checkpoint=self.model_config.EncoderModelVersion,
            rank=self.global_rank,
        )
        colbert_config.checkpoint = colbert_config.checkpoint or 'bert-base-uncased'
        if self.model_config.ModelClass == 'ColBERT':
            self.model = ModelClass(name=colbert_config.checkpoint, colbert_config=colbert_config)
        else:
            self.model = ModelClass(name=colbert_config.checkpoint, colbert_config=colbert_config, global_config=self.global_config)
        
        self.model.colbert_config.nway = self.config.model_config.num_negative_samples + 1
        
    
    def prepare_data(self):
        super().prepare_data()
        
    
    def setup(self, stage):
        super().setup(stage)
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)
        
        print(len(self.prepared_data.vqa_data_with_dpr_output.get('lookup', {})))
        if len(self.prepared_data.vqa_data_with_dpr_output.get('lookup', {})) == 0:
            self.prepared_data.vqa_data_with_dpr_output.lookup = {}
            print("Loading lookup table...")
            for data_split in ['train', 'valid', 'test']:
                if data_split not in self.prepared_data.vqa_data_with_dpr_output:
                    continue
                ds_split = self.prepared_data.vqa_data_with_dpr_output[data_split]
                lookup_dict = ds_split.to_pandas().set_index("question_id", drop=False).to_dict(orient="index")
                self.prepared_data.vqa_data_with_dpr_output.lookup.update(lookup_dict)
            print(f"Rank {torch.distributed.get_rank()} Done loading lookup table.")

        # if isinstance(self.prepared_data.train_passages, datasets.Dataset):
        # ds = self.prepared_data.train_passages
        test_ds = self.prepared_data.valid_passages if self.config.mode == 'train' else self.prepared_data.test_passages
        self.prepared_data.passages = EasyDict({
            'dataset': test_ds,
            'id2doc': {},
        })
        
        if self.validation_indexing_source is not None:
            for name in self.validation_indexing_source:
                self.prepared_data[name] = EasyDict({
                    'id2doc': {},
                })
        
        logger.info(f"Preparing {len(test_ds)} passage data in id2doc...")
        test_df = test_ds.to_pandas()
        for _, entry_data in tqdm(test_df.iterrows(), total=len(test_ds), desc="formatting the test passages"):
            k = entry_data['passage_id']
            v = entry_data
            self.prepared_data.passages.id2doc[k] = v['passage_content']
            if self.validation_indexing_source is not None:
                source_name = v['source_name']
                if source_name in self.validation_indexing_source:
                    self.prepared_data[source_name].id2doc[k] = v['passage_content']
        
        if self.validation_indexing_source is not None:
            for name in self.validation_indexing_source:
                logger.info(f"passages from the source {name} has {len(self.prepared_data[name].id2doc)}")
        
        logger.info(f"Passages prepared.")

        self.data_loaders = self.prepared_data['data_loaders']

        self.train_dataloaders = list(self.data_loaders['train'].values())
        self.valid_dataloaders = list(self.data_loaders['valid'].values())
        self.test_dataloaders = list(self.data_loaders['test'].values())

        self.tokenizers = self.prepared_data['tokenizers']

        self.tokenizer = self.tokenizers['tokenizer']
        self.decoder_tokenizer = self.tokenizers['decoder_tokenizer']

        self.model.raw_tokenizer = self.decoder_tokenizer.tok

        checkpoint_to_load = self.global_config.train.get('load_model_path', '')

        # Resize the bert embedding space to accommodate special tokens
        logger.info(f'tokenizer lengths = {len(self.tokenizer.tok)} and {len(self.decoder_tokenizer.tok)}')
        self.model.bert.resize_token_embeddings(
            max(len(self.tokenizer.tok), len(self.decoder_tokenizer.tok))
        )

        print(self.model.bert.embeddings.word_embeddings.weight.shape)

        if not checkpoint_to_load or checkpoint_to_load == '':
            logger.warning("No checkpoint found. First time to train...")
        else:
            # We manually load the state dict
            logger.info(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)['state_dict']
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            if self.load_only_vision_projection_weights:
                pretrained_dict = {k: v for k, v in state_dict_from_ckpt.items() if k in model_dict and "vision_projection" in k}
            else:
                pretrained_dict = {k: v for k, v in state_dict_from_ckpt.items()}
            logger.info(f"Load the following parameters from the given checkpoint: {pretrained_dict.keys()}")
            
            if "separate_question_encoder" in self.model_config.modules:
                print("===========================================")
                copied_params = {k.replace("model.bert", "query_encoder"): v for k, v in pretrained_dict.items() if "model.bert" in k}
                copied_params.update({k.replace("model.linear", "query_linear"): v for k, v in pretrained_dict.items() if "model.linear" in k})
                
                FOUND_SAME_KEYS = False
                for k in copied_params.keys():
                    if k in pretrained_dict:
                        logger.error(f"same keys found: {k}")
                        FOUND_SAME_KEYS = True
                
                if FOUND_SAME_KEYS:
                    logger.warning("found same keys - stop copying params from document encoder to query encoder.")
                else:
                    print("copied_params", copied_params.keys())
                    pretrained_dict.update(copied_params)
                
            if "enable_doc_encoder_lora" in self.model_config.modules:
                # pprint(model_dict.keys())
                # input()
                # print("===========================================")
                # pprint(pretrained_dict.keys())
                # input()
                new_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    if k.startswith("model.model.bert."):
                        new_pretrained_dict[k.replace("model.model.bert.", "model.model.bert.base_model.model.")] = v
                        # model.query_encoder.base_model.model.
                        # model.query_encoder.pooler.dense.weight
                    elif k.startswith("model.query_encoder.") and "separate_question_encoder" in self.model_config.modules:
                        # logger.warning(f'LoRA weights: {k.replace("model.model.bert.", "model.query_encoder.base_model.model.")}')
                        new_pretrained_dict[k.replace("model.query_encoder.", "model.query_encoder.base_model.model.")] = v
                    else:
                        new_pretrained_dict[k] = v
                pretrained_dict = new_pretrained_dict
                logger.info("reformatted pretrained weights to accommodate LoRA.")
                
            logger.info(f"Loading the following parameters into the current model: {pretrained_dict.keys()}")
            
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            
            # When multimodal docs are enabled, load the vision_projection into doc_vision_projection
            if self.multimodal_docs:
                doc_vision_projection_dict = {k.replace('vision_projection', 'doc_vision_projection'): v for k, v in state_dict_from_ckpt.items() if 'vision_projection' in k}
                model_dict.update(doc_vision_projection_dict)
                logger.info(f"Load the following parameters from the given checkpoint: {doc_vision_projection_dict.keys()}")
            
            # 3. load the new state dict
            self.load_state_dict(model_dict, strict=False)
        
        

        self.model.skiplist.update({w: True
                             for symbol in self.tokenizer.tok.additional_special_tokens
                             for w in [symbol, self.model.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]})
        pprint(self.model.skiplist)

    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """
        optimizer_name = self.optimizer_config['optimizer_name']
        optimizer_params = self.optimizer_config.get('optimizer_params', {})

        optimization_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if "vision_projection" not in n],
                'lr': optimizer_params.lr,
                'initial_lr': optimizer_params.lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if "vision_projection" in n],
                'lr': self.optimizer_config.get("mapping_network_lr", optimizer_params.lr),
                'initial_lr': self.optimizer_config.get("mapping_network_lr", optimizer_params.lr),
            },
        ]
        
        for group in optimization_parameters:
            logger.info('#params: {}   lr: {}'.format(len(group['params']), group['lr']))
            
        """define optimizer"""
        
        if optimizer_name == 'AdamW':
            self.optimizer = AdamW(optimization_parameters, **optimizer_params)
        elif optimizer_name == 'Adafactor':
            self.optimizer = Adafactor(optimization_parameters, **optimizer_params)
        elif optimizer_name == 'Adam':
            self.optimizer = Adam(optimization_parameters, **optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {optimizer_name}")
        
        num_warmup_steps = self.optimizer_config.get('scheduler_params', {}).get('num_warmup_steps', 0)
        if self.optimizer_config.get('scheduler', None) == 'linear':
            from transformers import get_linear_schedule_with_warmup
            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.optimizer_config.get('scheduler', None) == 'cosine':
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
        train_batch = {
            'Q': [
                sample_batched['input_ids'].to(self.device),
                sample_batched['attention_mask'].to(self.device)
            ],
            "D": [
                sample_batched['decoder_input_ids'].to(self.device),
                sample_batched['decoder_input_attention_mask'].to(self.device)
            ],
            # 'labels': sample_batched['labels'].to(self.device),
        }
        
        # if there is vision input, add it to the batch
        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            train_batch['Q'].append(pixel_values.to(self.device))
        
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            train_batch['Q'].append(image_features.to(self.device))
        item_image_features = sample_batched.get('item_image_features', None)
        if item_image_features is not None:
            train_batch['D'].append(item_image_features.to(self.device))
        
        scores = self.model(**train_batch)
        
        config = self.model.colbert_config
        if config.use_ib_negatives:
            scores, ib_loss = scores
            loss = ib_loss
        else:
            scores = scores.view(-1, config.nway)
            labels = torch.zeros(sample_batched['input_ids'].shape[0]*dist.get_world_size(), dtype=torch.long, device=self.device)
            loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
        
        batch_loss = loss
        
        # print("final scores", scores.shape, scores)
        # input()

        # if unwrap_model(self.model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #     batch_loss = self.label_smoother(forward_results, train_batch.labels, shift_labels=True)
        # else:
        #     batch_loss = self.label_smoother(forward_results, train_batch.labels)
        
        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(f"train/lr[{index}]", current_lr, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", batch_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train/ib_loss", ib_loss, on_step=True, logger=True, sync_dist=True)
        
        data_to_return = {
            'loss': batch_loss,
        }
        return data_to_return
    
    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred = self._compute_query_embeddings_step(sample_batched, batch_idx)
        self.validation_step_outputs[dataloader_idx].append(pred)
        return pred

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        if len(validation_step_outputs) == 0:
            return None
        for i in range(len(self.val_dataloader())):
            validation_step_output = validation_step_outputs[i]
            
            log_dict = self.evaluate_outputs(validation_step_output, self.val_dataloader()[i], self.val_dataloader_names[i], dataloader_idx=i)
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        
        # if "freeze_colbert_doc_encoder" not in self.model_config.modules:
        #     # when validation finishes, remove tmp index
        #     self.tmp_index = None
        self.tmp_index = defaultdict(None)

        self.validation_step_outputs = defaultdict(list)

        return None
    
    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred = self._compute_query_embeddings_step(sample_batched, batch_idx)
        self.test_step_outputs[dataloader_idx].append(pred)
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

        self.save_HF_model()
        for i in range(len(self.test_dataloader())):
            test_step_output = test_step_outputs[i]
            log_dict = self.evaluate_outputs(test_step_output, self.test_dataloader()[i], self.test_dataloader_names[i], dataloader_idx=i)
            self.logging_results(log_dict, prefix=f"{self.config.test_suffix}_{self.test_dataloader_names[i]}")
        
        # when testing finishes, remove tmp index
        self.tmp_index = defaultdict(None)

        self.test_step_outputs = defaultdict(list)

        return None

        

    def _compute_query_embeddings_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        test_batch = [
            sample_batched['input_ids'].to(self.device),
            sample_batched['attention_mask'].to(self.device)
        ]
        # if there is vision input, add it to the batch
        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            test_batch.append(pixel_values.to(self.device))
        
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            test_batch.append(image_features.to(self.device))

        # batch_size x hidden_states
        query_emb = self.model.query(*test_batch)

        data_to_return = {
            'btach_idx': batch_idx,
            'query_emb': query_emb.cpu(),
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'pos_item_ids': sample_batched['pos_item_ids'],
            'questions': sample_batched['questions'],
        }

        return data_to_return
    
    
    def prepare_item_embeddings(self, current_data_loader, dataloader_idx, mode):
        
        use_index = getattr(self, 'use_index', None)

        passage_id2doc = self.prepared_data.passages.id2doc
        if self.validation_indexing_source is not None:
            use_source_name = self.validation_indexing_source[dataloader_idx]
            passage_id2doc = self.prepared_data[use_source_name].id2doc
            logger.info(f"Using {use_source_name} as validation/test corpus...")
        
        
        n_items = len(passage_id2doc)

        if self.trainer.state.stage in ['sanity_check'] and not self.perform_zero_shot_eval:
            # sanity check
            if use_index is None:
                logging.warning('No steps have been taken. Reducing number of items to speed up the sanity check.')
                n_items = 100
            else:
                logging.warning('Using custom index files - #items will not be reduced.')
        
        i_batch_size = self.config[mode].batch_size
        
        n_item_batchs = n_items // i_batch_size + 1

        # Create mapping between matrix indice and passage ids
        # Using only train passage corpus
        passage_index2id = {index:passage_id for index, passage_id in enumerate(passage_id2doc.keys()) if index < n_items}
        decoder_input_modules = self.model_config.decoder_input_modules.module_list
        passage_contents = []

        multimodal_docs = self.model_config.get('multimodal_docs', False)
        if multimodal_docs:
            passage_image_features = []

        for passage_id in passage_index2id.values():
            sample = EasyDict(passage_content=passage_id2doc[passage_id], passage_id=passage_id)
            parsed_data = current_data_loader.dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
            passage_contents.append(parsed_data.text_sequence)
            if multimodal_docs:
                passage_image_features.append(parsed_data.img.image_features)

        if use_index is None:
            # When no index file is specified
        
            logger.info(f'Generating embeddings for items; there are {n_items} items.')
            exhaustive_search_in_testing = "exhaustive_search_in_testing" in self.config.model_config.modules
            # Move model to cpu to save some memory
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            
            print(f"rank {self.global_rank} moved to cpu.")
            # GPUtil.showUtilization()

            if self.global_rank == 0 and not exhaustive_search_in_testing:
                logger.info(f"Global rank {self.global_rank} starts indexing job...")

                # First, we need to save the model checkpoint so that we can run the indexing process
                # Currently the ColBERT's engine does not provide on-the-fly indexer
                if self.trainer.state.stage == 'test':
                    tmp_model_path = os.path.join(
                        self.config.ckpt_dir,
                        f"test_temp_model",
                    )
                else:
                    tmp_model_path = os.path.join(
                        self.config.ckpt_dir,
                        f"validation_temp_model",
                    )
                logger.info(f"saving temp model to {tmp_model_path}...")
                if "enable_doc_encoder_lora" in self.config.model_config.modules:
                    # When using lora, we need to unload the model first
                    merged_model = copy.deepcopy(self.model)
                    merged_model.model.bert = merged_model.model.bert.merge_and_unload()
                    logger.info("LoRA unloaded and merged.")
                    merged_model.save(tmp_model_path)
                else:
                    self.model.save(tmp_model_path)
                
                # save model_config to tmp_model_path
                with open(os.path.join(tmp_model_path, "global_config.json"), "w") as f:
                    json.dump(self.global_config, f, indent=4)
                logger.info(f"global_config saved to {tmp_model_path}.")

                
                # Launch indexer
                with Run().context(RunConfig(nranks=get_world_size(), root=self.config.ckpt_dir, experiment=f"temp_index_{dataloader_idx}")):
                    nbits = self.model_config.get("nbits", 2)
                    config = ColBERTConfig(
                        nbits=nbits,
                        doc_maxlen=self.model_config.max_decoder_source_length,
                    )
                    print("indexing with", nbits, "bits")
                    if multimodal_docs:
                        custom_collection = [
                            (passage_content, passage_image_feature) for passage_content, passage_image_feature in zip(passage_contents, passage_image_features)
                        ]
                    else:
                        custom_collection = passage_contents
                    indexer = self.IndexerClass(checkpoint=tmp_model_path, config=config)
                    indexer.index(name=f"temp_index.nbits={nbits}", collection=custom_collection, overwrite=True)
                    index_path = indexer.get_index()
                    del indexer
            else:
                logger.info(f"Global rank {self.global_rank} waits for Rank 0...")
            
            # Use barrrier to sync all ranks. Only when Rank 0 finishes indexing, other ranks will move on
            torch.distributed.barrier()
            torch.cuda.empty_cache()

            self.model = self.model.to(self.device)

            ##### Restore the following part if you want to use joint training (not static retrieval) in RAG #####
            # if (self.trainer.state.stage == 'test' and self.global_rank==0) or (exhaustive_search_in_testing):
            #     logger.info("Since the embeddings of items are required in RAG when retriever is trained jointly, we simply rerun the embedding generation to save all item embeddings into files. \n However, if the retriever is kept frozen in RAG, this part can be commented out to save time.")
            #     # Add embeddings
            #     # For easier implementation, we just rerun the item embedding generation here
            #     i_count = 0
            #     item_embeddings = []
            #     item_embedding_mask = []
            #     for i_batch_id in tqdm(range(n_item_batchs)):
            #         i_start = i_batch_id * i_batch_size
            #         i_end = min((i_batch_id + 1) * i_batch_size, n_items)
            #         if i_end - i_start == 0:
            #             break
            #         passage_contents_batch = passage_contents[i_start:i_end]
                    
            #         # Encode this batch of data
            #         item_input_ids, item_attention_mask = self.decoder_tokenizer.tensorize(passage_contents_batch)
                
            #         test_batch = (
            #             item_input_ids.to(self.device),
            #             item_attention_mask.to(self.device),
            #         )

            #         # batch_size x hidden_states
            #         item_emb, item_emb_mask = self.model.doc(*test_batch, keep_dims='return_mask')
                    
            #         for x in item_emb:
            #             item_embeddings.append(x.cpu().detach().numpy())
            #         for x in item_emb_mask:
            #             item_embedding_mask.append(x.cpu().detach().numpy())
            #         # n_queries x batch_size
            #         # i_rate_batch = torch.matmul(query_embeddings, item_emb.t()).detach().cpu()

            #         # rate_batch[:, i_start:i_end] = i_rate_batch
            #         i_count += item_emb.shape[0]

            #     assert i_count == n_items
            #     logger.info("finished generating item embeddings...")

            #     # table_dataset = table_dataset.add_column("embeddings", item_embeddings)
            #     # table_dataset = table_dataset.add_column("embedding_mask", item_embedding_mask)

            #     if self.trainer.state.stage == 'test':
            #         # Save the dataset
            #         save_path = os.path.join(self.config.test_dir, 'step_{}'.format(self.global_step))
            #         create_dirs([save_path])

            #         embedding_save_path = os.path.join(save_path, "item_embeddings.pkl")
            #         logger.info(f"saving embedding files to {embedding_save_path}")
            #         with open(embedding_save_path, 'wb') as f:
            #             pickle.dump({
            #                 "item_embeddings": item_embeddings,
            #                 "item_embedding_mask": item_embedding_mask,
            #                 "passage_index2id": passage_index2id,
            #             }, f)
                
            #     self.item_embeddings = item_embeddings
            #     self.item_embedding_mask = item_embedding_mask
        

        # Sync all processes. If rank 0 starts saving item embeddings in testing, other processes will wait for it.
        torch.distributed.barrier()

        self.tmp_index[dataloader_idx] = {
            "passage_index2id": passage_index2id,
            "passage_contents": passage_contents,
        }

        # in testing mode, copy the temp index files
        if self.trainer.state.stage == 'test' and self.global_rank==0 and not exhaustive_search_in_testing:
            save_path = os.path.join(self.config.test_dir, 'step_{}'.format(self.global_step))
            create_dirs([save_path])
            
            index_path_from = os.path.join(
                self.config.ckpt_dir, f"temp_index_{dataloader_idx}"
            )
            index_path_to = os.path.join(save_path, "colbert_index")
            logger.info(f'saving index files from {index_path_from} into {index_path_to}')
            import shutil
            if os.path.exists(index_path_to):
                # Delete if the index already exists
                shutil.rmtree(index_path_to)

            shutil.copytree(index_path_from, index_path_to)
        
        # sync all processes
        torch.distributed.barrier()

        # When this is not sanity check mode, set self.index if self.only_encode_item_once
        if self.trainer.state.stage not in ['sanity_check'] and self.only_encode_item_once:
            if dataloader_idx == len(self.val_dataloader()) - 1:
                logger.info(f"setting self.index to {self.config.ckpt_dir}... In later runs the index will not be generated again.")
                self.use_index = self.config.ckpt_dir


    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, dataloader_idx=0, mode='test'):
        # Batching every validation step outputs
        # n_queries x hidden_size
        
        query_embeddings = []
        question_ids = []
        pos_item_ids = []
        questions = []
        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])
            question_ids += step_output['question_ids']
            pos_item_ids.extend(step_output['pos_item_ids'])
            questions.extend(step_output['questions'])
        
        query_embeddings = torch.cat(query_embeddings, dim=0)
        n_queries = query_embeddings.shape[0]
        hidden_size = query_embeddings.shape[1]

        ##################################
        ##    Generate embeds for items ##
        ##################################
        
        if self.tmp_index.get(dataloader_idx, None) is None:
            if self.validation_indexing_source is None and dataloader_idx > 0:
                # When no multiple indexing source are used, we can reuse the index generated previously
                dataloader_idx = 0
                logger.info("Forcing dataloader_idx = 0: reusing pre-computed indexes...")
            else:
                # When item embeddings are not indexed, call the function
                # this will not be called more than once during a validation step
                # which reduces the time needed for validating more than one datasets
                logger.info("No tmp exists, start building indexes...")
                self.prepare_item_embeddings(current_data_loader, dataloader_idx, mode)
        else:
            logger.info("reusing pre-computed indexes...")

        passage_index2id = self.tmp_index[dataloader_idx]['passage_index2id']
        passage_contents = self.tmp_index[dataloader_idx]['passage_contents']
        

        ##################################
        ##    Search Index              ##
        ##################################

        Ks = self.model_config.Ks

        # Create mapping between matrix indice and question ids
        question_index2id = {index:question_id for index, question_id in enumerate(question_ids)}
        assert len(question_index2id) == n_queries
        logger.info(f'There are {n_queries} queries.')

        if "exhaustive_search_in_testing" not in self.model_config.modules:
            with Run().context(RunConfig(nranks=1, rank=self.global_rank, root=self.config.ckpt_dir, experiment=f"temp_index_{dataloader_idx}")):
                if self.device == torch.device('cpu'):
                    total_visible_gpus = 0
                else:
                    # TODO: with multi-GPUs, the retrieval encounters CUDA errors. Therefore, we use CPU to perform retrieval to avoid extensive work in fixing codes. Efficiency can be further improved if this can be fixed.
                    if get_world_size() > 1:
                        logger.warning("multi-GPUs are not supported in retrieval. Using CPU instead.")
                        total_visible_gpus = 0
                    else:
                        total_visible_gpus = 1 #torch.cuda.device_count()
                
                config = ColBERTConfig(
                    total_visible_gpus=total_visible_gpus,
                )
                nbits = self.model_config.get("nbits", 2)
                searcher = Searcher(index=f"temp_index.nbits={nbits}", config=config)
                custom_quries = {question_id: question for question_id, question in zip(question_ids, questions)}
                queries = Queries(data=custom_quries)
                ranking = searcher._search_all_Q(queries, query_embeddings, k=max(Ks))
                
                ranking_dict = ranking.todict()

                torch.distributed.barrier()

                del searcher
        else:
            # exhaustive search
            ranking_dict = {}
            self.model.eval()

            item_embeddings = self.item_embeddings
            item_embedding_mask = self.item_embedding_mask

            n_items = len(item_embeddings)
            logger.info(f"n_items {n_items}")

            i_batch_size = 4 #self.global_config[mode].batch_size
            n_item_batchs = n_items // i_batch_size + 1

            rate_batch = torch.zeros((len(query_embeddings), n_items))
            print("rate_batch", rate_batch.shape)
            for i_batch_id in tqdm(range(n_item_batchs)):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)
                if i_end - i_start == 0:
                    break

                retrieved_item_embeddings = np.stack(item_embeddings[i_start:i_end])
                retrieved_item_embedding_mask = np.stack(item_embedding_mask[i_start:i_end])
                retrieved_item_embeddings = torch.from_numpy(retrieved_item_embeddings).to(self.device)
                retrieved_item_embedding_mask = torch.from_numpy(retrieved_item_embedding_mask).to(self.device)
                current_i_size = len(retrieved_item_embeddings)

                # self.model.colbert_config.nway = current_i_size
                Q_duplicated = query_embeddings.repeat_interleave(current_i_size, dim=0).contiguous().to(self.device)
                retrieved_item_embeddings = retrieved_item_embeddings.repeat(len(query_embeddings), 1, 1)
                retrieved_item_embedding_mask = retrieved_item_embedding_mask.repeat(len(query_embeddings), 1, 1)
                # print("Q_duplicated", Q_duplicated.shape)
                # print("retrieved_item_embeddings", retrieved_item_embeddings.shape)
                scores = self.model.score(Q_duplicated, retrieved_item_embeddings, retrieved_item_embedding_mask)
                scores = scores.reshape(len(query_embeddings), -1)
                rate_batch[:, i_start:i_end] = scores.cpu()
            
            logger.info("sorting...")
            sorted_scores, indices = torch.sort(rate_batch.to(self.device), dim=-1, descending=True)
            sorted_scores = sorted_scores[:, :max(Ks)].cpu()
            indices = indices[:, :max(Ks)].cpu()
            for query_index in range(len(query_embeddings)):
                table_indices = indices[query_index]
                table_scores = sorted_scores[query_index]
                ranking_list = [
                    (int(table_indices[i].numpy()), i, int(table_scores[i].numpy())) for i in range(max(Ks))
                ]
                ranking_dict[query_index] = ranking_list

            # Finally, restore the nway
            # self.model.colbert_config.nway = self.config.model_config.num_negative_samples + 1
        
        batch_result = []
        for question_id, ranking_list in zip(question_ids, ranking_dict.values()):

            retrieved_doc_sorted = []
            score = []
            retrieved_doc_indices = []
            for entry in ranking_list:
                retrieved_doc_index, _, retrieved_doc_score =  entry
                retrieved_doc_indices.append(retrieved_doc_index)
                score.append(retrieved_doc_score)
            
            max_K = max(Ks)
            if len(ranking_list) < max_K:
                # normally happens in sanity check
                # the number of documents may be less than max_K
                # this is because the system relies on centroids to retrieve items
                # therefore it is not guaranteed to have enough documents retrieved
                # In this case, we simply replicate the last element to avoid crash
                retrieved_doc_indices += [retrieved_doc_indices[-1]] * (max_K-len(ranking_list))
                score += [score[-1]] * (max_K-len(ranking_list))
            
            top_ranking_passages = [{
                'passage_index': i,
                'passage_id': passage_index2id[i],
                'content': passage_contents[i],
                'score': float(score[index]),
            } for index, i in enumerate(retrieved_doc_indices)]

            query_item = self.prepared_data.vqa_data_with_dpr_output.lookup[str(question_id)]
            answers = query_item.answers
            gold_answer = query_item.gold_answer
            
            batched_data = {
                "question_id": question_id,
                "top_ranking_passages": top_ranking_passages,
                "answers": answers,
                "gold_answer": gold_answer,
            }
            if query_item.get("pos_item_ids", None) is not None:
                batched_data["pos_item_ids"] = list(query_item.pos_item_ids)
            if query_item.get("related_item_ids", None) is not None:
                batched_data["related_item_ids"] = list(query_item.related_item_ids)

            batch_result.append(batched_data)

        if self.config.args.log_prediction_tables_with_images:
            artifact = self.wandb_logger.experiment.use_artifact(self.config.args.wandb_artifacts, type='dataset')
        
        # Log results
        columns=["question_id", "input_image", "image_key", "question", "caption", "answers", "gold_answer"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)
        
        to_write_data = {
            'output': [],
        }
        for re in tqdm(batch_result):
            to_write_data['output'].append(re)
            question_id = re['question_id']
            knowledge_item = self.prepared_data.vqa_data_with_dpr_output.lookup[str(question_id)]
            if knowledge_item.get('img_file_name', None) is None:
                knowledge_item['img_file_name'] = knowledge_item['image_path'].split('/')[-1]
            if knowledge_item.get('img_key', None) is None:
                knowledge_item['img_key'] = knowledge_item['image_id']
            if knowledge_item.get('img_caption', None) is None:
                image_caption = ""
            else:
                if isinstance(knowledge_item['img_caption'], str):
                    image_caption = knowledge_item['img_caption']
                else:
                    image_caption = knowledge_item['img_caption']['caption']
                
            table_entry = [
                knowledge_item['question_id'],
                knowledge_item['img_file_name'],
                knowledge_item['img_key'],
                knowledge_item['question'],
                image_caption,
                list(knowledge_item['answers']),
                knowledge_item['gold_answer'],
            ]

            if self.config.args.log_prediction_tables_with_images:
                # Replace image keys with real images
                input_image_file_name = knowledge_item['img_file_name']
                input_image = artifact.get(input_image_file_name)
                if input_image is None:
                    input_image = artifact.get(input_image_file_name)
                
                table_entry[1] = input_image

            for p in re['top_ranking_passages']:
                answers = knowledge_item['answers']
                found = False
                for ans in answers:
                    if ans.lower() in p['content'].lower():
                        found = True
                        break
                if found:
                    table_entry.append(f'true|{p["content"]}')
                else:
                    table_entry.append(f'false|{p["content"]}')

            # table_entry+=[p['content'] for p in re['top_ranking_passages']]
            # print(table_entry)
            test_table.add_data(*table_entry)
        
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_retrieval_result=batch_result,
            Ks=Ks,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)

        log_dict.artifacts.test_table = test_table
        log_dict.artifacts.to_write_data = to_write_data
        return log_dict

    
    def logging_results(self, log_dict, prefix='test'):
        
        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        artifacts_to_log = log_dict.artifacts
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f'{prefix}/{metric}'] = value
        
        # include other artifacts / metadata
        metrics_to_log[f'{prefix}/epoch'] = self.current_epoch
        wandb_artifacts_to_log.update({
            f"predictions/step_{self.global_step}_MODE({self.config.mode})_SET({prefix})_rank({self.global_rank})": log_dict.artifacts['test_table']
        })
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        logger.info(f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}")
        
        if self.trainer.state.stage in ['sanity_check'] and not self.perform_zero_shot_eval:
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
        
        if self.config.mode == "test":
            from utils.numpy_encoder import NpEncoder
            # Save predictions to files for DPR-based VQA systems
            json_path = os.path.join(self.config.test_dir, f'{prefix.replace("/", "_")}_predictions_rank_{self.global_rank}.json')
            with open(json_path, 'w') as json_f:
                json.dump(artifacts_to_log.to_write_data, json_f, indent=4, cls=NpEncoder)
                logger.info('Predictions have been saved to {}'.format(json_path))
        

    def save_HF_model(self):
        '''
        Save models with the Huggingface built-in save_pretrained() function.
        The checkpoints can be loaded by a RAG-like system.
        '''
        if self.global_rank != 0:
            logger.info('global rank is not 0, skip saving models')
            return
        logger.info('Saving model in the Huggingface format...')
        path_save_model = os.path.join(self.config.ckpt_dir, 'step_{}'.format(self.global_step))
        self.model.save(path_save_model)
        logger.info('Model has been saved to {}'.format(path_save_model))