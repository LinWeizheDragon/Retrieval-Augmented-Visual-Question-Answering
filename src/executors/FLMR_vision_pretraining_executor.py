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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import CheckpointIO


# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from functools import partial
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher

from metrics import MetricsProcessor
from models.retriever.FLMR import *
from models.retriever.retriever_dpr import RetrieverDPR
from utils.dirs import *
import faiss
import wandb


import logging
logger = logging.getLogger(__name__)


from executors.FLMR_executor import FLMRExecutor

@register_executor
class FLMRVisionPretrainingExecutor(FLMRExecutor):
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
        
        self.use_index = kwargs.get('use_index', None)

    

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
            'questions': sample_batched['questions'],
            'passage_ids': sample_batched['passage_ids'],
            'pos_item_ids': sample_batched['pos_item_ids'],
            'neg_item_ids': sample_batched['neg_item_ids'],
        }

        return data_to_return

    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, dataloader_idx=0, mode='test'):
        # Batching every validation step outputs
        # n_queries x hidden_size
        
        query_embeddings = []
        question_ids = []
        pos_item_ids = []
        neg_item_ids = []
        questions = []
        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])
            question_ids += step_output['question_ids']
            pos_item_ids.extend(step_output['pos_item_ids'])
            neg_item_ids.extend(step_output['neg_item_ids'])
            questions.extend(step_output['questions'])
        
        query_embeddings = torch.cat(query_embeddings, dim=0)
        n_queries = query_embeddings.shape[0]
        hidden_size = query_embeddings.shape[1]

        ##################################
        ##    Generate embeds for items ##
        ##################################
        
        if self.tmp_index.get(dataloader_idx, None) is None:
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
            index_path = self.config.ckpt_dir if getattr(self, 'use_index', None) is None else self.use_index

            with Run().context(RunConfig(nranks=1, rank=self.global_rank, root=index_path, experiment=f"temp_index_{dataloader_idx}")):

                config = ColBERTConfig(
                    total_visible_gpus=0,
                )
                nbits = self.model_config.get("nbits", 2)
                searcher = Searcher(index=f"temp_index.nbits={nbits}", config=config)
                custom_quries = {question_id: question for question_id, question in zip(question_ids, questions)}
                queries = Queries(data=custom_quries)
                # print(query_embeddings.shape) # n_queries x seq_len x hidden_size
                # print(query_embeddings[0, :, :2].tolist())
                # input()
                ranking = searcher._search_all_Q(queries, query_embeddings, k=max(Ks), remove_zero_tensors=True)
                
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

            i_batch_size = self.global_config[mode].batch_size
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
                    (table_indices[i].numpy(), i, table_scores[i].numpy()) for i in range(max(Ks))
                ]
                ranking_dict[query_index] = ranking_list

            # Finally, restore the nway
            # self.model.colbert_config.nway = self.config.model_config.num_negative_samples + 1
        

        batch_result = []
        for question_id, question, ranking_list, pos_ids, neg_ids in zip(question_ids, questions, ranking_dict.values(), pos_item_ids, neg_item_ids):

            retrieved_doc_sorted = []
            score = []
            retrieved_doc_indices = []
            for entry in ranking_list:
                retrieved_doc_index, _, retrieved_doc_score =  entry
                retrieved_doc_indices.append(int(retrieved_doc_index))
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
            
            # print("question", question)
            # print('question_id', question_id)
            # pprint(top_ranking_passages[:10])
            # print("pos_ids", pos_ids)
            # # check if any of the top K retrieved items is in the positive items
            # for p in top_ranking_passages:
            #     if p['passage_id'] in pos_ids:
            #         print("found!")
            # input()
            
            query_item = self.prepared_data.vqa_data_with_dpr_output.lookup[str(question_id)]
            # pos_item_contents = [self.prepared_data.passages.id2doc[pos_id] for pos_id in pos_ids]
            batched_data = {
                "question_id": question_id,
                "top_ranking_passages": top_ranking_passages,
                "pos_item_ids": pos_ids,
                "neg_item_ids": neg_ids,
            }
            if query_item.get("answers", None) is not None:
                batched_data["answers"] = list(query_item.answers)
            if query_item.get("gold_answer", None) is not None:
                batched_data["gold_answer"] = query_item.gold_answer
            batch_result.append(batched_data)
        
        if self.config.args.log_prediction_tables_with_images:
            artifact = self.wandb_logger.experiment.use_artifact(self.config.args.wandb_artifacts, type='dataset')
        
        # Log results
        columns=["question_id", "input_image", "image_key",  "pos_item_ids"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)
        
        to_write_data = {
            'output': [],
        }
        for re in tqdm(batch_result):
            to_write_data['output'].append(re)
            question_id = re['question_id']
            knowledge_item = self.prepared_data.vqa_data_with_dpr_output.lookup[str(question_id)]

            # pos_item_contents = [self.prepared_data.passages.id2doc[pos_id] for pos_id in pos_ids]
            table_entry = [
                knowledge_item['img_id'],
                knowledge_item['img_path'],
                knowledge_item['img_path'],
                str(knowledge_item['pos_item_ids']),
                # pos_item_contents,
            ]

            # if self.config.args.log_prediction_tables_with_images:
            #     # Replace image keys with real images
            #     input_image_file_name = knowledge_item['img_file_name']
            #     input_image = artifact.get(input_image_file_name)
            #     if input_image is None:
            #         input_image = artifact.get(input_image_file_name)
                
            #     table_entry[1] = input_image
            
            table_entry+=[p['content'] for p in re['top_ranking_passages']]
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
