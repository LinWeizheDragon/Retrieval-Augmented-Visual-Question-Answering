import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator
from trainers.base_executor import BaseExecutor
import wandb
import logging
logger = logging.getLogger(__name__)

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

from .metrics_processors import MetricsProcessor
from .base_executor import BaseExecutor
from models.retriever.retriever_dpr import RetrieverDPR
from utils.dirs import *
from utils.evaluation import ranklist_by_heapq, ranklist_by_sorted
import heapq
from torch.multiprocessing import Manager, spawn, Process
import faiss

class DPRExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.train_data_loader = self.data_loader.train_dataloader
        self.test_data_loader = self.data_loader.test_dataloader

        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer
        
        ModelClass = globals()[self.config.model_config.ModelClass]
        self.model = ModelClass(config=config)
        
        self.model.resize_token_embeddings(len(self.tokenizer), len(self.decoder_tokenizer))
        

        

    
    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """

        optimization_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()],
                'lr': self.config.train.lr,
                'initial_lr': self.config.train.lr,
            },
        ]
        
        for group in optimization_parameters:
            logger.info('#params: {}   lr: {}'.format(len(group['params']), group['lr']))
        
        """define optimizer"""
        self.optimizer = torch.optim.AdamW(
            optimization_parameters, lr=self.config.train.lr)

        if self.config.train.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.config.train.scheduler == 'cosine':
            t_total = self.config.train.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                            t_total, eta_min=1e-5, last_epoch=-1, verbose=False)
        else:
            from transformers import get_constant_schedule_with_warmup
            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
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
            'item_input_ids': sample_batched['decoder_input_ids'].to(self.device),
            'item_attention_mask': sample_batched['decoder_input_attention_mask'].to(self.device),
        })

        forward_results = self.model(**train_batch)
        batch_loss = forward_results.loss

        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(f"train/lr[{index}]", current_lr, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", batch_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        data_to_return = {
            'loss': batch_loss,
        }
        return data_to_return
    
    def validation_step(self, sample_batched, batch_idx):
        return self._compute_query_embeddings_step(sample_batched, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        log_dict = self.evaluate_outputs(validation_step_outputs, mode='valid')
        self.logging_results(log_dict)
        if self.config.valid.additional.save_HF_model:
            self.save_HF_model()
        return log_dict
    
    def test_step(self, sample_batched, batch_idx):
        return self._compute_query_embeddings_step(sample_batched, batch_idx)

    def test_epoch_end(self, validation_step_outputs):
        log_dict = self.evaluate_outputs(validation_step_outputs)
        self.logging_results(log_dict, prefix=self.config.test.evaluation_name)
        if self.config.test.additional.save_HF_model:
            self.save_HF_model()
        return log_dict

    def val_dataloader(self):
        return self.get_data_loader_wrt_mode()
    
    def test_dataloader(self):
        return self.get_data_loader_wrt_mode()

    def get_data_loader_wrt_mode(self):
        # Decide which VQA dataset split to use for evaluation
        if self.config.data_loader.use_dataset == 'train':
            query_data_loader = self.train_data_loader
        else:
            query_data_loader = self.test_data_loader
        return query_data_loader
        

    def _compute_query_embeddings_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        test_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
        })
        # batch_size x hidden_states
        query_emb = self.model.generate_query_embeddings(**test_batch)
        
        data_to_return = {
            'btach_idx': batch_idx,
            'query_emb': query_emb,
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
        }

        return data_to_return
    
    


    def evaluate_outputs(self, step_outputs, mode='test'):
        # Batching every validation step outputs
        # n_queries x hidden_size
        
        query_embeddings = []
        question_ids = []
        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])
            question_ids += step_output['question_ids']
        
        query_embeddings = torch.cat(query_embeddings, dim=0)

        ##################################
        ##    Generate embeds for items ##
        ##################################
        
        # Decide which corpus to use for evaluating the VQA queries
        if self.config.data_loader.additional.full_corpus_in_testing:
            passage_id2doc = self.data_loader.data.passages.id2doc 
        else:
            passage_id2doc = self.data_loader.data.passages.id2doc_train
        
        n_items = len(passage_id2doc)
        if self.trainer.state.stage in ['sanity_check']:
            # sanity check
            logging.warning('No steps have been taken. Reducing number of items to speed up the sanity check.')
            n_items = 100

        # query_embeddings = torch.zeros(len(self.data_loader.test_dataset), 768) 
        
        n_queries = query_embeddings.shape[0]
        hidden_size = query_embeddings.shape[1]

        i_batch_size = self.config[mode].batch_size
        
        n_item_batchs = n_items // i_batch_size + 1

        rate_batch = np.zeros(shape=(n_queries, n_items))
        # rate_batch = np.random.randint(0, 100, size=(n_queries, n_items))
        logger.info(f'rate_batch shape: {rate_batch.shape}')
        
        # Create mapping between matrix indice and passage ids
        # Using only train passage corpus
        passage_index2id = {index:passage_id for index, passage_id in enumerate(passage_id2doc.keys()) if index < n_items}
        decoder_input_modules = self.config.model_config.decoder_input_modules.module_list
        passage_contents = []
        for passage_id in passage_id2doc.keys():
            sample = EasyDict(passage_content=passage_id2doc[passage_id])
            parsed_data = self.data_loader.test_dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
            passage_contents.append(parsed_data.text_sequence)
        
        # Create mapping between matrix indice and question ids
        # Using test set questions
        question_index2id = {index:question_id for index, question_id in enumerate(question_ids)}

        assert len(passage_index2id) == n_items
        assert len(question_index2id) == n_queries
        logger.info(f'There are {n_queries} queries.')
        logger.info(f'Generating embeddings for items; there are {n_items} items.')
        i_count = 0
        item_embeddings = []
        for i_batch_id in tqdm(range(n_item_batchs)):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, n_items)

            passage_contents_batch = passage_contents[i_start:i_end]
            # print(passage_contents_batch)
            # Encode this batch of data
            item_encoding = self.decoder_tokenizer(passage_contents_batch,
                                padding='longest',
                                max_length=self.config.data_loader.additional.max_decoder_source_length,
                                truncation=True,
                                return_tensors="pt")
            
            item_input_ids, item_attention_mask = item_encoding.input_ids, item_encoding.attention_mask
            test_batch = EasyDict({
                'input_ids': item_input_ids.to(self.device),
                'attention_mask': item_attention_mask.to(self.device),
            })
            
            # batch_size x hidden_states
            item_emb = self.model.generate_item_embeddings(**test_batch)
            for x in item_emb:
                item_embeddings.append(x.cpu().detach().numpy())
            
            # n_queries x batch_size
            i_rate_batch = torch.matmul(query_embeddings, item_emb.t()).detach().cpu()

            rate_batch[:, i_start:i_end] = i_rate_batch
            i_count += i_rate_batch.shape[1]

        assert i_count == n_items
        

        # query_list_batch = list(range(n_queries))
        # # Zip data together
        # query_batch_rating_uid = zip(rate_batch, query_list_batch)

        # Ks = self.config.model_config.Ks
        # print('start multiprocessing...')
        # Launch multi-processing to speed up ranking
        # manager = Manager()
        # recorder_queue = manager.Queue()
        # task_queue = manager.Queue(100)
        # NUM_PROCESSES = 4
        # ps = []
        # for i in range(NUM_PROCESSES):
        #     p = spawn(test_thread, args=(i,
        #                                 self.config,
        #                                 task_queue,
        #                                 recorder_queue,
        #                                 passage_contents,
        #                                 passage_index2id,
        #                                 Ks), join=False)
        #     ps.append(p)
        
        # print('waiting for subprocesses to finish...')
        # for i_batch, sample_batched in enumerate(tqdm(query_batch_rating_uid, total=len(query_list_batch))):
        #     try:
        #         question_id = question_index2id[sample_batched[1]]
        #         query_item = self.data_loader.data.vqa_data_with_dpr_output.lookup[str(question_id)]
        #         task_queue.put((i_batch, sample_batched, query_item), block=True)
        #         # print('new task {} has been initialized'.format(i))
        #         i = i + 1
        #     except Exception as e:
        #         print(e)

        # # Wait for all processes done
        # for p in ps:
        #     p.join()

        # # Read recorder queue until finish all
        # batch_result = []
        # count_task = 0
        # while recorder_queue.qsize() > 0:
        #     output = recorder_queue.get(block=True)
        #     # print('getting', len(batch_result), '/', len(user_list_batch))
        #     batch_result.append(output)
        
        # Update: use faiss
        hidden_size = item_embeddings[0].shape[-1]
        print("hidden size", hidden_size)
        
        if "exhaustive_search_in_testing" in self.config.model_config.modules:
            index = faiss.IndexFlatIP(hidden_size)
        else:
            index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)

        Ks = self.config.model_config.Ks
        item_embeddings = np.stack(item_embeddings, 0)
        index.add(item_embeddings)
        search_res = index.search(query_embeddings.cpu().numpy(), k=max(Ks))

        batch_result = []
        for query_id, return_scores, return_passage_index in zip(range(len(query_embeddings)), search_res[0], search_res[1]):
            question_id = question_index2id[query_id]
            # Retrieve content from passages
            top_ranking_passages = [{
                'passage_index': i,
                'passage_id': passage_index2id[i],
                'content': passage_contents[i],
                'score': float(return_scores[index]),
                } for index, i in enumerate(return_passage_index)]
            
            query_item = self.data_loader.data.vqa_data_with_dpr_output.lookup[str(question_id)]
            answers = query_item.answers
            gold_answer = query_item.gold_answer
            log_result = {
                'recall': np.zeros(len(Ks)),
                'precision': np.zeros(len(Ks)),
                'gold_precision': np.zeros(len(Ks)),
                'gold_recall': np.zeros(len(Ks)),
            }
            for indexK, K in enumerate(Ks):
                found_answers = []
                found_gold_answers = []
                for passage_data in top_ranking_passages[:K]:
                    
                    for answer in answers:
                        if answer.lower() in passage_data['content'].lower():
                            found_answers.append(answer)
                            break
                    if gold_answer.lower() in passage_data['content'].lower():
                        found_gold_answers.append(answer)
                    
                if len(found_answers) > 0:
                    # At least one answer is retireved
                    log_result['recall'][indexK] += 1
                # The proportion of retrieved knowledge has an answer
                log_result['precision'][indexK] += len(found_answers) / K

                if len(found_gold_answers) > 0:
                    # if gold answer is found
                    log_result['gold_recall'][indexK] += 1
                # The proportion of retrieved knowledge has the gold answer
                log_result['gold_precision'][indexK] += len(found_gold_answers) / K

            log_result['output'] = {
                'question_id': str(question_id),
                'top_ranking_passages': top_ranking_passages,
            }
            batch_result.append(log_result)

        logger.info('Testing finished and data has been collected from the multi-processing unit.')

        # Log results
        columns=["question_id", "image_key", "question", "caption", "answers", "gold_answer"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)
        
        to_write_data = {
            'output': [],
        }
        for re in tqdm(batch_result):
            to_write_data['output'].append(re['output'])
            question_id = re['output']['question_id']
            knowledge_item = self.data_loader.data.vqa_data.lookup[str(question_id)]
            table_entry = [
                knowledge_item['question_id'],
                knowledge_item['img_key'],
                knowledge_item['question'],
                knowledge_item['img_caption']['caption'],
                knowledge_item['answers'],
                knowledge_item['gold_answer'],
            ]
            
            table_entry+=[p['content'] for p in re['output']['top_ranking_passages']]
            # print(table_entry)
            test_table.add_data(*table_entry)
        
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_result=batch_result,
            Ks=Ks,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)

        # to_write_data['result'] = log_dict.metrics.copy()

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
        
        if self.config.mode == "test":
            from utils.numpy_encoder import NpEncoder
            # Save predictions to files for DPR-based VQA systems
            json_path = os.path.join(self.config.results_path, '{}_predictions.json'.format(prefix))
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
        path_save_model = os.path.join(self.config.saved_model_path, 'epoch{}'.format(self.current_epoch))
        self.model.query_encoder.save_pretrained(os.path.join(path_save_model, 'query_encoder'))
        self.data_loader.tokenizer.save_pretrained(os.path.join(path_save_model, 'query_encoder_tokenizer'))
        self.model.item_encoder.save_pretrained(os.path.join(path_save_model, 'item_encoder'))
        self.data_loader.decoder_tokenizer.save_pretrained(os.path.join(path_save_model, 'item_encoder_tokenizer'))
        logger.info('Model has been saved to {}'.format(path_save_model))

    
    def forward(self, **kwargs):
        return self.model(**kwargs)





def test_thread(thread_subid, thread_index, config, 
                                            task_queue,
                                             recorder_queue, 
                                             passage_contents,
                                             passage_index2id,
                                             Ks):

    try:
        print('start!')
        # Set seed
        # if config.seed:
        #     # set_seed(config.seed)
        #     print("thread SEED is set to:", config.seed)
        print('thread {} initiated'.format(thread_index))
        time.sleep(5)
        RETRY = 0
        while True:
            if task_queue.qsize() == 0:
                if RETRY < 3:
                    print('thread {} retrying... {}'.format(thread_index, RETRY))
                    time.sleep(3)
                    RETRY += 1
                else:
                    break
            else:
                RETRY = 0
            
            CONTINUE = True
            try:
                i_batch, sample_batched, query_item = task_queue.get(block=False)
                # print('thread {} gets task {}'.format(thread_index, i_batch))
            except Exception as e:
                print(e)
                CONTINUE = False

            if CONTINUE:
                x = sample_batched
                # Ratings for this query
                rating = x[0]
                # Query index
                query_index = x[1]
                # Related item (with everything for this question)
                # query_item
                item_score = {}
                for i in range(len(rating)):
                    item_score[i] = rating[i]
                
                K_max = max(Ks)
                # Passages with the highest scores
                K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
                
                # Retrieve content from passages
                top_ranking_passages = [{
                                        'passage_index': i,
                                        'passage_id': passage_index2id[i],
                                        'content': passage_contents[i],
                                        'score': item_score[i],
                                        } for i in K_max_item_score]
                # pprint(top_ranking_passages)

                # Evaluate results
                answers = query_item.answers
                gold_answer = query_item.gold_answer
                log_result = {
                    'recall': np.zeros(len(Ks)),
                    'precision': np.zeros(len(Ks)),
                    'gold_precision': np.zeros(len(Ks)),
                    'gold_recall': np.zeros(len(Ks)),
                }
                for indexK, K in enumerate(Ks):
                    found_answers = []
                    found_gold_answers = []
                    for passage_data in top_ranking_passages[:K]:
                        
                        for answer in answers:
                            if answer.lower() in passage_data['content'].lower():
                                found_answers.append(answer)
                                break
                        if gold_answer.lower() in passage_data['content'].lower():
                            found_gold_answers.append(answer)
                        
                    if len(found_answers) > 0:
                        # At least one answer is retireved
                        log_result['recall'][indexK] += 1
                    # The proportion of retrieved knowledge has an answer
                    log_result['precision'][indexK] += len(found_answers) / K

                    if len(found_gold_answers) > 0:
                        # if gold answer is found
                        log_result['gold_recall'][indexK] += 1
                    # The proportion of retrieved knowledge has the gold answer
                    log_result['gold_precision'][indexK] += len(found_gold_answers) / K

                log_result['output'] = {
                    'question_id': query_item.question_id,
                    'top_ranking_passages': top_ranking_passages,
                }

                recorder_queue.put(log_result)

    except Exception as e:
        print(e)
    
    print('thread {} finished'.format(thread_index))