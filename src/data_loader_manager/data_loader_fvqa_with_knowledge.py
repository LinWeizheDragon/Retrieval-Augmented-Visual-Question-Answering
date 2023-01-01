import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64
from copy import deepcopy
from time import time
from datetime import datetime
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


import logging
logger = logging.getLogger(__name__)
from utils.cache_system import save_cached_data, load_cached_data


from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from transformers import ViTFeatureExtractor

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval

from data_loader_manager.data_loader_fvqa import DataLoaderFVQA
from data_loader_manager.datasets import *

class DataLoaderFVQAWithKnowledge(DataLoaderFVQA):
    '''
    Data loader for our F-VQA dataset
    Knowledge passages are incorporated
    '''

    def __init__(self, config):
        DataLoaderFVQA.__init__(self, config)


    def LoadFVQAPassageData(self, module_config):
        """
        This function loads data and features from the surface facts derived from KG triplets
        {
            "type": "LoadFVQAPassageData", "option": "default",
            "config": {
                "use_split": "0",
                "vqa_data_path": {
                    "question_files":{
                        "full": "..",
                    },
                    "kg_files":{
                        "full": "..",
                    },
                    "split_files": {
                        "full": "..",
                    }
                },
            },
        }
        """
        self.data.passages = {
            'id2doc': {}, # full corpus
            'id2doc_train': {}, # only train corpus
            'index2id': {},
            'annotations': {},
        }

        ######################
        # Read knowledge passage data
        ######################

        # Load KG triplets
        with open(module_config.config.vqa_data_path.kg_files.full, 'r') as f:
            all_kg_data = json.load(f)
        
        for fact_url, triplet_data in tqdm(all_kg_data.items(), desc='Loading KG triplets'):
            """
            "conceptnet/e/a9c086711e083ba3503a06846848e30999b4d3f7": {
                "KB": "conceptnet", 
                "e1_label": "monkey meat", 
                "e2_label": "meat", 
                "uri": "/a/[/r/RelatedTo/,/c/en/monkey_meat/n/,/c/en/meat/]", 
                "surface": "[[monkey meat]] is related to [[meat]]", 
                "dataset": "/d/wiktionary/en/en", 
                "sources": "/and/[/s/rule/definition_section/,/s/web/en.wiktionary.org/wiki/monkey_meat/]", 
                "r": "/r/RelatedTo", 
                "context": "/ctx/all", 
                "score": 1.0, 
                "e1": "/c/en/monkey_meat/n", 
                "e2": "/c/en/meat"}
            """
            content = triplet_data['surface'].replace('[[', '').replace(']]', '')
            self.data.passages.id2doc_train[fact_url] = content.lower()
            self.data.passages.id2doc[fact_url] = content.lower()
            self.data.passages.n_train = len(self.data.passages.id2doc_train)

        # Save to a csv file for future use
        csv_path = os.path.join(os.path.dirname(module_config.config.vqa_data_path.kg_files.full), "kg_surface_facts.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['kid', 'text'])
                for fact_url, content in self.data.passages.id2doc.items():
                    csv_writer.writerow([fact_url, content])
            logging.info("Saved knowledge passage data to {}".format(csv_path))
        
        # We use fact_url as the unique identifier for each fact
        # Therefore, we need to create a mapping from index to fact_url
        for index, fact_url in enumerate(self.data.passages.id2doc.keys()):
            self.data.passages.index2id[str(index)] = fact_url


    def LoadFVQAAnnotations(self, module_config):
        """
        This function loads FVQA Corpus annotations
        {
          "type": "LoadFVQAAnnotations", "option": "default",
          "config": {
            "config": {
                "use_split": "0",
                "vqa_data_path": {
                    "question_files":{
                        "full": "..",
                    },
                    "kg_files":{
                        "full": "..",
                    },
                    "split_files": {
                        "full": "..",
                    }
                },
            },
          },
        },
        """
        ######################
        #  Get supervision annotations
        ######################

        # load json file
        with open(module_config.config.vqa_data_path.question_files.full, 'r') as f:
            all_question_data = json.load(f)
        
        # use img filename as the key, create mappings
        img_to_question_dict = {}
        for question_dict in all_question_data.values():
            img_to_question_dict.setdefault(question_dict['img_file'], []).append(question_dict)
        
        
        # Load dataset split
        data_split_files = {
            'train': os.path.join(
                module_config.config.vqa_data_path.split_files.full, 
                f'train_list_{module_config.config.use_split}.txt'
            ),
            'test': os.path.join(
                module_config.config.vqa_data_path.split_files.full, 
                f'test_list_{module_config.config.use_split}.txt'
            ),
        }

        self.data.fvqa_data_with_dpr_output = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
        })
        self.data.passages.annotations = EasyDict({})

        for data_split in ['train', 'test']:
            for entry_data in self.data.fvqa_data[data_split].data_items:
                
                question_id = entry_data['question_id']
                # in this dataset, only one triplet is annotated
                knowledge_collection = [entry_data['fact']]

                if self.data.passages.annotations.get(str(question_id), None) is None:
                    self.data.passages.annotations[str(question_id)] = {
                        'passages': knowledge_collection,
                    }
                else:
                    logger.error(f'Duplicated entries in annotation files! question id: {str(question_id)} collision!')
                
        for data_split in ['train', 'test']:
            self.data.fvqa_data_with_dpr_output[data_split] = EasyDict({})
            self.data.fvqa_data_with_dpr_output[data_split].data_items = []
            
            missing_entries = []
            missing_data = []

            for item in self.data.fvqa_data[data_split].data_items:
                question_id = item['question_id']
                annotation = self.data.passages.annotations.get(str(question_id), None)
                
                if annotation is None:
                    missing_entries.append(str(question_id))
                    # logger.warning("question {} (split {}) not found in knowledge.".format(str(question_id), data_split))
                    if self.config.mode == 'train':
                        continue
                    else:
                        # in testing mode, all samples must be used
                        related_knowledge = [1]
                else: 
                    related_knowledge = annotation['passages']
                    if len(related_knowledge) == 0:
                        missing_data.append(str(question_id))
                        # logger.warning("question {} (split {}) has no related knowledge in annotations.".format(str(question_id), data_split))
                        related_knowledge = [1]
                
                if data_split == 'train':
                    if self.config.mode == 'train':
                        # in training, each knowledge is treated as one sample
                        for passage_id in related_knowledge:
                            knowledge_item = EasyDict(dict(item))
                            knowledge_item['passage_id'] = passage_id
                            # passage_id will be within train corpus
                            knowledge_item['passage_content'] = self.data.passages.id2doc[str(passage_id)]
                            self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                    else:
                        # in testing mode, each question is one sample
                        # pick one passage for each question
                        knowledge_item = EasyDict(dict(item))
                        passage_id = related_knowledge[0]
                        knowledge_item['passage_id'] = passage_id
                        knowledge_item['passage_content'] = self.data.passages.id2doc[str(passage_id)]
                        self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                else:
                    knowledge_item = EasyDict(dict(item))
                    knowledge_item['question_id'] = question_id
                    # For testset append only one item
                    passage_id = related_knowledge[0]
                    knowledge_item['passage_id'] = passage_id
                    knowledge_item['passage_content'] = self.data.passages.id2doc[str(passage_id)]
                    self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)

            if len(missing_entries) > 0:
                logger.warning(f"{len(missing_entries)} questions (split {data_split}) not found in knowledge. \n {missing_entries}")
            if len(missing_data) > 0:
                logger.warning(f"{len(missing_data)} questions (split {data_split}) has no annotations. \n {missing_data}")
            

            # Load item data into lookup with question_id as index
            logger.info('Indexing data items...')
            
            for item in tqdm(self.data.fvqa_data_with_dpr_output[data_split].data_items):
                question_id = item['question_id']
                self.data.fvqa_data_with_dpr_output.lookup[str(question_id)] = item
            
            # Report statistics
            logger.info('[Data statistics] loaded with knowledge data split: {}  entries: {}'.format(
                data_split,
                len(self.data.fvqa_data_with_dpr_output[data_split].data_items)))
        

        self.data.vqa_data_with_dpr_output = self.data.fvqa_data_with_dpr_output

    def LoadPretrainedDPROutputForGoogleSearchPassage(self, module_config):
        """
        This function loads static outputs from pre-trained DPR retriever
        {
            "type": "LoadPretrainedDPROutputForGoogleSearchPassage", "option": "default/random_pick/batch/none",
            "config": {
                "pretrained_dpr_outputs": {
                    "train": "..",
                    "test": "..",
                },
                "add_self_without_knowledge": True,
            },
        }
        option:
            default: Each question is associated to a retrieved document (from a pre-trained DPR model)
                    This results in more samples than the original VQA questions
            random_pick: Each question is associated to a retrieved document (from a pre-trained DPR model)
                    But a random document is selected for each question; will not result in more samples
            batch: Each question is associated to a batch of retrieved documents (from a pre-trained DPR model)
            none: Each question is not associated to any retrieved passage
                    This option can be used for RAG training, where retrieval is done by the RAG model
                    In this case, we only need to load static results
        """
        ######################
        #  Get knowledge output from pre-trained retrievers
        ######################
        self.data.fvqa_data_with_dpr_output = EasyDict({
            'questionId2topPassages': {},
            'train': {},
            'test': {},
            'lookup': {},
        })
        # Get all available data splits inc. additional test dataloaders
        available_data_splits = []
        for data_split, split_content in self.data.fvqa_data.items():
            try:
                if 'data_items' in split_content.keys():
                    available_data_splits.append(data_split)
            except Exception as e:
                continue
        
        num_knowledge_passages = self.config.data_loader.additional.num_knowledge_passages
        for data_split in available_data_splits:
            
            annotation_json = module_config.config.pretrained_dpr_outputs[data_split]
            with open(annotation_json, 'r') as f:
                annotations = json.load(f)['output']
            
            # Read annotations and (optionally) captions
            for annotation in tqdm(annotations):
                question_id = str(annotation['question_id'])
                top_ranking_passages = annotation['top_ranking_passages']
                self.data.fvqa_data_with_dpr_output.questionId2topPassages[question_id] = [passage for passage in top_ranking_passages]

            if module_config.option == 'none':
                # do not need to reload data items
                # copy data items from the original dataset
                self.data.fvqa_data_with_dpr_output[data_split].update(
                    self.data.fvqa_data[data_split])
                for item in tqdm(self.data.fvqa_data[data_split].data_items):
                    question_id = item['question_id']
                    self.data.fvqa_data_with_dpr_output.lookup[str(question_id)] = item
                continue

            self.data.fvqa_data_with_dpr_output[data_split] = EasyDict({})
            self.data.fvqa_data_with_dpr_output[data_split].data_items = []
            
            for item in self.data.fvqa_data[data_split].data_items:
                # read self.data.fvqa_data and form new self.data.fvqa_data_with_knowledge
                # since the number of samples could be different
                question_id = item['question_id']
                related_knowledge = self.data.fvqa_data_with_dpr_output.questionId2topPassages.get(str(question_id), None)
                related_knowledge = [x['passage_id'] for x in related_knowledge]

                if module_config.option == 'batch':
                    # Simply add retrieved documents to an item
                    knowledge_item = EasyDict(dict(item))
                    knowledge_item['passage_ids'] = [x for x in related_knowledge[:num_knowledge_passages]]
                    knowledge_item['passage_contents'] = [self.data.passages.id2doc[str(passage_id)] for passage_id in knowledge_item['passage_ids']]
                    # Add to self.data.fvqa_data_with_dpr_output
                    self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                    continue
                
                # Currently not available
                # if module_config.config.add_self_without_knowledge:
                #     # add self prediction without knowledge
                #     knowledge_item = {}
                #     knowledge_item['question_id'] = question_id
                #     knowledge_item['passage_id'] = -1
                #     knowledge_item['passage_content'] = ''
                #     self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                
                if module_config.option == 'random_pick':
                    # Select one random document for each question
                    # This is to avoid having too many samples
                    # Squeeze data into one item, help to cover more distinct questions in training
                    # But in testing, should still populate the dataset
                    if data_split == 'train':
                        knowledge_item = EasyDict(dict(item))
                        knowledge_item['question_id'] = question_id
                        knowledge_item['passage_ids'] = [x for x in related_knowledge[:num_knowledge_passages]]
                        knowledge_item['passage_contents'] = [self.data.passages.id2doc[str(passage_id)] for passage_id in knowledge_item['passage_ids']]
                        self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                    else:
                        for passage_id in related_knowledge[:num_knowledge_passages]:
                            knowledge_item = EasyDict(dict(item))
                            knowledge_item['question_id'] = question_id
                            knowledge_item['passage_id'] = passage_id
                            knowledge_item['passage_content'] = self.data.passages.id2doc[str(passage_id)]
                            self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                    continue

                if module_config.option == 'default':
                    # populate the dataset with all retrieved documents
                    for passage_id in related_knowledge[:num_knowledge_passages]:
                        knowledge_item = EasyDict(dict(item))
                        knowledge_item['question_id'] = question_id
                        knowledge_item['passage_id'] = passage_id
                        knowledge_item['passage_content'] = self.data.passages.id2doc[str(passage_id)]
                        self.data.fvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                    
            # Load item data into lookup with question_id as index
            logger.info('Indexing data items...')
            
            for item in tqdm(self.data.fvqa_data_with_dpr_output[data_split].data_items):
                question_id = item['question_id']
                self.data.fvqa_data_with_dpr_output.lookup[str(question_id)] = item
            
            # Report statistics
            logger.info('[Data statistics] loaded with knowledge data split: {}  entries: {}'.format(
                data_split,
                len(self.data.fvqa_data_with_dpr_output[data_split].data_items)))
        
        self.data.vqa_data_with_dpr_output = self.data.fvqa_data_with_dpr_output

    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'data': self.data.vqa_data.train if 'vqa_data_with_dpr_output' not in self.data.keys() \
                    else self.data.vqa_data_with_dpr_output.train,
            'passages': self.data.passages,
            'vinvl_features': self.data.vinvl_features,
            # 'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'train',
        }
        self.train_dataset = globals()[self.config.data_loader.dataset_type](self.config, train_dataset_dict)
        # for i in self.train_dataset:
        #     pprint(i)
        #     input()
        train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
        )
        # for i in self.train_dataloader:
        #     print(i)
        #     input()

        test_dataset_dict = {
            'data': self.data.vqa_data.test if 'vqa_data_with_dpr_output' not in self.data.keys() \
                    else self.data.vqa_data_with_dpr_output.test,
            'passages': self.data.passages,
            'vinvl_features': self.data.vinvl_features,
            # 'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'test',
        }
        self.test_dataset = globals()[self.config.data_loader.dataset_type](self.config, test_dataset_dict)

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size if self.config.mode=='train' else self.config.test.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )
        # for i in self.test_dataloader:
        #     print(i)
        #     input()
        logger.info('[Data Statistics]: training data loader: {};  test data loader: {}'.format(
                                len(self.train_dataloader), 
                                len(self.test_dataloader)))


