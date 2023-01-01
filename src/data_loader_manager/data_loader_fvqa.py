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

from fuzzywuzzy import fuzz
from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from utils.cache_system import save_cached_data, load_cached_data

from data_loader_manager.data_loader_okvqa import DataLoaderOKVQA
from data_loader_manager.datasets import *

from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class DataLoaderFVQA(DataLoaderOKVQA):
    '''
    Data loader for FVQA dataset
    '''

    def __init__(self, config):
        DataLoaderOKVQA.__init__(self, config)


    def LoadFVQAData(self, module_config):
        '''
        Load fvqa data into self.data.fvqa_data
        {
          "type": "LoadFVQAData", "option": "default",
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
            "image_data_path": {
                "full": "..",
            },
        },
        '''
        

        ######################
        #   Read F-VQA data
        ######################
        answer_candidate_list = []
        
        self.data.fvqa_data = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
            "triplets": {},
        })

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
        
        # Load KG triplets
        with open(module_config.config.vqa_data_path.kg_files.full, 'r') as f:
            all_kg_data = json.load(f)
        self.data.fvqa_data.triplets = all_kg_data
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
        
        for data_split in ['train', 'test']:

            # For each data split, prepare dataset
            self.data.fvqa_data[data_split] = load_cached_data(self.config, '{}_data_preprocessed'.format(data_split))
            if not self.data.fvqa_data[data_split]:
                # This split data is not cached
                self.data.fvqa_data[data_split] = EasyDict({}) # re-initialise
                # Get list of imgs from pre-defined data splits
                with open(data_split_files[data_split], 'r') as f:
                    all_img_names = [x.strip() for x in f.readlines()]
                img_data_path = module_config.config.image_data_path.full
                img_list = []
                for imgFilename in all_img_names:
                    img_path = os.path.join(img_data_path, imgFilename)
                    img_list.append((imgFilename, img_path))
                    if self.config.data_loader.dummy_dataloader:
                        # Load only a few samples for testing
                        if len(img_list) > 20:
                            break

                logging.info(f'Loaded {len(img_list)} images for {data_split}')

                # Create entries for each question and related answers
                self.data.fvqa_data[data_split].data_items = []
                for imgFilename, img_path in tqdm(img_list):
                    img_key = imgFilename.split('.')[0]
                    img_caption = self.data.caption_features.get(img_key, None)
                    if img_caption is not None: 
                        img_caption = img_caption[0] 
                    else: 
                        logger.warning(f'No caption found for {img_key}!')
                    
                    
                    img = cv2.imread(img_path)
                    
                    for question_and_answer in img_to_question_dict[imgFilename]:
                        """
                        {"fact_surface": "You are likely to find [[a trumpet]] in [[a jazz club]]", 
                        "ans_source": "image", 
                        "answer": "trumpet", 
                        "question": "Which object can be found in a jazz club", 
                        "img_file": "ILSVRC2012_test_00050748.JPEG", 
                        "visual_concept": "obj", 
                        "kb_source": "conceptnet", 
                        "fact": ["conceptnet/e/f768f157e4446dd594536f8ef02681515586ba2d"], 
                        "question_id": "270"}
                        """
                        # For each question and related answers, create an entry
                        entry_data = EasyDict()

                        if len(question_and_answer['fact']) > 1:
                            logging.warning(f'Multiple facts found for {entry_data.question_id}')
                        fact =  question_and_answer['fact'][0]
                        entry_data.fact = fact

                        ans = question_and_answer['answer']
                        # Some simple cleanup of the answers
                        ans = ans.strip().lower()
                        # ans = ans.replace('.', '')
                        # ans = ans.replace('a ', '')
                        # ans = ans.replace('an ', '')
                        # ans = ans.replace('the ', '')

                        # print(ans, all_kg_data[fact]['e1_label'], all_kg_data[fact]['e2_label'])
                        ratio1 = fuzz.WRatio(ans, all_kg_data[fact]['e1_label'])
                        ratio2 = fuzz.WRatio(ans, all_kg_data[fact]['e2_label'])
                        # if ratio1 > ratio2:
                        #     print(f'{ans} is closer to {all_kg_data[fact]["e1_label"]}')
                        # else:
                        #     print(f'{ans} is closer to {all_kg_data[fact]["e2_label"]}')
                        # input()
                        ans = all_kg_data[fact]['e1_label'] if ratio1 > ratio2 else all_kg_data[fact]['e2_label']

                        entry_data.answers = [ans]
                        entry_data.gold_answer = ans
                        entry_data.question = question_and_answer['question'].lower()
                        entry_data.question_id = str(question_and_answer['question_id'])
                        entry_data.img_path = img_path
                        entry_data.img_key = img_key
                        entry_data.img_key_full = img_key
                        entry_data.img = img
                        entry_data.img_caption = img_caption

                        

                        self.data.fvqa_data[data_split].data_items.append(entry_data)

                        # Collect answer candidates for evaluation
                        if ans not in answer_candidate_list:
                            answer_candidate_list.append(ans)
                
                # After building the data split, save to cache
                save_cached_data(self.config, self.data.fvqa_data[data_split], '{}_data_preprocessed'.format(data_split))

            for entry_data in self.data.fvqa_data[data_split].data_items:
                self.data.fvqa_data['lookup'][str(entry_data.question_id)] = entry_data

            
            # Report statistics
            logger.info('[Data statistics] split: {}  entries: {}'.format(
                data_split,
                len(self.data.fvqa_data[data_split].data_items)))

        # Save answer candidate list
        self.data.fvqa_data.answer_candidate_list = answer_candidate_list

        self.data.vqa_data = self.data.fvqa_data


    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'data': self.data.fvqa_data.train,
            'vinvl_features': self.data.vinvl_features,
            # 'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.fvqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'train',
        }
        self.train_dataset = FVQADataset(self.config, train_dataset_dict)
        # for i in self.train_dataset:
        #     pprint(i)
        #     input()
        train_sampler = RandomSampler(self.train_dataset)
        # train_sampler = SequentialSampler(self.train_dataset)
        
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
            'data': self.data.fvqa_data.test,
            'vinvl_features': self.data.vinvl_features,
            # 'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.fvqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'test',
        }
        self.test_dataset = FVQADataset(self.config, test_dataset_dict)

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )

        logger.info('[Data Statistics]: training data loader: {};  test data loader: {}'.format(
                                len(self.train_dataloader), 
                                len(self.test_dataloader)))




