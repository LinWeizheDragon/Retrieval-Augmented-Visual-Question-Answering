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
from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from data_loader_manager.module_parser import ModuleParser
from data_loader_manager.datasets.okvqa_datasets import OKVQADataset



class OKVQADatasetForDPR(OKVQADataset, ModuleParser):
    """
    This is a dataset class for OKVQA dataset used for DPR training
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.passages = dataset_dict['passages']
        """
        Negative samples are randomly sampled from the corpus
        Can choose whether sampling can access the full corpus
        """
        if self.mode == 'train':
            if self.config.data_loader.additional.full_corpus_in_training:
                # random sampling for training is limited to train split
                self.n_passages = len(self.passages.id2doc_train) # number of passages
            else:
                self.n_passages = len(self.passages.id2doc)
        else:
            # while testing, negative samples are not used; do not need to change this value
            self.n_passages = len(self.passages.id2doc) # number of passages

    def __getitem__(self, idx):
        def negative_sampling(question_id, answers, num_samples=1):
            """Generate negative samples for a query. ONLY used in training
            Args:
                user_item (int tensor): user id
                num_samples (int, optional): number of samples. Defaults to 1.
            Returns:
                neg_items: list of negative item ids.
            """
            neg_items = []
            annotations = self.passages.annotations.get(str(question_id), {'passages': []})['passages']
            
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
                question_id = str(question_id)
                while True:
                    # if self.p is not None:
                    neg_item = np.random.randint(low=0, high=self.n_passages-1, size=1)[0]
                    # else:
                    #     neg_item = np.random.choice(self.n_params.n_items, 1, p=self.p)[0]
                    # print(annotations, neg_item)

                    # neg_passage = self.passages.id2doc[str(neg_item)]
                    VALID = True
                    # Validate if this passage is a negative sample
                    # for answer in answers:
                    #     if answer in neg_passage:
                    #         VALID = False
                    if neg_item in annotations:
                        VALID = False
                    
                    if VALID == True:
                        break
                neg_items.append(neg_item)
            return neg_items
        
        sample = super().__getitem__(idx)
        item = self.data.data_items[idx]
        # these two belong to a positive sample (in annotations)
        passage_id = item.passage_id
        passage_content = item.passage_content

        sample.update({
            'passage_id': passage_id,
            'passage_content': passage_content,
            'neg_passage_ids': negative_sampling(item.question_id, item.answers, self.config.model_config.num_negative_samples),
        })
        return sample


    def collate_fn(self, batch):
        '''
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        # According to the settings in config file, prepare the input and output
        input_modules = self.config.model_config.input_modules.module_list
        decoder_input_modules = self.config.model_config.decoder_input_modules.module_list
        output_modules = self.config.model_config.output_modules.module_list
        
        input_data = EasyDict()
        decoder_input_data = EasyDict()
        pos_item_data = EasyDict()
        neg_item_data = EasyDict()
        output_data = EasyDict()

        #############################
        #  User ModuleParser to parse every sample
        #       according to what modules are selected
        #       modules are parsed in order
        #############################
        for sample in batch:
            parsed_data = self.parse_modules(sample, input_modules, type='input')
            for key, value in parsed_data.items():
                input_data.setdefault(key, []).append(value)
            
            # One positive sample + Multiple negative samples
            ###### For the positive passage, generate input #######
            parsed_data = self.parse_modules(sample, decoder_input_modules, type='decoder_input')
            for key, value in parsed_data.items():
                decoder_input_data.setdefault(key, []).append(value)
                pos_item_data.setdefault(key, []).append(value)

            for passage_id in sample.neg_passage_ids:
                ###### For each negative passage, generate input #######
                new_sample = EasyDict(sample.copy())
                passage_content = self.passages.id2doc[str(passage_id)]
                new_sample.passage_content = passage_content
                
                parsed_data = self.parse_modules(new_sample, decoder_input_modules, type='decoder_input')
                for key, value in parsed_data.items():
                    decoder_input_data.setdefault(key, []).append(value)
                    neg_item_data.setdefault(key, []).append(value)
            
            parsed_data = self.parse_modules(sample, output_modules, type='output')
            for key, value in parsed_data.items():
                output_data.setdefault(key, []).append(value)
        
        input_data = EasyDict(input_data)
        decoder_input_data = EasyDict(decoder_input_data)
        output_data = EasyDict(output_data)
        
        #############################
        #  Postprocessing Features
        #############################
        input_post_modules = self.config.model_config.input_modules.postprocess_module_list
        decoder_input_post_modules = self.config.model_config.decoder_input_modules.postprocess_module_list
        output_post_modules = self.config.model_config.output_modules.postprocess_module_list
        
        input_data = self.post_processing(input_data, input_post_modules)
        decoder_input_data = self.post_processing(decoder_input_data, decoder_input_post_modules)
        output_data = self.post_processing(output_data, output_post_modules)
        
        #############################
        #  Meta Features
        #############################
        question_ids = [sample.question_id for sample in batch]
        questions = [sample.question for sample in batch]
        answers = [sample.answers for sample in batch]
        gold_answers = [sample.gold_answer for sample in batch]

        batched_data = EasyDict({
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'gold_answers': gold_answers,
        })
        batched_data.update(input_data)
        batched_data.update(decoder_input_data)
        batched_data.update(output_data)

        return batched_data


        