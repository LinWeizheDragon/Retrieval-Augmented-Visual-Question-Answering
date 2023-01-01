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


class OKVQADatasetWithRandomPickPassages(OKVQADataset, ModuleParser):
    """
    This is a dataset class for OKVQA dataset with randomly picked passage inputs
    query + [passsages] will be transformed into query + passage{random_index}
    In training, we randomly pick a passage for each sample
    In testing, all passages are provided to run tests
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.passages = dataset_dict['passages']
        self.n_passages = len(self.passages.id2doc) # number of passages
       

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        item = self.data.data_items[idx]
        
        if self.mode == 'train':
            # in training, randomly pick a passage
            select_index = random.choice(range(len(item.passage_ids)))
            # load passage
            passage_id = item.passage_ids[select_index]
            passage_content = item.passage_contents[select_index]
        else:
            # in testing, use the provided passage
            passage_id = item.passage_id
            passage_content = item.passage_content

        sample.update({
            'passage_id': passage_id,
            'passage_content': passage_content,
        })
        return sample

class OKVQADatasetWithBatchPassages(OKVQADataset, ModuleParser):
    """
    This is a dataset class for OKVQA dataset with batched passage inputs
    query + [passsages] will be transformed into [query + passage1, query + passage2, ...]
    But the number of output elements remain the same (1 sample matches to 1 output)
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.passages = dataset_dict['passages']
        self.n_passages = len(self.passages.id2doc) # number of passages
       

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        item = self.data.data_items[idx]
        
        sample.update({
            'passage_ids': item.passage_ids,
            'passage_contents': item.passage_contents,
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
        output_data = EasyDict()

        #############################
        #  User ModuleParser to parse every sample
        #       according to what modules are selected
        #       modules are parsed in order
        #############################
        for sample in batch:
            for passage_index in range(len(sample.passage_contents)):
                ###### For each passage, generate input #######
                new_sample = EasyDict(sample.copy())
                new_sample.passage_content = sample.passage_contents[passage_index]
                parsed_data = self.parse_modules(new_sample, input_modules, type='input')
                for key, value in parsed_data.items():
                    input_data.setdefault(key, []).append(value)
                
                parsed_data = self.parse_modules(new_sample, decoder_input_modules, type='decoder_input')
                for key, value in parsed_data.items():
                    decoder_input_data.setdefault(key, []).append(value)

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


class OKVQADatasetWithAllPassages(OKVQADataset, ModuleParser):
    """
    This is a dataset class for OKVQA dataset with all passage inputs
    query + [passsages] will be transformed into [query + passage1, query + passage2, ...]
    All passages are provided for training and testing
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.passages = dataset_dict['passages']
        self.n_passages = len(self.passages.id2doc) # number of passages
       

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        item = self.data.data_items[idx]
        
        # use the provided passage
        passage_id = item.passage_id
        passage_content = item.passage_content

        sample.update({
            'passage_id': passage_id,
            'passage_content': passage_content,
        })
        return sample
