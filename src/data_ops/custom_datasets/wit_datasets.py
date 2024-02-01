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


from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from data_ops.custom_datasets.module_parser import ModuleParser

from .base_datasets import BaseDataset, DPRBaseDataset



class WITDatasetForDPR(DPRBaseDataset, ModuleParser):
    """
    This is a dataset class for VG dataset used for DPR training
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.passages = dataset_dict['passages']
        if 'images' in dataset_dict.keys():
            self.images = dataset_dict['images']
        if 'image_dataset_with_embeddings' in dataset_dict.keys():
            self.image_dataset_with_embeddings = dataset_dict['image_dataset_with_embeddings']
            self.image_dataset_with_embeddings = self.image_dataset_with_embeddings.to_pandas().set_index("id").to_dict(orient="index")

        s = time.time()
        
        self.passages = EasyDict({
            'dataset': self.passages,
        })
        # first, load full passages as train docs
        # logger.info(f"Using {len(ds)} passage data for training...")
        # self.passages.id2doc_train = ds.to_pandas().set_index("passage_id").to_dict(orient="index")
        # for k, v in tqdm(self.passages.id2doc_train.items(), desc="formatting the passages"):
        #     self.passages.id2doc_train[k] = v['passage_content']
        # self.passages.id2doc = self.passages.id2doc_train
        # for passage_data in tqdm(ds, desc="formatting the passages"):
        #     self.passages.id2doc_train[passage_data['passage_id']] = passage_data['passage_content']
        # self.passages.id2doc = self.passages.id2doc_train

        logger.info(f"passages prepared. used {time.time()-s} secs.")
        
        """
        Negative samples are randomly sampled from the corpus
        Can choose whether sampling can access the full corpus
        """
        self.n_passages = len(self.passages.dataset) # number of passages

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        def negative_sampling(pos_item_ids, negative_source=None, num_samples=1):
            """Generate negative samples for a query. ONLY used in training
            Args:
                user_item (int tensor): user id
                num_samples (int, optional): number of samples. Defaults to 1.
            Returns:
                neg_items: list of negative item ids.
            """
            neg_items = []
            
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
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
                    neg_item = self.passages.dataset[int(neg_item)]
                    if neg_item['passage_id'] in pos_item_ids:
                        VALID = False
                    if "use_self_negatives" in self.config.model_config.modules and negative_source is not None:
                        if neg_item['source_name'] != negative_source:
                            VALID = False
                    
                    if VALID == True:
                        break
                neg_items.append(neg_item)
            return neg_items
        
        sample = EasyDict(self.data[idx])
        item = sample
        # these two belong to a positive sample (in annotations)

        selected_pos_index = random.sample(range(len(item.pos_item_ids)), k=1)[0]
        passage_id = item.pos_item_ids[selected_pos_index]
        passage_content = item.pos_item_contents[selected_pos_index] #self.passages.id2doc[passage_id]
        
        neg_items = negative_sampling(
            item.pos_item_ids, 
            negative_source=item.get('use_negative_items', None),
            num_samples=self.config.model_config.num_negative_samples
        )
        neg_passage_ids = [neg_item['passage_id'] for neg_item in neg_items]
        neg_passage_contents = [neg_item['passage_content'] for neg_item in neg_items]

        sample.update({
            'img_path': sample['img_path'],
            'passage_id': passage_id,
            'passage_content': passage_content,
            'pos_item_ids': item.pos_item_ids,
            'neg_passage_ids': neg_passage_ids,
            'neg_passage_contents': neg_passage_contents,
        })
        return EasyDict(sample)


    def collate_fn(self, batch):
        '''
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        try:
            batched_data = super().collate_fn(batch)
        except:
            print("error!")
            print([sample['img_path'] for sample in batch])
            raise Exception
        #############################
        #  Meta Features
        #############################
        question_ids = [sample.question_id for sample in batch]
        passage_ids = [sample.passage_id for sample in batch]
        pos_item_ids = [sample.pos_item_ids for sample in batch]
        neg_item_ids = [
            sample.neg_passage_ids for sample in batch
        ]

        batched_data.update(EasyDict({
            'passage_ids': passage_ids, # currently used pos item
            'question_ids': question_ids,
            'questions': batched_data['input_text_sequences'],
            'pos_item_ids': pos_item_ids, # annotated pos items (all)
            'neg_item_ids': neg_item_ids, # currently used neg items
        }))

        return batched_data