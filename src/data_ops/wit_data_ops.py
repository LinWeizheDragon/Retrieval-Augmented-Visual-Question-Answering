"""
This file defines the data transforms that will be applied to the data. 
Each transform takes in an EasyDict object of in_features (key: feature_name, value: feature data)
It should output an EasyDict object of out_features (key: feature_name, value: feature_data)
Each transform defined here can be used as an independent unit to form a data pipeline
Some common transforms are provided by runway
"""
from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor, keep_ds_columns

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
from datasets import load_dataset, Dataset, DatasetDict
from datasets import concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AutoImageProcessor

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import PIL.Image

from datasets.utils.file_utils import get_datasets_user_agent
import requests
import hashlib

@register_transform_functor
class LoadWITData(HFDatasetTransform):
    """
    This functor loads WIT data
    1. Load dataset
    2. return the training/valid dataset
    """
    def setup(self, only_main_image=False, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.only_main_image = only_main_image
        

    def _call(self, *args, **kwargs):   
        
        module_config = self.module_config

        ######################
        #   Read WIT data
        ######################
        
        data_paths = module_config.data_paths
        image_data_path = data_paths.image_data_path

        self.data.wit_data = DatasetDict({
            'train': None,
            'valid': None,
            'test': None,
        })

        def add_image_id(batch):
            image_urls = batch['image_url']
            image_ids = [
                hashlib.md5(i.encode()).hexdigest() for i in image_urls
            ]
            batch['image_id'] = image_ids
            return batch
        
        for split in self.data.wit_data.keys():
            split_paths = data_paths[split]
            if self.use_dummy_data:
                split_paths = split_paths[:1]
            
            split_ds = []
            for split_path in split_paths:
                logger.info(f"reading {split_path}...")
                ds = Dataset.from_csv(split_path, delimiter='\t')
                
                # conversion_success = len()
                # print(ds.to_pandas().value_counts(subset=['language']))
                # print(ds[:1])
                # input()
                if self.only_main_image:
                    ds = ds.filter(lambda x: x["language"] == 'en' and x['is_main_image'] == True)
                else:
                    ds = ds.filter(lambda x: x["language"] == 'en')

                if self.use_dummy_data:
                    ds = ds.select(range(min(5000, len(ds))))
                
                split_ds.append(ds)
                
                
            split_ds = concatenate_datasets(split_ds)

            # if self.use_dummy_data:
            #     split_ds = split_ds.select(range(500))

            split_ds = split_ds.map(add_image_id, batched=True, num_proc=16, desc="Add hashed id to images")
            split_ds = split_ds.add_column('original_data_id', [f"{split}_{i}" for i in range(len(split_ds))])

            logger.info(f"Split {split} has {len(split_ds)} entries.")

            self.data.wit_data[split] = split_ds
        
        # full_corpus_ds = concatenate_datasets(concatenate_datasets)
        return self.data.wit_data

@register_transform_functor
class PrepareImagesForWITData(HFDatasetTransform):
    """
    This functor prepares images for WIT data
    1. Download images if `_fetch_images` is set to True
    2. Filter and return only examples that have downloaded images
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        """
        inputs: wit_data: with train and valid HF Datasets
        """
        self.data.update(inputs)
        module_config = self.module_config

        data_paths = module_config.data_paths
        image_data_path = data_paths.image_data_path

        output_data = DatasetDict({
            "images": {},
        })

        for split in ['train', 'valid', 'test']:
            split_ds = self.data[split]

            USER_AGENT = get_datasets_user_agent()
            num_threads = 128

            from random_user_agent.user_agent import UserAgent
            from random_user_agent.params import SoftwareName, OperatingSystem


            # you can also import SoftwareEngine, HardwareType, SoftwareType, Popularity from random_user_agent.params
            # you can also set number of user agents required by providing `limit` as parameter
            
            software_names = [SoftwareName.CHROME.value]
            operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   
            user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

            def random_userAgent():
                # user_agent_list = ["Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
                # "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
                # "Mozilla/5.0 (Windows NT 10.0;) Gecko/20100101 Firefox/61.0",
                # "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
                # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
                # "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
                # "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
                # "Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
                # ]
                return user_agent_rotator.get_random_user_agent()
                
            def fetch_single_image(image_url, image_path, timeout=30, retries=3):
                if os.path.exists(image_path):
                    # print("skipping", image_url)
                    return True
                for _ in range(retries + 1):
                    try:
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}
                        headers['User-Agent'] = random_userAgent()
                        response = requests.get(image_url, stream=True, timeout=timeout, headers=headers)
                        # print(image_url, response)
                        if response:
                            image = PIL.Image.open(response.raw)
                            image = image.convert('RGB')
                            # print('saving image to ', image, image_path)
                            image = image.save(image_path)
                            return True
                        else:
                            image = None
                    except Exception:
                        image = None
                # print("result:", image_url, image)
                return False


            def get_images(batch, batch_id, num_threads, timeout=30, retries=3):
                
                image_urls = batch["image_url"]
                image_ids = batch['image_id']
                image_paths = [
                    os.path.join(image_data_path, f"{i}.jpg") for i in image_ids
                ]
                batch['img_id'] = image_ids
                batch['img_path'] = image_paths

                if module_config._fetch_images:

                    fetch_single_image_with_args = partial(
                        fetch_single_image, timeout=timeout, retries=retries
                    )
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        batch_images = list(
                            executor.map(fetch_single_image_with_args, batch["image_url"], image_paths)
                        )
                    batch["image_downloaded"] = batch_images
                    logger.info(f"Fetch rate {sum(batch_images)/len(batch_images)}")

                else:
                    batch_images = []
                    for image_path in image_paths:
                        if os.path.exists(image_path):
                            try:
                                image = PIL.Image.open(image_path)
                                image = image.convert('RGB')
                                if image.size[0] > 32 and image.size[1] > 32:
                                    batch_images.append(True)
                                else:
                                    logger.warning(f"Image size too small, removed {image_path} {image}")
                                    batch_images.append(False)
                            except Exception:
                                logger.error(f"Error opening image {image_path}")
                                print(f"Error opening image {image_path}")
                                batch_images.append(False)
                        else:
                            batch_images.append(False)
                    # batch_images = [
                    #     os.path.exists(image_path) for image_path in image_paths
                    # ]
                    batch["image_downloaded"] = batch_images
                
                return batch
            
            # split_ds = split_ds.filter(lambda x: x["mime_type"] == "image/jpeg")
            
            split_ds = split_ds.map(
                get_images,
                batched=True,
                batch_size=512,
                with_indices=True,
                num_proc=128,
                fn_kwargs={
                    "num_threads": num_threads,
                },
            )
            logger.info("Image download finished.")
            valid_split_ds = split_ds.filter(lambda x: x["image_downloaded"] == True, num_proc=128)
            logger.info(f"After downloading, split {split} has {len(valid_split_ds)} entries.")

            output_data[split] = valid_split_ds
            for item in tqdm(valid_split_ds):
                image_id = item['img_id']
                image_path = item['img_path']
                output_data['images'][image_path] = {
                    "id": image_path,
                    "img_id": image_id,
                    "img_path": image_path,
                }
        
        output_data['images'] = pd.DataFrame.from_dict(output_data['images'], orient="index").set_index("id")
        output_data['images'] = Dataset.from_pandas(output_data['images'])
        print(output_data)
        logger.info(f"There are in total {len(output_data['images'])} images.")
        return output_data
        

@register_transform_functor
class PrepareImagesForWITDataFromPassages(BaseTransform):
    """
    This functor prepares images for WIT data
    1. Download images if `fetch_images` is set to True
    2. Filter and return only examples that have downloaded images
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        """
        inputs: wit_data: with train and valid HF Datasets
        """
        self.data.update(inputs)
        module_config = self.module_config

        data_paths = module_config.data_paths
        image_data_path = data_paths.image_data_path

        output_data = EasyDict({
            "images": {},
            "passages": self.data.passages,
        })

        split_ds = self.data.passages.dataset

        USER_AGENT = get_datasets_user_agent()
        num_threads = 128

        def random_userAgent():
            user_agent_list = ["Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0;) Gecko/20100101 Firefox/61.0",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
            "Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
            ]
            return random.choice(user_agent_list)
            
        def fetch_single_image(image_url, image_path, timeout=30, retries=3):
            if os.path.exists(image_path):
                # print("skipping", image_url)
                return True
            for _ in range(retries + 1):
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}
                    headers['User-Agent'] = random_userAgent()
                    response = requests.get(image_url, stream=True, timeout=timeout, headers=headers)
                    if response:
                        image = PIL.Image.open(response.raw)
                        # print('saving image to ', image, image_path)
                        image = image.save(image_path)
                        return True
                    else:
                        image = None
                except Exception:
                    image = None
            # print("result:", image_url, image)
            return False


        def get_images(batch, num_threads, timeout=30, retries=3):
            image_urls = batch["image_url"]
            image_ids = batch['image_id']
            image_paths = [
                os.path.join(image_data_path, f"{i}.jpg") for i in image_ids
            ]
            batch['img_id'] = image_ids
            batch['img_path'] = image_paths

            if module_config.fetch_images:
                fetch_single_image_with_args = partial(
                    fetch_single_image, timeout=timeout, retries=retries
                )
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    batch_images = list(
                        executor.map(fetch_single_image_with_args, batch["image_url"], image_paths)
                    )
                batch["image_downloaded"] = batch_images
                logger.info(f"Fetch rate {sum(batch_images)/len(batch_images)}")

            else:
                batch_images = [
                    os.path.exists(image_path) for image_path in image_paths
                ]
                batch["image_downloaded"] = batch_images
            
            return batch
        
        split_ds = split_ds.filter(lambda x: x["mime_type"] == "image/jpeg", num_proc=8)

        split_ds = split_ds.map(
            get_images,
            batched=True,
            batch_size=512,
            num_proc=8,
            fn_kwargs={
                "num_threads": num_threads,
            },
        )
        logger.info("Image download finished.")
        valid_split_ds = split_ds.filter(lambda x: x["image_downloaded"] == True)
        logger.info(f"After downloading, there are {len(valid_split_ds)} entries.")
        available_documents = valid_split_ds['passage_id']
        available_documents = {i: 1 for i in available_documents}
        output_data.passages.available_documents = available_documents
        output_data.passages.dataset = valid_split_ds

        for item in tqdm(valid_split_ds):
            image_id = item['img_id']
            image_path = item['img_path']
            output_data.images[image_path] = {
                "img_id": image_id,
                "img_path": image_path,
            }
            
        logger.info(f"There are in total {len(output_data.images)} images.")
        logger.info(f"There are in total {len(output_data.passages.dataset)} documents.")
        return output_data

@register_transform_functor
class LoadWITPassages(HFDatasetTransform):
    """
    This functor loads WIT passages and assign passage ids
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        """
        inputs: DatasetDict: with train and valid HF Datasets
        """
        self.data.update(inputs)
        module_config = self.module_config
        
        full_corpus_ds = [split_ds for split, split_ds in self.data.items() if split in ['train', 'valid']]
        full_corpus_ds = concatenate_datasets(full_corpus_ds)
        
        id_column = [f"WIT_{i}" for i in range(len(full_corpus_ds))]
        full_corpus_ds = full_corpus_ds.add_column(name="passage_id", column=id_column)

        def process_example(item):
            passage_content = f"title: {item['page_title']}"
            if item['section_title'] is not None:
                passage_content += f" section title: {item['section_title']}"
            if item['hierarchical_section_title'] is not None:
                passage_content += f" hierarchical section title: {item['hierarchical_section_title']}"
            if item['caption_reference_description'] is not None:
                passage_content += f" caption reference description: {item['caption_reference_description']}"
            if item['caption_attribution_description'] is not None:
                passage_content += f" caption attribution description: {item['caption_attribution_description']}"
            if item['caption_alt_text_description'] is not None:
                passage_content += f" caption alt text description: {item['caption_alt_text_description']}"
            
            passage_content += f" content: {item['context_page_description']}"

            item['passage_content'] = passage_content

            return item

        full_corpus_ds = full_corpus_ds.map(
            process_example, num_proc=32,
        )
        
        # for item in tqdm(full_corpus_ds):
        #     passage_id = item['passage_id']
        #     image_id = item['image_id']
        #     self.data.passages['id2doc'][passage_id] = item['passage_content']
        #     self.data.passages['imageid2docid'][image_id] = passage_id
        print(full_corpus_ds)
        output_data = DatasetDict({
            "passages": full_corpus_ds
        })
        return output_data




@register_transform_functor
class TruncateWITPassages(BaseTransform):
    """
    This functor loads WIT passages and assign passage ids
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        """
        inputs: wit_data: with train and valid HF Datasets
        """
        self.data.update(inputs)
        module_config = self.module_config
        truncation_length = module_config.truncation_length

        ds = self.data.passages.dataset

        # a map function to process and split the text in batch
        def process_text(batch):
            irrelevant_keys = list(batch.keys())
            irrelevant_keys.remove('passage_content')

            final_batch = {k: [] for k in batch.keys()}

            for index in range(len(batch['passage_id'])):
                
                original_text = batch['passage_content'][index]
                # print(index, original_text)
                # print('=====================')
                # if item['text'] is more than truncation length (approximately in tokens)
                # when counting the length, split the sentence by both space and newline
                
                # split the text by newline
                text_list = original_text.split(' ')
                # remove empty string
                text_list = [text for text in text_list if text != ""]
                # split the text by space
                text_list = [text.split() for text in text_list]
                # flatten the list
                text_list = [item for sublist in text_list for item in sublist]
                # print("flattened text_list: ", text_list)
                # get length of the text
                text_length = len(text_list)
                # print("text_length: ", text_length)
                # if the text is longer than truncation length, split the text at only "\n"
                if text_length > truncation_length:
                    # split the text by newline
                    text_list = original_text.split(' ')
                    # remove empty string
                    text_list = [text.strip() for text in text_list if text != ""]

                    # add items in text)_list gradually until the tuncation_length is met
                    # if the truncation_length is not met, add the next item in text_list
                    # if the truncation_length is met, refresh the final_text and add the next item in text_list
                    final_text = ""

                    for text in text_list:
                        # print("final_text: ", final_text)
                        if len(final_text.split()) + len(text.split()) < truncation_length:
                            final_text += text + ' '
                        else:
                            final_batch['passage_content'].append(final_text)
                            for k in irrelevant_keys:
                                final_batch[k].append(batch[k][index])
                            
                    # if the final_text is not empty, add it to the final_batch
                    if final_text.strip() != "":
                        final_batch['passage_content'].append(final_text)
                        for k in irrelevant_keys:
                            final_batch[k].append(batch[k][index])
                    
                else:
                    for k in irrelevant_keys + ['passage_content']:
                        final_batch[k].append(batch[k][index])

            return final_batch

        logger.info(f"Loaded {len(ds)} passages")
        ds = ds.map(process_text, batched=True, num_proc=32)
        logger.info(f"After truncation, there are {len(ds)} passages")
        ds = ds.rename_column('passage_id', 'original_id')
        id_column = [f"WIT_{i}" for i in range(len(ds))]
        ds = ds.add_column('passage_id', id_column)
        ds = ds.remove_columns(['original_id'])
        self.data.passages.dataset = ds

        output_data = EasyDict({
            "passages": self.data.passages
        })
        return output_data



@register_transform_functor
class PrepareWITDataForRetrieval(HFDatasetTransform):
    """
    This functor conducts the following operations
    1. Gather all passages
    2. Create train/val/test examples
    3. Assign positive passages to each example, associate with images
    4. Pack examples, split train and valid
    input:
    {'images': Dataset({
        features: ['img_id', 'img_path', '__index_level_0__'],
        num_rows: 179080
    }), 'train': Dataset({
        features: ['language', 'page_url', 'image_url', 'page_title', 'section_title', 'hierarchical_section_title', 'caption_reference_description', 'caption_attribution_description', 'caption_alt_text_description', 'mime_type', 'original_height', 'original_width', 'is_main_image', 'attribution_passes_lang_id', 'page_changed_recently', 'context_page_description', 'context_section_description', 'image_id', 'img_id', 'img_path', 'image_downloaded'],
        num_rows: 189611
    }), 'valid': Dataset({
        features: ['language', 'page_url', 'image_url', 'page_title', 'section_title', 'hierarchical_section_title', 'caption_reference_description', 'caption_attribution_description', 'caption_alt_text_description', 'mime_type', 'original_height', 'original_width', 'is_main_image', 'attribution_passes_lang_id', 'page_changed_recently', 'context_page_description', 'context_section_description', 'image_id', 'img_id', 'img_path', 'image_downloaded'],
        num_rows: 1520
    }), 'passages': Dataset({
        features: ['language', 'page_url', 'image_url', 'page_title', 'section_title', 'hierarchical_section_title', 'caption_reference_description', 'caption_attribution_description', 'caption_alt_text_description', 'mime_type', 'original_height', 'original_width', 'is_main_image', 'attribution_passes_lang_id', 'page_changed_recently', 'context_page_description', 'context_section_description', 'image_id', 'passage_id', 'passage_content'],
        num_rows: 551755
    })}
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        for i in inputs:
            self.data.update(i)
        module_config = self.module_config

        iglue_test_file = module_config.get("iglue_test_file", None)

        output_data = DatasetDict({})

        passages_ds = self.data.passages
        # keep only relevant terms and convert to dict
        passages_ds = passages_ds.select_columns(["original_data_id", "passage_id", "passage_content"])
        passages_df = passages_ds.to_pandas()
        # print(passages_df)
        # print("before grouping", len(passages_df))
        # tqdm.pandas()
        # def agg_func(x):
        #     all_passage_ids = x.values.tolist()
        #     return all_passage_ids
        # passage_df = passages_df.groupby("image_id").progress_aggregate(agg_func)#.progress_aggregate(pos_item_ids=pd.NamedAgg(column="passage_id", aggfunc=agg_func))
        # print(passage_df)
        # print("after grouping", len(passages_df))
        # input()
        # imageid2docid = passages_ds.to_pandas().drop_duplicates(["image_id"]).set_index("image_id").to_dict(orient="index")

        # def process_example(example):
        #     pos_item_ids = [imageid2docid[example['image_id']]['passage_id']]
        #     example['pos_item_ids'] = pos_item_ids
            
        #     return example

        for split in ['train', 'valid', 'test']:
            split_ds = self.data[split]

            if iglue_test_file is not None and split == 'valid' and (not self.use_dummy_data):
                # load json line file
                # with open(iglue_test_file, 'r') as f:
                #     iglue_test_data = [json.loads(line) for line in f.readlines()]
                # convert to Huggingface dataset
                iglue_test_set = pd.read_json(iglue_test_file, lines=True)
                iglue_test_set = Dataset.from_pandas(iglue_test_set)
                logger.info(f"There are {len(iglue_test_set)} examples in iglue test set.")
                print(split_ds)
                print(iglue_test_set)
                iglue_all_page_urls_list = iglue_test_set['page_url']
                iglue_all_image_urls_list = iglue_test_set['image_url']
                iglue_all_captions_list = iglue_test_set['caption_reference_description']
                
                iglue_match_tuples = [(i, j, k) for i, j, k in zip(iglue_all_page_urls_list, iglue_all_image_urls_list, iglue_all_captions_list)]
                iglue_match_dict = {i: 1 for i in iglue_match_tuples}

                # keep only examples that are in iglue test set
                # split_ds = split_ds.filter(lambda x: x['image_url'] in iglue_all_image_urls_list)
                split_ds = split_ds.filter(lambda x: iglue_match_dict.get((x['page_url'], x['image_url'], x['caption_reference_description']), None) is not None)
                logger.info(f"After filtering, there are {len(split_ds)} examples in split {split}.")
                

            # split_ds = split_ds.map(process_example)
            # merge with passages according to image_id
            split_df = split_ds.to_pandas()

            tqdm.pandas()
            def agg_func(x):
                # concatenate all items
                # only aggregate necessary columns
                if x.name not in ["passage_id", "passage_content", "original_data_id"]:
                    if len(set(x.values.tolist())) != 1:
                        print(x)
                        input()
                    assert len(set(x.values.tolist())) == 1
                    return x.values.tolist()[0]
                else:
                    return [i for i in x.values.tolist()]
            # print("before groupping", split_df)
            # split_df = split_df.groupby("image_id").progress_aggregate(agg_func)
            # print(split_df)
            # input()
            print('joinning...')
            split_df = split_df.join(passages_df.set_index("original_data_id"), on="original_data_id", how="left")
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     print(split_df.iloc[:3])
            
            
            # keep only relevant data
            # split_ds = split_ds.select_columns(["image_id", "passage_content", "page_title", "page_url", "img_id", "img_path"])
            split_df = split_df[["original_data_id", "image_id", "passage_id", "passage_content", "img_id", "img_path"]]
            # split_ds = split_ds.rename_column("image_id", "img_id")
            # split_ds = split_ds.rename_column("image_path", "img_path")

            # df = df.drop_duplicates(["img_id"])

            # Merge entries that have the same passage ids
            print("before grouping", len(split_df))
            
            split_df = split_df.groupby("image_id").progress_aggregate(agg_func)#.progress_aggregate(pos_item_ids=pd.NamedAgg(column="passage_id", aggfunc=agg_func))
            
            print("after grouping", len(split_df))
            
            split_ds = Dataset.from_pandas(split_df)

            # rename the merged columns
            split_ds = split_ds.rename_column("passage_id", "pos_item_ids")
            split_ds = split_ds.rename_column("passage_content", "pos_item_contents")

            # duplicate img_id as question_id for future use
            split_ds = split_ds.add_column("question_id", split_ds["img_id"])

            output_data[split] = split_ds
        
        logger.info('[Data Statistics] passages {}'.format(
                        len(self.data['passages'])))
        logger.info('[Data Statistics] WIT data train entries {}'.format(
                        len(output_data['train'])))
        logger.info('[Data Statistics] WIT data valid entries {}'.format(
                        len(output_data['valid'])))
        logger.info('[Data Statistics] WIT data test entries {}'.format(
                        len(output_data['test'])))
        print(output_data)
        return output_data




@register_transform_functor
class IndexWITPassagesWithElasticSearch(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function indexes passages into ElasticSearch
        {
            index_name: "wikipedia",
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        module_config = self.module_config

        ######################
        #  Get weak supervision annotations
        ######################
        
        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]

        es = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.environ["ELASTIC_CA_CERTS"],
            basic_auth=("elastic", ELASTIC_PASSWORD)
        )

        # Successful response!
        es.info()

        # ds = self.data.passages.dataset
        # index_name = module_config.index_name
        

        # # delete the current index
        # if es.indices.exists(index=index_name):
        #     es.indices.delete(index=index_name)

        # def all_actions():
        #     for index, i in tqdm(enumerate(ds), total=len(ds)):
        #         # doc = {
        #         #     'title': i['title'],
        #         #     'text': i['text'],
        #         # }
        #         # resp = es.index(index="wikipedia", id=i['id'], document=doc)
        #         action = {
        #             '_op_type': "index",
        #             '_index': index_name,
        #             '_id': i['passage_id'],
        #             '_source': {
        #                 'title': i['page_title'],
        #                 'text': i['passage_content'],
        #             }
        #         }
        #         yield action

        # # all_actions = []
        # # for index, i in tqdm(enumerate(ds), total=len(ds)):
        # #     # doc = {
        # #     #     'title': i['title'],
        # #     #     'text': i['text'],
        # #     # }
        # #     # resp = es.index(index="wikipedia", id=i['id'], document=doc)
        # #     action = {
        # #         '_op_type': "index",
        # #         '_index': index_name,
        # #         '_id': i['passage_id'],
        # #         '_source': {
        # #             'title': i['page_title'],
        # #             'text': i['passage_content'],
        # #         }
        # #     }
        # #     all_actions.append(action)
        # #     # if index > 1000:
        # #     #     break
        
        # res = helpers.bulk(es, all_actions(), request_timeout=120)

        # # batch_size = 10000
        # # n_actions = len(all_actions)
        # # for i in range(0, n_actions, batch_size):
        # #     print(f"processing...{i}-{i+batch_size}/{n_actions}")
        # #     actions = all_actions[i:min(i+batch_size, n_actions)]
            
        # #     res = helpers.bulk(es, actions, request_timeout=120)
        # #     pprint(res)
        # #     print(f"number of success {res[0]}")
        # #     if res[0] != batch_size:
        # #         print("errors", res[1])
        # logger.info(f"Successfully indexed {len(ds)} items into ES.")

        return self.data



@register_transform_functor
class RemoveWITPassagesWithoutImages(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function removes passages that don't have images
        """
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        full_ds = self.data.passages.dataset

        # filter out docs that don't have answers
        self.data.passages.full_dataset = full_ds
        self.data.passages.dataset = full_ds.filter(lambda x: self.data.passages.available_documents.get(x['passage_id'], None) is not None, load_from_cache_file=False)
        logger.info(f"reducing #docs (full) to {len(self.data.passages.full_dataset)}")
        return self.data

@register_transform_functor
class PrepareWITPassageAnnotations(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function prepares Wikipedia passage annotations (pseudo labels)
        {
            "annotations_path": {
                "train": "..",
                "valid": "..",
                "test": "..",
            },
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        module_config = self.module_config

        ######################
        #  Get weak supervision annotations
        ######################
        self.data.okvqa_data_with_dpr_output = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
        })
        self.data.passages.annotations = EasyDict({})

        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]

        es = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.environ["ELASTIC_CA_CERTS"],
            basic_auth=("elastic", ELASTIC_PASSWORD)
        )

        # Successful response!
        es.info()

        ds = self.data.passages.dataset
        index_name = module_config.index_name
        
        def search_for_a_string(query):
            resp = es.search(index=index_name, query={
                "multi_match" : {
                    "query": query,
                    "fields": ["title", "text"],
                    "type": "phrase",
                }
            }, timeout="60s")
            # pprint(resp)
            # print("Got %d Hits:" % resp['hits']['total']['value'])
            # for hit in resp['hits']['hits']:
            #     print(hit)
            return resp
        
        # def match_a_question(query):
        #     resp = es.search(index="wikipedia", query={
        #         "match" : {
        #             "text": {
        #                 "query": query,
        #                 "fuzziness": "AUTO"
        #             }
        #         }
        #     })
        #     pprint(resp)
        #     return resp


        from thefuzz import fuzz
        from thefuzz import process

        available_documents = {}

        # reindex the dataset id column and convert to strings. TODO: move this to the previous stage.
        # ds = ds.rename_column('id', 'original_id')
        # list_ids = [str(i) for i in range(len(ds))]
        # ds = ds.add_column('id', list_ids)
        # ds = ds.remove_columns(['original_id'])
        
        for data_split in ['train', 'test']:

            for item in tqdm(self.data.okvqa_data[data_split].data_items):
                question_id = item.question_id
                
                # Search ES and return all passages containing answers
                passages_match_answer = []

                for answer in set(item.answers):
                    passages_match_answer.extend(
                        search_for_a_string(answer)['hits']['hits']
                    )
                
                # print("answers", item.answers)
                
                for i in passages_match_answer:
                    available_documents[str(i['_id'])] = 1

                # Rate passages according to query information (e.g. question, objects in the image)
                choices = {
                    i['_id']: i['_source']['text'] for i in passages_match_answer
                }
                
                element_string_in_query = f'{item.gold_answer} {item.gold_answer} {item.question} {item.img_caption["caption"]}'
                
                for obj in item.objects:
                    element_string_in_query += f" {obj['class'].strip().lower()}"

                res = process.extract(element_string_in_query, choices, limit=10, scorer=fuzz.token_set_ratio)
                # print("rating", choices, 'according to', item.question)
                # input()
                # drop lowest score item to further filter down the annotations
                if len(res) > 0:
                    lowest_score = res[-1][1]
                    res = [i for i in res if i[1] > lowest_score]
                else:
                    res = []
                
                knowledge_collection = [
                    i[2] for i in res
                ]
                self.data.passages.annotations[str(question_id)] = {
                    'passages': knowledge_collection,
                }
                # print(f"question {question_id} has {len(knowledge_collection)} passages")
        
        print(f"total #docs {len(ds)}")
        print(f"total #docs with answers {len(available_documents)}")
        
        # full_ds = ds
        # self.data.passages.full_dataset = full_ds
        
        # # filter out docs that don't have answers
        # ds = ds.filter(lambda x: available_documents.get(x['id'], None) == 1)
        # logger.info(f"reducing #docs to {len(ds)}")
        

        # self.data.passages.dataset = ds

        self.data.passages.available_documents = available_documents

        for data_split in ['train', 'test']:
            self.data.okvqa_data_with_dpr_output[data_split] = EasyDict({})
            self.data.okvqa_data_with_dpr_output[data_split].data_items = []
            
            missing_entries = []
            missing_data = []

            for item in self.data.okvqa_data[data_split].data_items:
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
                        # related_knowledge = [1]
                        if self.config.mode == 'train':
                            continue
                        else:
                            # in testing mode, all samples must be used
                            related_knowledge = [1]
                
                knowledge_item = EasyDict(dict(item))
                knowledge_item['pos_item_ids'] = related_knowledge
                # knowledge_item['pos_item_contents'] = [
                #     self.data.passages.id2doc[str(passage_id)] for passage_id in related_knowledge
                # ]
                self.data.okvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                

            if len(missing_entries) > 0:
                logger.warning(f"{len(missing_entries)} questions (split {data_split}) not found in knowledge. \n {missing_entries}")
            if len(missing_data) > 0:
                logger.warning(f"{len(missing_data)} questions (split {data_split}) has no annotations. \n {missing_data}")
            

            # Load item data into lookup with question_id as index
            logger.info('Indexing data items...')
            
            for item in tqdm(self.data.okvqa_data_with_dpr_output[data_split].data_items):
                question_id = item['question_id']
                self.data.okvqa_data_with_dpr_output.lookup[str(question_id)] = item
            
            # Report statistics
            logger.info('[Data statistics] loaded with knowledge data split: {}  entries: {}'.format(
                data_split,
                len(self.data.okvqa_data_with_dpr_output[data_split].data_items)))

        output_data = EasyDict(
            okvqa_data_with_dpr_output = self.data.okvqa_data_with_dpr_output,
            passages=self.data.passages,
        )
        return output_data



@register_transform_functor
class ReduceWITPassagesSize(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function loads GoogleSearch Corpus annotations
        {
            "annotations_path": {
                "train": "..",
                "valid": "..",
                "test": "..",
            },
        },
        """
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        full_ds = self.data.passages.dataset

        # # Prepare ElasticSearch
        # from elasticsearch import Elasticsearch, helpers

        # # Password for the 'elastic' user generated by Elasticsearch
        # ELASTIC_PASSWORD = "EKJ8kIbmMg=e6sI5vikP"

        # es = Elasticsearch(
        #     "https://localhost:9200",
        #     ca_certs="/home/wl356/cvnlp_rds/wl356/elasticsearch-8.7.0/config/certs/http_ca.crt",
        #     basic_auth=("elastic", ELASTIC_PASSWORD)
        # )

        # # Successful response!
        # es.info()

        # ds = self.data.passages.dataset
        # index_name = module_config.index_name

        # # filter out docs that don't have answers
        # # part dataset
        # self.data.passages.dataset = ds.filter(lambda x: self.data.passages.available_documents.get(x['id'], None) is not None, load_from_cache_file=False)
        # logger.info(f"reducing #docs (train) to {len(self.data.passages.dataset)}")
        
        # def search_for_a_title(query):
        #     resp = es.search(index=index_name, query={
        #         "multi_match" : {
        #             "query": query,
        #             "fields": ["title"],
        #             "type": "phrase",
        #         }
        #     }, timeout="60s")
        #     # pprint(resp)
        #     # print("Got %d Hits:" % resp['hits']['total']['value'])
        #     # for hit in resp['hits']['hits']:
        #     #     print(hit)
        #     return resp
        
        # for json_path in self.module_config.include_concepts:
        #     with open(json_path, 'r') as f:
        #         concepts = json.load(f)['label_to_idx']
        #     print(f"loaded {len(concepts)} concepts from {json_path}")
            
        #     for concept in concepts.keys():
        #         resp = search_for_a_title(concept)
        #         if resp['hits']['total']['value'] == 0:
        #             logger.warning(f"not found: {concept}")
        #         else:
        #             for i in resp['hits']['hits']:
        #                 logger.info(f"found: {concept} passage title {i['_source']['title']} in {i['_id']}")
        #                 doc_id = str(i['_id'])
        #                 self.data.passages.available_documents[doc_id] = 1

        # filter out docs that don't have answers
        self.data.passages.full_dataset = full_ds
        self.data.passages.dataset = full_ds.filter(lambda x: self.data.passages.available_documents.get(x['passage_id'], None) is not None)
        logger.info(f"reducing #docs (full) to {len(self.data.passages.full_dataset)}")
        logger.info(f"reducing #docs to {len(self.data.passages.dataset)}")

        return self.data



@register_transform_functor
class ReduceWITImagesSize(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function reduces image dict to images that appear in the passages
        """
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        full_ds = self.data.passages.dataset
        images = self.data.images
        
        new_images = {}
        
        # select a column and convert to a dict
        available_images = full_ds['image_id']
        # convert to dict
        available_images = {k: 1 for k in available_images}

        imgId2path = {}

        for k, v in tqdm(images.items()):
            if available_images.get(v['img_id'], None) is not None:
                new_images[k] = v
                imgId2path[v['img_id']] = v['img_path']
        
        logger.info(f"reducing #images (full {len(images)}) to {len(new_images)}")
        
        return {
            'images': new_images,
            'imgId2path': imgId2path,
        }


@register_transform_functor
class ConcatenateImageCorpus(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function reduces image dict to images that appear in the passages
        """

        full_images = {}
        full_image_dataset_with_embeddings = []

        for input_data in inputs:
            if 'images' in input_data.keys():
                print("input images:", len(input_data.images))
                full_images.update(input_data.images)
            if 'image_dataset_with_embeddings' in input_data.keys():
                print("input image_dataset_with_embeddings:", len(input_data.image_dataset_with_embeddings))
                full_image_dataset_with_embeddings.append(input_data.image_dataset_with_embeddings)
                print(input_data.image_dataset_with_embeddings[:1])
        
        # concatenate image dataset
        full_image_dataset_with_embeddings = concatenate_datasets(full_image_dataset_with_embeddings)

        print("full:", len(full_images), len(full_image_dataset_with_embeddings))
        
        return {
            'images': full_images,
            'image_dataset_with_embeddings': full_image_dataset_with_embeddings,
        }

        
@register_transform_functor
class SplitWITPassagesForLargeScaleTraining(HFDatasetTransform):
    """
    This functor split the full passage corpus into a full corpus and a smaller corpus for validation.
    This validation corpus contains all validation passages and a subset of training passages.
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        """
        inputs: DatasetDict: with train and valid HF Datasets
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        module_config = self.module_config
        full_corpus_ds = self.data.passages

        # get all validation passages + test passages
        valid_ds = self.data.valid
        test_ds = self.data.test
        valid_passage_ids = valid_ds['pos_item_ids'] + test_ds['pos_item_ids']
        valid_passage_ids = [item for sublist in valid_passage_ids for item in sublist]
        print("valid_passage_ids", len(valid_passage_ids))

        # convert into a dict
        valid_passage_ids = {k: 1 for k in valid_passage_ids}

        ds = full_corpus_ds.filter(lambda x: valid_passage_ids.get(x['passage_id'], None) is not None)
        # remaining_ds = full_corpus_ds.filter(lambda x: valid_passage_ids.get(x['passage_id'], None) is None)
        print("passages associated with validation/test samples", len(ds))
        # add random passages from training set
        # if self.use_dummy_data:
        #     select_num = min(len(remaining_ds), 100)
        # else:
        #     select_num = min(len(remaining_ds), 100000)
        
        # ds = concatenate_datasets([ds, remaining_ds.shuffle(seed=42).select(range(select_num))])
        print("all selected passages for testing:", len(ds))

        output_data = DatasetDict({
            "train_passages": full_corpus_ds,
            "valid_passages": ds,
            "test_passages": ds,
        })

        return output_data