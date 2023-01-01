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

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from data_loader_manager.datasets import *

from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class DataLoaderOKVQA(DataLoaderWrapper):
    '''
    Data loader for OKVQA dataset
    '''

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)


    def LoadVinVLFeatures(self, module_config):
        '''
        Load vinvl features
        {
          "type": "LoadVinVLFeatures", "option": "default", 
          "config": {
                "train": "..",
                "test": "..",
            },
        },
        '''
        ######################
        #   Read VinVL data
        ######################
        csv.field_size_limit(100000000)
        # print(csv.field_size_limit())

        self.data.vinvl_features = load_cached_data(self.config, 'vinvl_feature_preprocessed')
        if not self.data.vinvl_features:
            self.data.vinvl_features = EasyDict()
            for data_split in ['train', 'test']:
                # Read pre-extracted features
                VinVL_feature_file = module_config.config[data_split]
                logger.info(f'Reading: {VinVL_feature_file}')
                with open(VinVL_feature_file, 'r') as csv_file:
                    
                    count_total = sum(1 for row in csv_file)
                    csv_file.seek(0)
                    read_tsv = csv.reader(csv_file, delimiter="\t")

                    for row in tqdm(read_tsv, total=count_total):
                        image_key, prediction = row
                        prediction = json.loads(prediction)
                        # print(image_key)
                        # print(prediction.keys())
                        self.data.vinvl_features[image_key] = prediction
                        # for obj in prediction['objects']:
                        #     print(obj['rect'])
                        #     print(obj['class'])
                        #     print(obj['conf'])
                        #     print(obj['attributes'])
                        # input()
            save_cached_data(self.config, self.data.vinvl_features, 'vinvl_feature_preprocessed')

        logger.info('[Data Statistics] VinVL features {}'.format(
                        len(self.data.vinvl_features)))


    def LoadGoogleOCRFeatures(self, module_config):
        '''
        Load google ocr features
        {
          "type": "LoadGoogleOCRFeatures", "option": "default",
          "config": {
                "combine_with_vinvl": True,
                "train": "..",
                "test": "..",
            },
        },
        '''

        ######################
        #   Read OCR data and combine with VinVL features
        ######################
        self.data.ocr_features = load_cached_data(self.config, 'ocr_feature_preprocessed')
        if not self.data.ocr_features:
            self.data.ocr_features = EasyDict()
            for data_split in ['train', 'test']:
                # Read pre-extracted features
                ocr_feature_path = module_config.config[data_split]
                logger.info(f'Reading data in: {ocr_feature_path}')
                for image_key, prediction in tqdm(self.data.vinvl_features.items()):
                    ocr_feature_file = os.path.join(ocr_feature_path, '{}_ocr.json'.format(image_key))
                    if os.path.exists(ocr_feature_file):
                        with open(ocr_feature_file, 'r') as json_fp:
                            ocr_data = json.load(json_fp)
                        self.data.ocr_features[image_key] = ocr_data
                    else:
                        logger.debug(f'Skipping: {ocr_feature_file}')
            
            save_cached_data(self.config, self.data.ocr_features, 'ocr_feature_preprocessed')

        count = 0
        for img_key, annotation in self.data.ocr_features.items():
            if len(annotation['filtered_text_annotations']) > 0:
                count += 1
        logger.info('[Data Statistics] OCR features {}, {} has annotations.'.format(len(self.data.ocr_features), count))
        

        if module_config.config['combine_with_vinvl']:

            ## After reading OCR, check if VinVL feature is linked
            def PolyArea(x,y):
                return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            
            
            first_prediction = next(iter(self.data.vinvl_features.items()))[1]
            if 'ocr' not in first_prediction.keys():
                logger.info('OCR feature not detected in VinVL feature dict...Recomputing.')
                for image_key, prediction in tqdm(self.data.vinvl_features.items()):
                    text_annotations = self.data.ocr_features[image_key]['filtered_text_annotations']
                    count = 0
                    for text_annoation in text_annotations:
                        description = text_annoation['description']
                        description = description.replace('\n', " ") # remove line switching
                        vertices = text_annoation['vertices']
                        # print(image_key)
                        # print(description, vertices)
                        xs, ys = list(zip(*vertices))
                        area = PolyArea(xs, ys)
                        
                        for obj in prediction['objects']:
                            obj_class = obj['class']
                            obj_rect = obj['rect']
                            xmin, ymin, xmax, ymax = obj_rect
                            obj_area = (ymax - ymin) * (xmax - xmin)
                            # Check if the description is inside the object
                            np_vertices = np.array(vertices)
                            within_obj = np.all(np_vertices[:, 0] >= xmin) and np.all(np_vertices[:, 0] <= xmax)\
                                and np.all(np_vertices[:, 1] >= ymin) and np.all(np_vertices[:, 1] <= ymax)
                            if within_obj:
                                score = area / obj_area
                            else:
                                score = 0.0
                            if score > 0:
                                count += 1
                                obj.setdefault('ocr', []).append({
                                    'text': description,
                                    'score': score,
                                })
                    prediction['ocr'] = count
                        
                logger.info('OCR feature computed into VinVL feature dict...saving..')
                save_cached_data(self.config, self.data.vinvl_features, 'vinvl_feature_preprocessed')
            else:
                logger.info('OCR feature detected in VinVL feature dict...skipping..')


    def LoadOscarCaptionFeatures(self, module_config):
        '''
        Load oscar caption features
        {
          "type": "LoadOscarCaptionFeatures", "option": "default",
          "config": {
                "train": "..",
                "valid": "..",
                "test": "..",
            },
        },
        '''
        ######################
        #   Read caption data
        ######################
        self.data.caption_features = EasyDict()
        for caption_file_path in module_config.config.values():
            with open(caption_file_path, "r") as f:
                caption_data = json.load(f)
                self.data.caption_features.update(caption_data)
        print('[Data Statistics] Caption features {}'.format(len(self.data.caption_features)))


    def LoadOKVQAData(self, module_config):
        '''
        Load vqa data into self.data.okvqa_data
        {
          "type": "LoadOKVQAData", "option": "default",
          "config": {
            "vqa_data_path": {
                "question_files":{
                    "train": "..",
                    "test": "..",
                },
                "annotation_files": {
                    "train": "..",
                    "test": "..",
                },
            },
            "image_data_path": {
                "train": "..",
                "valid": "..",
            },
        },
        '''
        ######################
        #   Read OK-VQA data
        ######################
        def most_frequent(List):
            return max(set(List), key = List.count)
        
        answer_candidate_list = []
        vqa_helpers = EasyDict({
            'train': VQA(module_config.config.vqa_data_path.annotation_files.train, 
                            module_config.config.vqa_data_path.question_files.train),
            'test': VQA(module_config.config.vqa_data_path.annotation_files.test, 
                            module_config.config.vqa_data_path.question_files.test),
        })
        
        self.data.okvqa_data = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
            'vqa_helpers': vqa_helpers,
        })
        
        for data_split, vqa_helper in vqa_helpers.items():
            vqa_helper.createIndex()
            vqa_helper.info()

            # For each data split, prepare dataset
            self.data.okvqa_data[data_split] = load_cached_data(self.config, '{}_data_preprocessed'.format(data_split))
            if not self.data.okvqa_data[data_split]:
                # This split data is not cached
                self.data.okvqa_data[data_split] = EasyDict({}) # re-initialise
                # Create list of images from helper
                img_data_path = module_config.config.image_data_path[data_split]
                img_list = []
                for imgId in vqa_helper.imgToQA.keys():
                    dataSubType = vqa_helper.dataSubType
                    imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
                    img_path = os.path.join(img_data_path, imgFilename)
                    img_list.append((imgId, img_path))
                    if self.config.data_loader.dummy_dataloader:
                        # Load only a few samples for testing
                        if len(img_list) > 20:
                            break
                
                # Create entries for each question and related answers
                self.data.okvqa_data[data_split].data_items = []
                for imgId, img_path in tqdm(img_list):
                    # avoid error in splitting: must remove ".." in "../path/to/file"
                    # img_key = img_p.replace('..', '').split('.')[0].split('_')[-1]
                    img_key = imgId
                    img_key_str = str(img_key)
                    img_caption = self.data.caption_features.get(img_key_str, None)
                    if img_caption is not None: 
                        img_caption = img_caption[0] 
                    else: 
                        logger.debug('No caption found for {}!'.format(img_key))
                    
                    img_key_full = str(img_key).zfill(12)
                    img = cv2.imread(img_path)
                    # img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
                    
                    # Read predictions from VinVL features
                    # print(self.data.vinvl_features.keys()[:10])
                    # print(self.data.vinvl_features[img_key_full])
                    # VinVL_prediction = self.data.vinvl_features.get(img_key_full, None)
                    # if not VinVL_prediction:
                    #     logger.print('img', img_key_full, 'no match found!', mode='error')
                    
                    # DEBUG
                    # prediction = self.data.vinvl_features.get(img_key_full, None)
                    # b_imgs = []
                    # for obj in prediction['objects']:
                    #     print(obj['rect']) # min x min y max x max y
                    #     minx, miny, maxx, maxy = [int(x) for x in obj['rect']]
                    #     # img: H x W x C
                    #     # print(img[miny:maxy, minx:maxx])

                    #     #  H x W x C --> C x H x W
                    #     # b_imgs.append(torch.Tensor(img[miny:maxy, minx:maxx]).permute(2,0,1))
                    #     print(obj['class'])

                    #     # with open(os.path.join(self.config.imgs_path, img_key_full+'.jpg'), 'w') as img_fp:
                    #     cv2.imwrite(os.path.join(self.config.imgs_path, img_key_full+'.jpg'), img[miny:maxy, minx:maxx])
                    #     input()
                    # input()

                    related_question_ids = vqa_helper.getQuesIds(imgIds=[imgId])
                    related_answers = vqa_helper.loadQA(ids=related_question_ids)
                    related_question_and_answers = vqa_helper.returnQA(related_answers)
                    
                    for question_and_answer in related_question_and_answers:
                        # For each question and related answers, create an entry
                        entry_data = EasyDict()
                        entry_data.answers = list(question_and_answer['answers'].values())
                        entry_data.answers = [answer for answer in entry_data.answers if answer != '']
                        entry_data.gold_answer = most_frequent(entry_data.answers)
                        entry_data.question = question_and_answer['question']
                        entry_data.question_id = question_and_answer['question_id']
                        entry_data.img_path = img_path
                        entry_data.img_key_full = img_key_full
                        entry_data.img_key = img_key
                        entry_data.img = img
                        entry_data.img_caption = img_caption
                        self.data.okvqa_data[data_split].data_items.append(entry_data)

                        # Collect answer candidates for evaluation
                        for ans in list(question_and_answer['answers'].values()):
                            if ans not in answer_candidate_list:
                                answer_candidate_list.append(ans)
                                # if data_split == 'test':
                                #     print(ans, 'is added from test set!')
                
                # After building the data split, save to cache
                save_cached_data(self.config, self.data.okvqa_data[data_split], '{}_data_preprocessed'.format(data_split))

            for entry_data in self.data.okvqa_data[data_split].data_items:
                self.data.okvqa_data['lookup'][str(entry_data.question_id)] = entry_data

            

            # Report statistics
            logger.info('[Data statistics] split: {}  entries: {}'.format(
                data_split,
                len(self.data.okvqa_data[data_split].data_items)))

        # Save answer candidate list
        self.data.okvqa_data.answer_candidate_list = answer_candidate_list

        self.data.vqa_data = self.data.okvqa_data

    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'data': self.data.vqa_data.train,
            'vinvl_features': self.data.vinvl_features,
            'ocr_features': self.data.ocr_features,
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
        # train_sampler = SequentialSampler(self.train_dataset)
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            # num_workers=8,
        )
        # for i in self.train_dataloader:
        #     print(i)
        #     input()
        
        test_dataset_dict = {
            'data': self.data.vqa_data.test,
            'vinvl_features': self.data.vinvl_features,
            'ocr_features': self.data.ocr_features,
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
            batch_size=self.config.valid.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            # num_workers=4,
        )
        # for i in self.test_dataloader:
        #     print(i)
        #     input()

        # input_text_sequences = []
        # output_text_sequences = []
        # answers = []
        # gold_answers = []
        # question_ids = []
        # import time
        # for item in tqdm(self.train_dataloader):
        #     input_text_sequences += item['input_text_sequences']
        #     output_text_sequences += item['output_text_sequences']
        #     question_ids += item['question_ids']
        #     gold_answers += item['gold_answers']
        #     answers += item['answers']
        #     # time.sleep(0.3)
        #     del item
        # print(input_text_sequences[:10])
        # print(output_text_sequences[:10])
        # print(len(input_text_sequences), len(output_text_sequences))
        # with open('./train.source', 'w') as f:
        #     f.writelines("{}\t{}\n".format(id, text) for id, text in zip(question_ids, input_text_sequences))
        #     f.close()
        # with open('./train.target', 'w') as f:
        #     f.writelines("{}\t{}\n".format(gold_answer, '\t'.join(answer_list)) for gold_answer, answer_list in zip(gold_answers, answers))
        #     f.close()
        
        # input_text_sequences = []
        # output_text_sequences = []
        # answers = []
        # gold_answers = []
        # question_ids = []
        # import time
        # for item in tqdm(self.test_dataloader):
        #     input_text_sequences += item['input_text_sequences']
        #     output_text_sequences += item['output_text_sequences']
        #     question_ids += item['question_ids']
        #     gold_answers += item['gold_answers']
        #     answers += item['answers']
        #     del item
        #     # time.sleep(0.3)
        # print(input_text_sequences[:10])
        # print(output_text_sequences[:10])
        # print(len(input_text_sequences), len(output_text_sequences))
        # with open('./test.source', 'w') as f:
        #     f.writelines("{}\t{}\n".format(id, text) for id, text in zip(question_ids, input_text_sequences))
        #     f.close()
        # with open('./test.target', 'w') as f:
        #     f.writelines("{}\t{}\n".format(gold_answer, '\t'.join(answer_list)) for gold_answer, answer_list in zip(gold_answers, answers))
        #     f.close()
        # with open('./val.source', 'w') as f:
        #     f.writelines("{}\t{}\n".format(id, text) for id, text in zip(question_ids, input_text_sequences))
        #     f.close()
        # with open('./val.target', 'w') as f:
        #     f.writelines("{}\t{}\n".format(gold_answer, '\t'.join(answer_list)) for gold_answer, answer_list in zip(gold_answers, answers))
        #     f.close()
        # input('finished!')
        logger.info('[Data Statistics]: training data loader: {};  test data loader: {}'.format(
                                len(self.train_dataloader), 
                                len(self.test_dataloader)))




