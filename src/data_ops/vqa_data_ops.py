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
import math
import psutil

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval

from transformers import AutoImageProcessor, CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
from torchvision import transforms
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets
import PIL

from models.custom_clip_processor import CustomCLIPImageProcessor



@register_transform_functor
class LoadVinVLFeatures(BaseTransform):
    """
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, *args, **kwargs):   
        
        module_config = self.module_config

        ######################
        #   Read VinVL data
        ######################
        csv.field_size_limit(100000000)
        # print(csv.field_size_limit())
        
        self.data.vinvl_features = EasyDict()
        for data_split in module_config.VinVL_features.keys():
            # Read pre-extracted features
            VinVL_feature_file = module_config.VinVL_features[data_split]
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
                    for obj in prediction['objects']:
                        obj.pop('feature', None)
                    
                    self.data.vinvl_features[image_key] = prediction
                    # for obj in prediction['objects']:
                    #     pprint(obj)
                    #     # print(obj['rect'])
                    #     # print(obj['class'])
                    #     # print(obj['conf'])
                    #     # print(obj['attributes'])
                    #     input()
            # if self.use_dummy_data:
            #     break
            
        logger.info('[Data Statistics] VinVL features {}'.format(
                        len(self.data.vinvl_features)))
        
        return self.data


@register_transform_functor
class LoadOscarCaptionFeatures(BaseTransform):
    """
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, *args, **kwargs):   
        
        module_config = self.module_config

        ######################
        #   Read caption data
        ######################
        self.data.caption_features = EasyDict()
        for caption_file_path in module_config.caption_features.values():
            with open(caption_file_path, "r") as f:
                caption_data = json.load(f)
                self.data.caption_features.update(caption_data)
        print('[Data Statistics] Caption features {}'.format(len(self.data.caption_features)))
        return self.data

@register_transform_functor
class LoadGoogleOCRFeatures(BaseTransform):
    def setup(self, *args, global_config=None, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, input_data, *args, **kwargs):
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

        module_config = self.module_config
        self.data.vinvl_features = input_data.vinvl_features

        ######################
        #   Read OCR data and combine with VinVL features
        ######################
        self.data.ocr_features = EasyDict()
        for data_split in ['train', 'test']:
            # Read pre-extracted features
            ocr_feature_path = module_config.ocr_features[data_split]
            logger.info(f'Reading data in: {ocr_feature_path}')
            for image_key, prediction in tqdm(self.data.vinvl_features.items()):
                ocr_feature_file = os.path.join(ocr_feature_path, '{}_ocr.json'.format(image_key))
                if os.path.exists(ocr_feature_file):
                    with open(ocr_feature_file, 'r') as json_fp:
                        ocr_data = json.load(json_fp)
                    self.data.ocr_features[image_key] = ocr_data
                else:
                    logger.debug(f'Skipping: {ocr_feature_file}')
            
        # save_cached_data(self.config, self.data.ocr_features, 'ocr_feature_preprocessed')

        count = 0
        for img_key, annotation in self.data.ocr_features.items():
            if len(annotation['filtered_text_annotations']) > 0:
                count += 1
        logger.info('[Data Statistics] OCR features {}, {} has annotations.'.format(len(self.data.ocr_features), count))
        

        if module_config.ocr_features['combine_with_vinvl']:

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
                # save_cached_data(self.global_config, self.data.vinvl_features, 'vinvl_feature_preprocessed')
            else:
                logger.info('OCR feature detected in VinVL feature dict...skipping..')
        
        return self.data


@register_transform_functor
class LoadOKVQAData(HFDatasetTransform):
    """
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        
    def _call(self, inputs, **kwargs):
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        ######################
        #   Read OK-VQA data
        ######################
        def most_frequent(List):
            return max(set(List), key = List.count)
        

        answer_candidate_list = []
        vqa_helpers = EasyDict({
            'train': VQA(module_config.vqa_data_path.annotation_files.train, 
                            module_config.vqa_data_path.question_files.train),
            'test': VQA(module_config.vqa_data_path.annotation_files.test, 
                            module_config.vqa_data_path.question_files.test),
        })
        
        self.data.okvqa_data = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
            'vqa_helpers': vqa_helpers,
        })
        self.data.images = EasyDict({})

        output_data = DatasetDict()
        
        for data_split, vqa_helper in vqa_helpers.items():
            vqa_helper.createIndex()
            vqa_helper.info()

            # For each data split, prepare dataset
            self.data.okvqa_data[data_split] = EasyDict({}) # re-initialise
            # Create list of images from helper
            img_data_path = module_config.image_data_path[data_split]
            img_list = []
            for imgId in vqa_helper.imgToQA.keys():
                dataSubType = vqa_helper.dataSubType
                imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
                img_path = os.path.join(img_data_path, imgFilename)
                img_list.append((imgId, img_path, imgFilename))
                # if self.config.data_loader.dummy_dataloader:
                # Load only a few samples for testing
                if self.use_dummy_data:
                    if len(img_list) > 20:
                        break
            
            # Create entries for each question and related answers
            self.data.okvqa_data[data_split].data_items = {}
            for imgId, img_path, imgFilename in tqdm(img_list):
                # avoid error in splitting: must remove ".." in "../path/to/file"
                # img_key = img_p.replace('..', '').split('.')[0].split('_')[-1]
                img_key = imgId
                img_key_str = str(img_key)

                if module_config.add_caption_features:
                    img_caption = self.data.caption_features.get(img_key_str, None)
                    if img_caption is not None: 
                        img_caption = img_caption[0] 
                    else: 
                        logger.debug('No caption found for {}!'.format(img_key))
                
                img_key_full = str(img_key).zfill(12)
                if module_config.add_images:
                    img = cv2.imread(img_path)
                    self.data.images[img_path] = img
                else:
                    self.data.images[img_path] = {
                        'img_path': img_path,
                    }
                
                # convert self.data.images into a dataset
                image_df = pd.DataFrame.from_dict(self.data.images, orient="index")
                image_ds = Dataset.from_pandas(image_df).rename_columns({"__index_level_0__": "id"})
                output_data['images'] = image_ds
                
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
                    entry_data.question_id = str(question_and_answer['question_id'])
                    entry_data.img_path = img_path
                    entry_data.img_key_full = img_key_full
                    entry_data.img_key = img_key
                    entry_data.img_file_name = imgFilename
                    entry_data.img = None # we don't add images to individual samples any more because this is expensive

                    # Add features in accordance with settings
                    if module_config.add_caption_features:
                        entry_data.img_caption = img_caption

                    if module_config.add_VinVL_features:
                        # Read predictions from VinVL features
                        VinVL_prediction = self.data.vinvl_features.get(entry_data.img_key_full, None)
                        if not VinVL_prediction:
                            logger.error(f"Image {entry_data.img_key_full} does not find associated VinVL features!")
                            raise KeyError(f"Image {entry_data.img_key_full} does not find associated VinVL features!")

                        objects = []
                        for obj in VinVL_prediction['objects']:
                            # obj_feature = np.frombuffer(base64.b64decode(obj['feature']), np.float32)
                            # obj_feature_ts = torch.FloatTensor(obj_feature.copy())
                            obj_class = obj['class']
                            obj_rect = obj['rect']
                            obj_attributes = obj['attributes']
                            obj_attribute_scores = obj['attr_scores']
                            obj_ocr = obj.get('ocr', [])
                            objects.append({
                                'class': obj_class,
                                # 'obj_feature': obj_feature_ts,
                                'rect': obj_rect,
                                'attributes': obj_attributes,
                                'attribute_scores': obj_attribute_scores,
                                'ocr': obj_ocr,
                            })
                        entry_data.objects = objects
                    
                    if module_config.add_OCR_features:
                        text_annotations = self.data.ocr_features[entry_data.img_key_full]['filtered_text_annotations']
                        entry_data.img_ocr = text_annotations

                    self.data.okvqa_data[data_split].data_items[entry_data.question_id] = entry_data
                    
                    # Collect answer candidates for evaluation
                    for ans in list(question_and_answer['answers'].values()):
                        if ans not in answer_candidate_list:
                            answer_candidate_list.append(ans)
                            # if data_split == 'test':
                            #     print(ans, 'is added from test set!')
            
            # # convert to Dataset
            split_df = pd.DataFrame.from_dict(self.data.okvqa_data[data_split].data_items, orient="index")
            split_ds = Dataset.from_pandas(split_df)
            output_data[data_split] = split_ds

            # for entry_data in self.data.okvqa_data[data_split].data_items:
            #     self.data.okvqa_data['lookup'][str(entry_data.question_id)] = entry_data

            # Report statistics
            logger.info('[Data statistics] split: {}  entries: {}'.format(
                data_split,
                len(self.data.okvqa_data[data_split].data_items)))

        # Save answer candidate list
        self.data.okvqa_data.answer_candidate_list = answer_candidate_list

        # self.output_data = EasyDict(
        #     okvqa_data=self.data.okvqa_data,
        #     images=self.data.images,
        #     # ocr_features=self.data.ocr_features,
        #     # vinvl_features=self.data.vinvl_features,
        # )
        # NOTE: we have merged features with items directly, 
        #       so they are not required in later stages
        
        return output_data


@register_transform_functor
class CropRegionOfInterestImages(HFDatasetTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function crops ROIs from the object detection results of each image
        These ROIs are saved to data.images so that their features are processed automatically.
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        new_image_dict = {}

        for data_split in ['train', 'test']:
            def add_ROIs(item):
                item = EasyDict(item)
                # print(item.question)
                selected_objects = []
                objects = []
                for obj in item.objects:
                    # print(obj['rect'], obj['class'])
                    xmin, ymin, xmax, ymax = obj['rect']
                    obj_area = (ymax - ymin) * (xmax - xmin)
                    objects.append((obj_area, obj))
                    if obj['class'].lower().strip() in item.question.lower():
                        selected_objects.append(obj)
                
                objects = sorted(objects, key=lambda x: x[0], reverse=True)
                
                for obj_area, obj in objects:
                    xmin, ymin, xmax, ymax = obj['rect']
                    if len(selected_objects) >= self.module_config.max_objects:
                        break
                    else:
                        valid = True
                        # Remove duplications
                        for existing_obj in selected_objects:
                            if existing_obj['class'] == obj['class']:
                                e_xmin, e_ymin, e_xmax, e_ymax = existing_obj['rect']
                                if xmin >= e_xmin and ymin >= e_ymin and xmax <= e_xmax and ymax <= e_ymax:
                                    # this object is contained in an existing object with the same class name
                                    valid = False
                        if valid:
                            selected_objects.append(obj)

                img_path = item.img_path

                ROIs = []
                for obj in selected_objects:
                    xmin, ymin, xmax, ymax = obj['rect']
                    xmin, ymin, xmax, ymax = round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)
                    new_id = f"{img_path}|||{obj['class']}_{xmin}_{ymin}_{xmax}_{ymax}"
                    new_img_dict = {
                        'id': new_id,
                        'img_path': img_path,
                        'obj': obj,
                        'crop': [xmin, ymin, xmax, ymax],
                    }
                    new_image_dict[new_id] = new_img_dict
                    ROIs.append(new_id)
                
                item.ROIs = ROIs
                return item
        

            split_df = self.data[data_split]
            split_df = split_df.map(add_ROIs)
            self.data[data_split] = split_df
        
        # convert new_image_dict into dataset and concatenate it with existing images
        new_image_df = pd.DataFrame.from_dict(new_image_dict, orient="index").set_index("id")
        print(new_image_df)
        old_image_df = self.data['images'].to_pandas().set_index("id")
        print(old_image_df)
        new_image_df = pd.concat([old_image_df, new_image_df])
        print(new_image_df)
        
        new_image_ds = Dataset.from_pandas(new_image_df)
        output_data = DatasetDict(self.data)
        
        output_data['images'] = new_image_ds
        
        print(output_data)
        return output_data


@register_transform_functor
class CropRandomRegionOfInterestImages(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function crops ROIs from the object detection results of each image
        These ROIs are saved to data.images so that their features are processed automatically.
        """
        for input_data in inputs:
            self.data.update(input_data)
        

        for data_split in ['train', 'test']:
            for item in tqdm(self.data.okvqa_data[data_split].data_items, desc="generate random image crops"):
                # print(item.question)
                selected_objects = []
                
                img_path = item.img_path
                img_size = PIL.Image.open(img_path).size
                w, h = img_size[0], img_size[1]

                if self.module_config.even_split:
                    size = math.sqrt(self.module_config.max_objects)
                    assert size.is_integer() == True
                    size = int(size)
                    for i in range(size):
                        for j in range(size):
                            xmin = w / size * i
                            xmax = w / size * (i+1)
                            ymin = h / size * j
                            ymax = h / size * (j+1)
                            xmax = min(w, xmax)
                            ymax = min(h, ymax)
                            selected_objects.append({
                                'class': 'even_split',
                                'rect': [xmin, ymin, xmax, ymax],
                            })

                else:
                    min_size = 10

                    for _ in range(self.module_config.max_objects):
                        xmin = min_size + np.random.rand()* (w - 2*min_size)
                        xmax = xmin + min_size + (w - xmin - min_size) * np.random.rand()
                        xmax = min(w, xmax)
                        ymin = min_size + np.random.rand()* (h - 2*min_size)
                        ymax = ymin + min_size + (h - ymin - min_size) * np.random.rand()
                        ymax = min(h, ymax)
                        selected_objects.append({
                            'class': 'random',
                            'rect': [xmin, ymin, xmax, ymax],
                        })

                ROIs = []
                for obj in selected_objects:
                    xmin, ymin, xmax, ymax = obj['rect']
                    xmin, ymin, xmax, ymax = round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)
                    new_id = f"{img_path}|||{obj['class']}_{xmin}_{ymin}_{xmax}_{ymax}"
                    new_img_dict = {
                        'img_path': img_path,
                        'obj': obj,
                        'crop': [xmin, ymin, xmax, ymax],
                    }
                    self.data.images[new_id] = new_img_dict
                    ROIs.append(new_id)
                item.ROIs = ROIs
        
        return self.data


@register_transform_functor
class LoadGoogleSearchPassageData(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, *args, **kwargs):
        """
        This function loads data and features from Google Search Passage Corpus
        This corpus contains a train split and a full split
        use_full_split: if True, lodas the full split, otherwise use only the train split
                        all the following training/testing will be done on the train split
                        However, please note that if using static outputs from a pre-trained DPR model,
                        it is possible that loaded ids are out of range!
                        Warning: this may impact the options of the DPR training:
                            When "self.config.data_loader.additional.full_corpus_in_training" is set to true,
                            the negative samples can only be taken from the train split.
        {
            "passage_data_path": {
                "train": "..",
                "full": "..",
            },
            "use_full_split": True,
        }
        """
        
        module_config = self.module_config
        self.data.passages = {
            'id2doc': {}, # full corpus
            'id2doc_train': {}, # only train corpus
            'annotations': {},
        }

        ######################
        # Read knowledge passage data
        ######################
        passage_file = module_config.passage_data_path.full
        with open(passage_file, 'r') as csv_file:
            count_total = sum(1 for row in csv_file)
            csv_file.seek(0)
            read_tsv = csv.reader(csv_file, delimiter=",")
            END_OF_TRAIN = False
            for row in tqdm(read_tsv, total=count_total):
                id, content = row
                if id == 'kid':
                    continue

                if id == '112724' or id == 112724:
                    # Mark n_train since the corpus has all documents
                    self.data.passages.n_train = len(self.data.passages.id2doc_train)
                    END_OF_TRAIN = True
                    if not module_config.use_full_split:
                        # if we don't use full split, we stop here
                        break
                id = f"GS_{id}"
                if not END_OF_TRAIN:
                    self.data.passages.id2doc_train[id] = content
                self.data.passages.id2doc[id] = content
    
        return self.data



@register_transform_functor
class PrepareGoogleSearchPassages(HFDatasetTransform):
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
        
        train_df = pd.DataFrame.from_dict(self.data.passages.id2doc_train, orient="index")
        train_ds = Dataset.from_pandas(train_df)
        train_ds = train_ds.rename_columns({
            '__index_level_0__': 'passage_id',
            '0': 'passage_content',
        })
        
        test_df = pd.DataFrame.from_dict(self.data.passages.id2doc, orient="index")
        test_ds = Dataset.from_pandas(test_df)
        test_ds = test_ds.rename_columns({
            '__index_level_0__': 'passage_id',
            '0': 'passage_content',
        })
        
        print(train_ds[:3])
        output_data = DatasetDict(
            train_passages=train_ds,
            valid_passages=test_ds,
            test_passages=test_ds,
        )
        print(output_data)
        
        return output_data


@register_transform_functor
class LoadGoogleSearchAnnotations(HFDatasetTransform):
    def setup(self, use_all_samples=False, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.use_all_samples = use_all_samples

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

        ######################
        #  Get weak supervision annotations
        ######################
        self.data.okvqa_data_with_dpr_output = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
        })
        self.data.passages.annotations = EasyDict({})
        output_data = DatasetDict()
        for data_split in ['train', 'test']:
            
            annotation_json = module_config.annotations_path[data_split]
            with open(annotation_json, 'r') as f:
                annotations = json.load(f)
            

            # Read annotations and (optionally) captions
            for annotation in tqdm(annotations):
                question_id = annotation['question_id']
                caption = annotation['caption']
                passages = annotation['ctxs']
                knowledge_collection = []
                for passage in passages:
                    passage_id = passage['id']
                    passage_id = f"GS_{passage_id}"
                    has_answer = passage['has_answer']
                    # print(passage)
                    # print(self.data.passages.id2doc[str(passage_id)])
                    knowledge_collection.append(passage_id)
                
                if self.data.passages.annotations.get(str(question_id), None) is None:
                    self.data.passages.annotations[str(question_id)] = {
                        'passages': knowledge_collection,
                    }
                else:
                    logger.error(f'Duplicated entries in annotation files! question id: {str(question_id)} collision!')
                
        for data_split in ['train', 'test']:
            self.data.okvqa_data_with_dpr_output[data_split] = EasyDict({})
            self.data.okvqa_data_with_dpr_output[data_split].data_items = []
            
            missing_entries = []
            missing_data = []

            for item in tqdm(self.data[data_split], desc="Adding knowledge annotation to samples"):
                question_id = str(item['question_id'])
                annotation = self.data.passages.annotations.get(str(question_id), None)
                
                if annotation is None:
                    missing_entries.append(str(question_id))
                    # logger.warning("question {} (split {}) not found in knowledge.".format(str(question_id), data_split))
                    if self.use_all_samples:
                        # in testing mode, all samples must be used
                        related_knowledge = ['GS_0']
                else: 
                    related_knowledge = annotation['passages']
                    if len(related_knowledge) == 0:
                        missing_data.append(str(question_id))
                        # logger.warning("question {} (split {}) has no related knowledge in annotations.".format(str(question_id), data_split))
                        if self.use_all_samples:
                            # in testing mode, all samples must be used
                            related_knowledge = ['GS_0']
                        else:
                            continue
                
                knowledge_item = EasyDict(dict(item))
                knowledge_item['question_id'] = question_id
                knowledge_item['pos_item_ids'] = related_knowledge
                knowledge_item['pos_item_contents'] = [
                    self.data.passages.id2doc[str(passage_id)] for passage_id in related_knowledge
                ]
                self.data.okvqa_data_with_dpr_output[data_split].data_items.append(knowledge_item)
                

            if len(missing_entries) > 0:
                logger.warning(f"{len(missing_entries)} questions (split {data_split}) not found in knowledge. \n {missing_entries}")
            if len(missing_data) > 0:
                logger.warning(f"{len(missing_data)} questions (split {data_split}) has no annotations. \n {missing_data}")
            
            df = pd.DataFrame.from_dict(self.data.okvqa_data_with_dpr_output[data_split].data_items)
            print(df)
            ds = Dataset.from_pandas(df)
            # print(ds[0])
            # Load item data into lookup with question_id as index
            # logger.info('Indexing data items...')
            
            # for item in tqdm(self.data.okvqa_data_with_dpr_output[data_split].data_items):
            #     question_id = item['question_id']
            #     self.data.okvqa_data_with_dpr_output.lookup[str(question_id)] = item
            output_data[data_split] = ds
            
            # Report statistics
            logger.info('[Data statistics] loaded with knowledge data split: {}  entries: {}'.format(
                data_split,
                len(ds)))

        output_data['valid'] = output_data['test']
        print(output_data)
        return output_data



@register_transform_functor
class ExtractImageFeaturesWithViT(BaseTransform):
    def setup(self, image_processor_config, vit_model_config, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.image_processor_config = image_processor_config
        self.vit_model_config = vit_model_config

        def _get_image_processor(image_processor_config):
            if 'ImageProcessorClass' in image_processor_config:
                ImageProcessorClass = globals()[image_processor_config.ImageProcessorClass]
                return ImageProcessorClass.from_pretrained(image_processor_config.ImageProcessorModelVersion)
            else:
                return None
        
        self.image_processors = {
            name: _get_image_processor(cfg) 
            for name, cfg in self.image_processor_config.items()
        }

        self.image_processor = self.image_processors['vit_image_processor']

        VisionModelConfigClass = globals()[vit_model_config.VisionModelConfigClass]
        VisionModelClass = globals()[vit_model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(vit_model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(vit_model_config.VisionModelVersion, config=vision_model_config)
        self.vision_model.eval()
        if torch.cuda.is_available():
            logger.info("using GPU for encoding...")
            self.vision_model = self.vision_model.cuda() 

    def _call(self, inputs, *args, **kwargs):
        """
        This function extract image features using the input ViT model. The last hidden states are extracted.
        {
            input_column: "images",
            image_processor_config: image_processor_config,
            vit_model_config: {
                "VisionModelConfigClass": "CLIPVisionConfig",
                "VisionModelClass": "CLIPVisionModel",
                "VisionModelVersion": "openai/clip-vit-base-patch32",
            },
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        input_column = self.module_config.input_column
        images = self.data[input_column]

        # for image_path, image_dict in tqdm(self.data[input_column].items()):
        #     images[image_dict['image_path']] = image_dict

        if isinstance(images, Dataset):
            image_dataset = images
        else:
            df = pd.DataFrame.from_dict(images, orient="index")
            image_dataset = Dataset.from_pandas(df)

        def open_images(batch, image_processor=None):
            try:
                batch_images = [
                    PIL.Image.open(img_path).convert("RGB") for img_path in batch['img_path']
                ]
                batch_crops = batch.get('crop', [None]*len(batch_images))

                cropped_images = []
                for img, crop in zip(batch_images, batch_crops):
                    # print(img.size, crop)
                    if crop is not None:
                        xmin, ymin, xmax, ymax = crop
                        # if the size of the crop is too small, enlarge the crop to at least 5 pixels in size
                        if xmax - xmin < 5 and ymax - ymin < 5:
                            if xmax - xmin < 5:
                                xmin = max(0, xmin - 2.5)
                                xmax = min(img.size[0], xmax + 2.5)
                            if ymax - ymin < 5:
                                ymin = max(0, ymin - 2.5)
                                ymax = min(img.size[1], ymax + 2.5)
                            print("enlarged: ", crop, (xmin, ymin, xmax, ymax))
                            crop = (xmin, ymin, xmax, ymax)
                        cropped_images.append(img.crop(crop))
                        # print("cropped: ", crop)
                    else:
                        cropped_images.append(img)
            except Exception as e:
                print(cropped_images)
                print(e)

            print(f"[Preparing Encoding] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
            encoded = image_processor(cropped_images, return_tensors='pt')
            pixel_values = encoded.pixel_values
            batch['pixel_values'] = pixel_values

            return batch


        def encode_images(batch, vision_model):
            # batch_images = [
            #     PIL.Image.open(img_path) for img_path in batch['img_path']
            # ]
            # batch_crops = batch.get('crop', [None]*len(batch_images))

            # cropped_images = []
            # for img, crop in zip(batch_images, batch_crops):
            #     # print(img.size, crop)
            #     if crop is not None:
            #         cropped_images.append(img.crop(crop))
            #         # print("cropped: ", crop)
            #     else:
            #         cropped_images.append(img)
            
            # encoded = self.image_processor(cropped_images, return_tensors='pt')
            # pixel_values = encoded.pixel_values.to(self.vision_model.device)
            pixel_values = batch['pixel_values']
            pixel_values = torch.FloatTensor(pixel_values).to(vision_model.device)
            outputs = vision_model(pixel_values)
            image_features = outputs.last_hidden_state[:, 0]
            batch['image_features'] = image_features
            return batch

        logger.info("start generating image embeddings...")
        image_dataset = image_dataset.map(
            open_images,
            batched=True,
            batch_size=self.module_config.batch_size,
            fn_kwargs={
                "image_processor": self.image_processor,
            },
            num_proc=16,
            writer_batch_size=1024,
        )
        image_dataset_with_embeddings = image_dataset.map(
            encode_images,
            batched=True,
            batch_size=self.module_config.batch_size,
            remove_columns=["pixel_values"],
            fn_kwargs={
                "vision_model": self.vision_model,
            },
        )
        # image_dataset_with_embeddings = image_dataset_with_embeddings.rename_columns({'__index_level_0__': 'index'})
        self.vision_model = self.vision_model.cpu()
        del self.vision_model

        output_data = DatasetDict(
            image_dataset_with_embeddings=image_dataset_with_embeddings,
        )
        return output_data
            
from safetensors import safe_open
from safetensors.torch import save_file

@register_transform_functor
class ExtractImageFeaturesWithViTv2(BaseTransform):
    def setup(self, image_processor_config, vit_model_config, cache_folder="./embedding_cache", *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.image_processor_config = image_processor_config
        self.vit_model_config = vit_model_config

        def _get_image_processor(image_processor_config):
            if 'ImageProcessorClass' in image_processor_config:
                ImageProcessorClass = globals()[image_processor_config.ImageProcessorClass]
                return ImageProcessorClass.from_pretrained(image_processor_config.ImageProcessorModelVersion)
            else:
                return None
        
        self.image_processors = {
            name: _get_image_processor(cfg) 
            for name, cfg in self.image_processor_config.items()
        }

        self.image_processor = self.image_processors['vit_image_processor']

        VisionModelConfigClass = globals()[vit_model_config.VisionModelConfigClass]
        VisionModelClass = globals()[vit_model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(vit_model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(vit_model_config.VisionModelVersion, config=vision_model_config)
        self.vision_model.eval()
        if torch.cuda.is_available():
            logger.info("using GPU for encoding...")
            self.vision_model = self.vision_model.cuda() 

        self.cache_folder = cache_folder
        # make dir
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        


    def _call(self, inputs, *args, **kwargs):
        """
        This function extract image features using the input ViT model. The last hidden states are extracted.
        {
            input_column: "images",
            image_processor_config: image_processor_config,
            vit_model_config: {
                "VisionModelConfigClass": "CLIPVisionConfig",
                "VisionModelClass": "CLIPVisionModel",
                "VisionModelVersion": "openai/clip-vit-base-patch32",
            },
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        input_column = self.module_config.input_column
        images = self.data[input_column]

        # for image_path, image_dict in tqdm(self.data[input_column].items()):
        #     images[image_dict['image_path']] = image_dict

        df = pd.DataFrame.from_dict(images, orient="index")
        image_dataset = Dataset.from_pandas(df)

        def load_cached_embeddings(batch):
            batch_indices = batch['__index_level_0__']
            batch_cache_filenames = [
                os.path.join(self.cache_folder, f"{idx.split('/')[-1]}.safetensors") for idx in batch_indices
            ]
            image_features = []
            for cache_filename in batch_cache_filenames:
                if not os.path.exists(cache_filename):
                    image_features.append(torch.FloatTensor([0]*768))
                else:
                    # logger.info(f"loading image features from {cache_filename}")
                    # with open(cache_filename, 'rb') as f:
                    #     image_features.append(pickle.load(f))
                    with safe_open(cache_filename, framework="pt", device="cpu") as f:
                        image_features.append(torch.FloatTensor(f.get_tensor("data")))
            batch['image_features'] = image_features
            print(f"[Reading] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
            
            return batch

        def encode_images(batch, vision_model, image_processor):
            batch_indices = batch['__index_level_0__']
            batch_cache_filenames = [
                os.path.join(self.cache_folder, f"{idx.split('/')[-1]}.safetensors") for idx in batch_indices
            ]

            all_prepared = True
            for cache_filename in batch_cache_filenames:
                if not os.path.exists(cache_filename):
                    all_prepared = False
                    break
            
            if all_prepared:
                # image_features = []
                # for cache_filename in batch_cache_filenames:
                #     logger.info(f"loading image features from {cache_filename}")
                #     # with open(cache_filename, 'rb') as f:
                #     #     image_features.append(pickle.load(f))
                #     with safe_open(cache_filename, framework="pt", device="cpu") as f:
                #         image_features.append(f.get_tensor("data"))

                # batch['image_features'] = image_features
                print("skipped encoding!")
                return batch
            
            else:
                batch_images = [
                    PIL.Image.open(img_path).convert("RGB") for img_path in batch['img_path']
                ]
                batch_crops = batch.get('crop', [None]*len(batch_images))

                cropped_images = []
                for img, crop in zip(batch_images, batch_crops):
                    # print(img.size, crop)
                    if crop is not None:
                        xmin, ymin, xmax, ymax = crop
                        # if the size of the crop is too small, enlarge the crop to at least 5 pixels in size
                        if xmax - xmin < 5 and ymax - ymin < 5:
                            if xmax - xmin < 5:
                                xmin = max(0, xmin - 2.5)
                                xmax = min(img.size[0], xmax + 2.5)
                            if ymax - ymin < 5:
                                ymin = max(0, ymin - 2.5)
                                ymax = min(img.size[1], ymax + 2.5)
                            print("enlarged: ", crop, (xmin, ymin, xmax, ymax))
                            crop = (xmin, ymin, xmax, ymax)
                        cropped_images.append(img.crop(crop))
                        # print("cropped: ", crop)
                    else:
                        cropped_images.append(img)

                
                encoded = image_processor(cropped_images, return_tensors='pt')
                pixel_values = encoded.pixel_values
                pixel_values = torch.FloatTensor(pixel_values).to(vision_model.device)
                outputs = vision_model(pixel_values)
                image_features = outputs.last_hidden_state[:, 0]
                batch['image_features'] = image_features

                print(f"[Encoding] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
                for cache_filename, image_feature in zip(batch_cache_filenames, image_features):
                    logger.info(f"save image feature to {cache_filename}")
                    # with open(cache_filename, 'wb') as f:
                    #     pickle.dump(image_feature, f)
                    save_file({'data': image_feature}, cache_filename)

                return batch

        logger.info("start generating image embeddings...")
        image_dataset = image_dataset.map(
            load_cached_embeddings,
            batched=True,
            batch_size=32,
            # num_proc=32,
            # writer_batch_size=1024,
        )
        image_dataset_with_embeddings = image_dataset.map(
            encode_images,
            batched=True,
            batch_size=self.module_config.batch_size,
            # remove_columns=["pixel_values"],
            fn_kwargs={
                "vision_model": self.vision_model,
                "image_processor": self.image_processor,
            },
        )
        # image_dataset_with_embeddings = image_dataset_with_embeddings.rename_columns({'__index_level_0__': 'index'})
        self.vision_model = self.vision_model.cpu()
        del self.vision_model

        output_data = EasyDict(
            image_dataset_with_embeddings=image_dataset_with_embeddings,
        )
        return output_data

@register_transform_functor
class ExtractImageFeaturesWithViTv3(HFDatasetTransform):
    """
    This v3 version saves tensors to ES
    """
    def setup(self, image_processor_config, vit_model_config, index_name="encoded_image_features", _num_proc=None, _num_gpu_proc=None, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.image_processor_config = image_processor_config
        self.vit_model_config = vit_model_config

        def _get_image_processor(image_processor_config):
            if 'ImageProcessorClass' in image_processor_config:
                ImageProcessorClass = globals()[image_processor_config.ImageProcessorClass]
                return ImageProcessorClass.from_pretrained(image_processor_config.ImageProcessorModelVersion)
            else:
                return None
        
        self.image_processors = {
            name: _get_image_processor(cfg) 
            for name, cfg in self.image_processor_config.items()
        }

        self.image_processor = self.image_processors['vit_image_processor']

        VisionModelConfigClass = globals()[vit_model_config.VisionModelConfigClass]
        VisionModelClass = globals()[vit_model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(vit_model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(vit_model_config.VisionModelVersion, config=vision_model_config)
        self.vision_model.eval()
        if torch.cuda.is_available():
            logger.info("using GPU for encoding...")
            self.vision_model = self.vision_model.cuda() 

        self.index_name = index_name
        self.num_proc = _num_proc
        self.num_gpu_proc = _num_gpu_proc
        


    def _call(self, inputs, *args, **kwargs):
        """
        This function extract image features using the input ViT model. The last hidden states are extracted.
        {
            input_column: "images",
            image_processor_config: image_processor_config,
            vit_model_config: {
                "VisionModelConfigClass": "CLIPVisionConfig",
                "VisionModelClass": "CLIPVisionModel",
                "VisionModelVersion": "openai/clip-vit-base-patch32",
            },
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        input_column = self.module_config.input_column
        image_dataset = self.data[input_column]

        # for image_path, image_dict in tqdm(self.data[input_column].items()):
        #     images[image_dict['image_path']] = image_dict

        
        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]

        es = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.environ["ELASTIC_CA_CERTS"],
            basic_auth=("elastic", ELASTIC_PASSWORD),
            timeout=60,
        )

        index_name = self.index_name

        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name)
        
        def load_cached_embeddings(batch):
            es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
                timeout=60,
            )
            batch_indices = batch.get('__index_level_0__', False) or batch['id']
            batch_indices = [f"{idx.split('/')[-1]}" for idx in batch_indices]

            found = []
            image_features = []

            queries = batch_indices
            docs = [
                {
                    '_index': index_name,
                    '_id': q,
                } for q in queries
            ]
            resp = es.mget(index=index_name, docs=docs)
            for doc in resp['docs']:
                if doc['found']:
                    image_features.append(torch.FloatTensor(doc['_source']['feature']))
                    found.append(True)
                else:
                    image_features.append(torch.FloatTensor([0]*768))
                    found.append(False)

            # report number of found
            # print(f"found {sum(found)} out of {len(found)}")

            batch['image_features'] = image_features
            batch['found'] = found
            # print(f"[Reading] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
            
            return batch

        def encode_images(batch, rank, vision_model, image_processor):
            es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
                timeout=60,
            )
            batch_indices = batch.get('__index_level_0__', False) or batch['id']
            batch_indices = [f"{idx.split('/')[-1]}" for idx in batch_indices]
            found = batch['found']
            all_prepared = (sum(found) == len(found))
            
            if all_prepared:
                print("skipped encoding!")
                return batch
            
            else:
                if rank is not None:
                    vision_model = vision_model.to(torch.device(f"cuda:{rank}"))
                
                batch_images = [
                    PIL.Image.open(img_path).convert("RGB") for img_path in batch['img_path']
                ]
                

                batch_crops = batch.get('crop', [None]*len(batch_images))

                cropped_images = []
                for img, crop in zip(batch_images, batch_crops):
                    # print(img.size, crop)
                    if crop is not None:
                        xmin, ymin, xmax, ymax = crop
                        # if the size of the crop is too small, enlarge the crop to at least 5 pixels in size
                        if xmax - xmin < 5 and ymax - ymin < 5:
                            if xmax - xmin < 5:
                                xmin = max(0, xmin - 2.5)
                                xmax = min(img.size[0], xmax + 2.5)
                            if ymax - ymin < 5:
                                ymin = max(0, ymin - 2.5)
                                ymax = min(img.size[1], ymax + 2.5)
                            print("enlarged: ", crop, (xmin, ymin, xmax, ymax))
                            crop = (xmin, ymin, xmax, ymax)
                        cropped_images.append(img.crop(crop))
                        # print("cropped: ", crop)
                    else:
                        cropped_images.append(img)

                
                encoded = image_processor(cropped_images, return_tensors='pt')
                pixel_values = encoded.pixel_values
                pixel_values = torch.FloatTensor(pixel_values).to(vision_model.device)
                outputs = vision_model(pixel_values)
                image_features = outputs.last_hidden_state[:, 0]
                batch['image_features'] = image_features

                print(f"[Encoding] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
                
                actions = []

                # read pickle files and save as safe tensors
                for file_id, is_found, feature in zip(batch_indices, found, image_features):
                    feature = feature.tolist()
                    action = {
                        '_op_type': "index",
                        '_index': index_name,
                        '_id': file_id,
                        '_source': {
                            'feature': feature,
                        }
                    }
                    if not is_found:
                        actions.append(action)
                
                if len(actions) > 0:
                    res = helpers.bulk(es, actions, request_timeout=120)
                    # print(f"number of success {res[0]}")
                    if res[0] != len(actions):
                        print("errors", res[1])
                    print(f"[Rank {rank}] Successfully indexed {len(actions)} items into ES.")
                else:
                    print("No new items to index.")

                # vision_model = vision_model.cpu()

                return batch

        logger.info("start generating image embeddings...")
        image_dataset = image_dataset.map(
            load_cached_embeddings,
            batched=True,
            batch_size=1000,
            num_proc=self.num_proc,
            load_from_cache_file=False,
            # writer_batch_size=1024,
        )
        # sort by found == True
        print("Total number of images: ", len(image_dataset))
        remaining_image_dataset = image_dataset.filter(lambda x: x['found'], num_proc=16, load_from_cache_file=False)
        print("Already encoded number of images: ", len(remaining_image_dataset))
        image_dataset = image_dataset.filter(lambda x: not x['found'], num_proc=16, load_from_cache_file=False)
        print("Number of images to encode: ", len(image_dataset))
        # count number of found == True
        # founded_ds = image_dataset.filter(lambda x: x['found'], num_proc=16)
        # logger.info(f"found {len(founded_ds)} out of {len(image_dataset)}")
        # founded_ds.cleanup_cache_files()
        # del founded_ds

        if len(image_dataset) != 0:

            if self.num_gpu_proc is not None:
                from multiprocess import set_start_method
                try:
                    set_start_method('spawn', force=True)
                    print("spawned")
                except RuntimeError:
                    pass
                
                image_dataset_with_embeddings = image_dataset.map(
                    encode_images,
                    batched=True,
                    batch_size=self.module_config.batch_size,
                    with_rank=True,
                    num_proc=self.num_gpu_proc,
                    # remove_columns=["pixel_values"],
                    fn_kwargs={
                        "vision_model": self.vision_model,
                        "image_processor": self.image_processor,
                    },
                )
            else:
                image_dataset_with_embeddings = image_dataset.map(
                    encode_images,
                    batched=True,
                    batch_size=self.module_config.batch_size,
                    num_proc=None,
                    # remove_columns=["pixel_values"],
                    fn_kwargs={
                        "rank": None,
                        "vision_model": self.vision_model,
                        "image_processor": self.image_processor,
                    },
                )

            # Concatenate with remaining dataset
            from datasets import concatenate_datasets
            image_dataset_with_embeddings = concatenate_datasets([image_dataset_with_embeddings, remaining_image_dataset])
        else:
            image_dataset_with_embeddings = remaining_image_dataset
            
        # image_dataset_with_embeddings = image_dataset_with_embeddings.rename_columns({'__index_level_0__': 'index'})
        self.vision_model = self.vision_model.cpu()
        del self.vision_model

        output_data = DatasetDict(
            image_dataset_with_embeddings=image_dataset_with_embeddings,
        )
        print(output_data)
        return output_data
