import sys
import time
import json
import copy
import numpy as np
import json
import torch

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict

import logging
logger = logging.getLogger(__name__)

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from transformers import ViTFeatureExtractor
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer


class DataLoaderWrapper():
    '''
    Data loader wrapper, general class definitions
    '''

    def __init__(self, config):
        self.config = config

        # Prepare for tokenizers
        TokenizerClass = globals()[self.config.model_config.TokenizerClass]
        self.tokenizer = TokenizerClass.from_pretrained(self.config.model_config.TokenizerModelVersion)
        self.SPECIAL_TOKENS = self.config.model_config.SPECIAL_TOKENS
        self.SPECIAL_TOKENS['additional_special_tokens'] = self.tokenizer.additional_special_tokens + self.SPECIAL_TOKENS['additional_special_tokens']
        
        self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        
        # Load second tokenizer if specified
        if self.config.model_config.get('DecoderTokenizerClass', None) is not None:
            DecoderTokenizerClass = globals()[self.config.model_config.DecoderTokenizerClass]
            self.decoder_tokenizer = DecoderTokenizerClass.from_pretrained(self.config.model_config.DecoderTokenizerModelVersion)
            self.DECODER_SPECIAL_TOKENS = self.config.model_config.DECODER_SPECIAL_TOKENS
            self.DECODER_SPECIAL_TOKENS['additional_special_tokens'] = self.decoder_tokenizer.additional_special_tokens + self.DECODER_SPECIAL_TOKENS['additional_special_tokens']
            self.decoder_tokenizer.add_special_tokens(self.DECODER_SPECIAL_TOKENS)
            
        else:
            self.decoder_tokenizer = self.tokenizer

        # Load feature extractor if specified
        if self.config.model_config.get('FeatureExtractorClass', None) is not None:
            # Load image feature extractor
            FeatureExtractorClass = globals()[self.config.model_config.FeatureExtractorClass]
            self.feature_extractor = FeatureExtractorClass.from_pretrained(self.config.model_config.FeatureExtractorModelVersion)
        else:
            self.feature_extractor = None

        # Prepare data for training
        # self.build_dataset()

    def build_dataset(self):
        """
        This function loads data and features required for building the dataset
        """

        self.data = EasyDict()

        dataset_modules = self.config.data_loader.dataset_modules.module_list
        for dataset_module in dataset_modules:
            module_config = self.config.data_loader.dataset_modules.module_dict[dataset_module]
            logger.info('Loading dataset module: {}'.format(module_config))
            loading_func = getattr(self, dataset_module)
            loading_func(module_config)
            print('data columns: {}'.format(self.data.keys()))
            
