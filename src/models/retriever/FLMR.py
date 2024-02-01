"""
This file defines FLMR models which inherits from ColBERT
Author: Weizhe Lin
Date: 23/01/2024
"""

from colbert.modeling.colbert import ColBERT

from transformers import ViTModel, ViTConfig
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers import ViTMAEConfig, ViTMAEModel
from transformers.modeling_utils import ModuleUtilsMixin
import torch
import torch.nn as nn


import logging
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class FLMR(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        image_features = image_features.to(self.device)
        
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        
        # image_features: batch_size (x num_ROIs) x ViT hidden_size
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz (x num_ROIs) x 32*128
        
        # last_hidden_states = last_hidden_states.view(
        #     -1, self.mapping_network_prefix_length, self.lm_embedding_size
        # )
        
        last_hidden_states = last_hidden_states.reshape(
            batch_size, -1, self.lm_embedding_size
        )

        Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states

        return torch.nn.functional.normalize(Q, p=2, dim=2)




class FLMRForPretraining(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        image_features = image_features.to(self.device)

        last_hidden_states = image_features
        last_hidden_states = self.vision_projection(last_hidden_states) # bz x 32*128
        
        last_hidden_states = last_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        
        Q = last_hidden_states
        
        return torch.nn.functional.normalize(Q, p=2, dim=2)



class FLMRForPretrainingWithVisionModel(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        outputs = self.vision_model(pixel_values)
        last_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz x 32*128
        
        last_hidden_states = last_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        Q = last_hidden_states

        return torch.nn.functional.normalize(Q, p=2, dim=2)
    

class FLMRWithVisionModel(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        outputs = self.vision_model(pixel_values)
        last_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz x 32*128
        
        last_hidden_states = last_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        
        Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states

        return torch.nn.functional.normalize(Q, p=2, dim=2)

