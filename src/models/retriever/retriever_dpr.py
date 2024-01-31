import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from transformers import T5EncoderModel, T5Config
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers import CLIPTextModel, CLIPTextConfig
from easydict import EasyDict

def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


class RetrieverDPR(pl.LightningModule):
    """
    Class of retriever model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]

        QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
        query_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
        # if query_model_config.model_type == 'clip_text_model':
        #     query_model_config.max_position_embeddings = 512
        self.query_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion, config=query_model_config, ignore_mismatched_sizes=True)
        
        self.SEP_ENCODER = True if 'separate_query_and_item_encoders' in self.config.model_config.modules else None
        
        if self.SEP_ENCODER:
            ItemEncoderModelClass = globals()[self.config.model_config.ItemEncoderModelClass]

            ItemEncoderConfigClass = globals()[self.config.model_config.ItemEncoderConfigClass]
            item_model_config = ItemEncoderConfigClass.from_pretrained(self.config.model_config.ItemEncoderModelVersion)
            # if item_model_config.model_type == 'clip_text_model':
            #     item_model_config.max_position_embeddings = 512
            self.item_encoder = ItemEncoderModelClass.from_pretrained(self.config.model_config.ItemEncoderModelVersion, config=item_model_config, ignore_mismatched_sizes=True)
        else:
            # Use the same model for query and item encoders
            item_model_config = query_model_config
            self.item_encoder = self.query_encoder
        

        self.query_pooler = None
        self.item_pooler = None
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        

    
    def resize_token_embeddings(self, dim, decoder_dim=None):
        self.query_encoder.resize_token_embeddings(dim)
        if 'separate_query_and_item_encoders' in self.config.model_config.modules:
            self.item_encoder.resize_token_embeddings(decoder_dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        labels=None,
        span_labels=None,
        **kwargs
    ):
        # query encoder
        query_outputs = self.query_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        query_embeddings = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_embeddings = self.query_pooler(query_last_hidden_states)
        # query_embeddings = query_last_hidden_states
        # print('query_embeddings', query_embeddings.shape)

        # item encoder
        item_outputs = self.item_encoder(input_ids=item_input_ids,
                                            attention_mask=item_attention_mask)
        item_embeddings = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_embeddings = self.item_pooler(item_last_hidden_states)
        # item_embeddings = item_last_hidden_states
        # print('item_embeddings', item_embeddings.shape)
        
        query_embeddings = query_embeddings.contiguous()
        item_embeddings = item_embeddings.contiguous()
        
        ################## in-batch negative sampling ###############
        if 'negative_samples_across_gpus' in self.config.model_config.modules:
            # print("get rank", get_rank())
            # print("get world size", get_world_size())
            # Gather embeddings from other GPUs
            n_nodes = get_world_size()
            
            # Create placeholder to hold embeddings passed from other ranks
            global_query_embeddings_placeholder = [torch.zeros(*query_embeddings.shape).to(query_embeddings.device) for _ in range(n_nodes)]
            global_item_embeddings_placeholder = [torch.zeros(*item_embeddings.shape).to(item_embeddings.device) for _ in range(n_nodes)]
            dist.all_gather(global_query_embeddings_placeholder, query_embeddings.detach())
            dist.all_gather(global_item_embeddings_placeholder, item_embeddings.detach())

            global_query_embeddings = []
            global_item_embeddings = []
            # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
            # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
            # input()
            current_rank = get_rank()
            for rank_index, remote_q_embeddings in enumerate(global_query_embeddings_placeholder):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_query_embeddings.append(remote_q_embeddings)
                else:
                    global_query_embeddings.append(query_embeddings)

            for rank_index, remote_item_embeddings in enumerate(global_item_embeddings_placeholder):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_item_embeddings.append(remote_item_embeddings)
                else:
                    global_item_embeddings.append(item_embeddings)

            # Replace the previous variables with gathered tensors
            query_embeddings = torch.cat(global_query_embeddings)
            item_embeddings = torch.cat(global_item_embeddings)
            

        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos
        
        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size  
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(labels.device)
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step*i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)

        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return EasyDict({
            'loss': loss,
        })

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        # query encoder
        query_outputs = self.query_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        query_last_hidden_states = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states
        return query_embeddings

    def generate_item_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        # item encoder
        item_outputs = self.item_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        item_last_hidden_states = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_last_hidden_states = self.item_pooler(item_last_hidden_states)
        item_embeddings = item_last_hidden_states
        return item_embeddings
    
    
    def create_bpr_loss(self, query, pos_items, neg_items):
        """[summary]
        Args:
            query ([type]): batch_size x hidden_size
            pos_items ([type]): batch_size x hidden_size
            neg_items ([type]): batch_size*num_neg_samples x hidden_size
        Returns:
            [type]: [description]
        """
        batch_size = query.shape[0]
        hidden_size = query.shape[1]
        num_neg_samples = neg_items.shape[0] // batch_size

        # extend the query for mapping with any number of neg samples
        extend_query = query.repeat(1, num_neg_samples).reshape(-1, hidden_size)

        pos_scores = torch.sum(torch.mul(query, pos_items), axis=1) # batch_size
        if num_neg_samples > 1:
            # extend pos_scores to match with neg scores
            pos_scores = pos_scores.repeat(num_neg_samples, 1).permute(1,0).reshape(-1)
        # print('pos_scores', pos_scores)
        neg_scores = torch.sum(torch.mul(extend_query, neg_items), axis=1)
        # print('neg_scores', neg_scores)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        
        return mf_loss