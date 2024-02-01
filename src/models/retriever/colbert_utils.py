
import os
import time
from tqdm import tqdm
import ujson
import torch
import random
import numpy as np
import torch.multiprocessing as mp

try:
    import faiss
except ImportError as e:
    print("WARNING: faiss must be imported for indexing")

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import Launcher

from colbert.utils.utils import create_directory, print_message

from colbert import Indexer


from colbert.infra.config.config import ColBERTConfig

import colbert.utils.distributed as distributed

from colbert.infra.run import Run
from colbert.infra.launcher import print_memory_stats
# from colbert.modeling.checkpoint import Checkpoint

from colbert.data.collection import Collection

from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.utils import optimize_ivf
from colbert.utils.utils import flatten, print_message

from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.collection_indexer import CollectionIndexer

class MultiModalIndexer(Indexer):
    def __init__(self, checkpoint, config=None):
        """
           Use Run().context() to choose the run's configuration. They are NOT extracted from `config`.
        """

        self.index_path = None
        self.checkpoint = checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)

        self.config = ColBERTConfig.from_existing(self.checkpoint_config, config, Run().config)
        self.configure(checkpoint=checkpoint)

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def get_index(self):
        return self.index_path

    def erase(self):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            delete = delete or filename.endswith(".pt")
            
            if delete:
                deleted.append(filename)
        
        if len(deleted):
            print_message(f"#> Will delete {len(deleted)} files already at {directory} in 3 seconds...")
            time.sleep(3)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name, collection, overwrite=False):
        assert overwrite in [True, False, 'reuse', 'resume']

        self.configure(collection=collection, index_name=name, resume=overwrite=='resume')
        self.configure(bsize=64, partitions=None)

        self.index_path = self.config.index_path_
        index_does_not_exist = (not os.path.exists(self.config.index_path_))

        assert (overwrite in [True, 'reuse', 'resume']) or index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != 'reuse':
            self.__launch(collection)

        return self.index_path

    def __launch(self, collection):
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        launcher = Launcher(encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues)




def encode(config, collection, shared_lists, shared_queues):
    encoder = MultiModalCollectionIndexer(config=config, collection=collection)
    encoder.run(shared_lists)


class MultiModalCollectionIndexer(CollectionIndexer):
    def __init__(self, config: ColBERTConfig, collection):
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        self.use_gpu = self.config.total_visible_gpus > 0

        # if self.config.rank == 0:
        #     self.config.help()

        self.collection = Collection.cast(collection)
        self.checkpoint = MultiModalCheckpoint(self.config.checkpoint, colbert_config=self.config)
        if self.use_gpu:
            self.checkpoint = self.checkpoint.cuda()

        self.encoder = CollectionEncoder(config, self.checkpoint)
        self.saver = IndexSaver(config)

        print_memory_stats(f'RANK:{self.rank}')

    def index(self):
        with self.saver.thread():
            batches = self.collection.enumerate_batches(rank=self.rank)
            for chunk_idx, offset, passages in tqdm(batches, disable=self.rank > 0):
                if self.config.resume and self.saver.check_chunk_exists(chunk_idx):
                    Run().print_main(f"#> Found chunk {chunk_idx} in the index already, skipping encoding...")
                    continue
                embs, doclens = self.encoder.encode_passages(passages)
                if self.use_gpu:
                    assert embs.dtype == torch.float16
                else:
                    assert embs.dtype == torch.float32
                    embs = embs.half()

                Run().print_main(f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                                 f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")

                self.saver.save_chunk(chunk_idx, offset, embs, doclens)
                del embs, doclens



from models.retriever.FLMR import *


from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
import json
from pprint import pprint
from easydict import EasyDict

class MultiModalCheckpoint(ColBERTWithMultimodalDocs):
    """
        Easy inference with ColBERT.

        TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name, colbert_config=None):
        # load global config
        config_path = os.path.join(colbert_config.checkpoint, "global_config.json")
        with open(config_path, "r") as f:
            global_config = EasyDict(json.load(f))

        super().__init__(name, colbert_config, global_config=global_config)
        assert self.training is False

        checkpoint_to_load = os.path.join(colbert_config.checkpoint, 'vision_projection.pt')
        if not checkpoint_to_load or checkpoint_to_load == '':
            print("No checkpoint found.")
        else:
            # We manually load the state dict
            print(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)
            self.vision_projection.load_state_dict(state_dict_from_ckpt)
            print(f"Load the following parameters to vision_projection from the given checkpoint: {state_dict_from_ckpt.keys()}")
        
        checkpoint_to_load = os.path.join(colbert_config.checkpoint, 'doc_vision_projection.pt')
        if not checkpoint_to_load or checkpoint_to_load == '':
            print("No checkpoint found.")
        else:
            # We manually load the state dict
            print(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)
            self.doc_vision_projection.load_state_dict(state_dict_from_ckpt)
            print(f"Load the following parameters to doc_vision_projection from the given checkpoint: {state_dict_from_ckpt.keys()}")
        

        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        # self.skiplist.update({w: True
        #                      for symbol in self.doc_tokenizer.tok.additional_special_tokens
        #                      for w in [symbol, self.doc_tokenizer.tok.encode(symbol, add_special_tokens=False)[0]]})

        self.amp_manager = MixedPrecisionManager(True)

    
    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)

                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, context=context, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=context)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']

        if isinstance(docs[0], tuple):
            image_features = [i for _, i in docs]
            image_features = torch.FloatTensor(np.stack(image_features))
            docs = [doc for doc, _ in docs]
            multimodal_docs = True
        else:
            image_features = None
            multimodal_docs = False

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, batch_image_features=image_features, bsize=bsize)
            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]
            
            # split image_features into batch size chuncks
            if multimodal_docs:
                # image_features_batches = [image_features[i:i + bsize] for i in range(0, len(image_features), bsize)]
                # # convert to torch tensors
                # image_features_batches = [torch.FloatTensor(np.stack(image_features_batch)) for image_features_batch in image_features_batches]
                # text_img_batches = [
                #     (text_batch[0], text_batch[1], image_features_batch) for text_batch, image_features_batch in zip(text_batches, image_features_batches)
                # ]
                text_img_batches = text_batches
                # print(text_img_batches)
                # print('----------------------')
                # print(self.doc_tokenizer.tok.decode(text_img_batches[0][0][34]))
                # print(text_img_batches[0][2][34])

            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            if multimodal_docs:
                batches = [self.doc(input_ids, attention_mask, image_features, keep_dims=keep_dims_, to_cpu=to_cpu)
                        for input_ids, attention_mask, image_features in tqdm(text_img_batches, disable=not showprogress)]
            else:
                batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                        for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == 'flatten':
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)

                D = D[mask.bool().flatten()].cpu()
                
                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)
        
        if multimodal_docs:
            input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
            image_features = torch.FloatTensor(np.stack(image_features))
            print('warning!!!!!')
            return self.doc(input_ids, attention_mask, image_features, keep_dims=keep_dims, to_cpu=to_cpu)
        else:
            input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
            return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)

        assert False, "Implement scoring"

    def score(self, Q, D, mask=None, lengths=None):
        assert False, "Call colbert_score"
        # EVENTUALLY: Just call the colbert_score function!

        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.device).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output