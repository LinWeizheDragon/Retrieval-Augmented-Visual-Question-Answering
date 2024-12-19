# Retrieval-augmented Visual Question Answering with Fine-grained Late-interaction Multi-modal Retrieval

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/preflmr-scaling-up-fine-grained-late/retrieval-on-infoseek)](https://paperswithcode.com/sota/retrieval-on-infoseek?p=preflmr-scaling-up-fine-grained-late)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/preflmr-scaling-up-fine-grained-late/visual-question-answering-vqa-on-infoseek)](https://paperswithcode.com/sota/visual-question-answering-vqa-on-infoseek?p=preflmr-scaling-up-fine-grained-late)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-late-interaction-multi-modal-1/retrieval-on-ok-vqa)](https://paperswithcode.com/sota/retrieval-on-ok-vqa?p=fine-grained-late-interaction-multi-modal-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-late-interaction-multi-modal-1/visual-question-answering-on-ok-vqa)](https://paperswithcode.com/sota/visual-question-answering-on-ok-vqa?p=fine-grained-late-interaction-multi-modal-1)

This is the official repository of the Retrieval Augmented Visual Question Answering (RAVQA) project.
The project covers RAVQA and RAVQA-v2 (equipped with Fine-grained Late-interaction Multi-modal Retrieval).




# 🔥🔥News
- [19/12/2024] 🔥🔥🔥 We released the multilingual version(Chinese + English ) of PreFLMR, you can download PreFLMR ENCN model [here](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L_ENCN).
- [03/09/2024] We have uploaded the images used in the M2KR benchmark [here](https://huggingface.co/datasets/BByrneLab/M2KR_Images) .
- [10/08/2024] We received many requests regarding adding multilingual abilities to PreFLMR. We announce that **we are now training the Chinese version of PreFLMR and will release it very soon**. Stay tuned!
- [05/06/2024] 🔥🔥🔥The PreFLMR paper has been accepted to appear at ACL 2024! The camera-ready version of the paper has been updated [here](https://arxiv.org/abs/2402.08327) to include more details and analyses. Along with the acceptance, we have made some important updates to help you use the model and extend your research easier:
  - Added an evaluation script that reproduces the results in the PreFLMR paper [here](https://github.com/LinWeizheDragon/FLMR?tab=readme-ov-file#new-evaluate-the-preflmr-models-on-all-m2kr-benchmarks)
  - Added the updated benchmark results with the transformer implementation [here](#benchmark-results-for-preflmr-in-the-dedicated-flmr-codebase)
  - Added an example script to fine-tune PreFLMR on a custom retrieval dataset [here](https://github.com/LinWeizheDragon/FLMR?tab=readme-ov-file#new-finetune-the-preflmr-model-on-downstream-datasets)
  - **IMPORTANT**: fixed the OVEN data splits in the M2KR benchmark, and updated each entry with a fixed instruction to ensure the evaluation result is not affected by random sampling of instructions. Please delete your local cache and download the dataset again.
- [13/04/2024] 🔥 We highlight another valuable and concurrent research on training instruction-following, universal, multi-task multi-modal retrievers: [UniIR: Training and Benchmarking Universal Multimodal Information Retrievers](https://tiger-ai-lab.github.io/UniIR/), which was done by the researchers of the University of Waterloo. They also shared the M-Beir benchmark which can be used to train and evaluate multi-modal universal information retrievers. In the near future, we may collaborate to combine the two benchmarks together to facilitate the advance of this field.
- [06/03/2024] 🔥🔥🔥The implementation based on huggingface-transformers is now available [here](https://github.com/linweizhedragon/FLMR)!
- [20/02/2024] 🔥🔥🔥 The [PreFLMR project page](https://preflmr.github.io/) has been launched! Explore a captivating demo showcasing PreFLMR_ViT-G, our largest model yet. Additionally, access pre-trained checkpoints and the M2KR benchmark, designed for assessing general-purpose knowledge retrievers. Stay tuned as we will soon upload a huggingface-compatible implementation along with example scripts for indexing and retrieval, providing effortless access via `FLMRModelForRetrieval.from_pretrained(...)`.
- [14/02/2024] 🔥Our follow-up work, PreFLMR, is now available [here](https://arxiv.org/abs/2402.08327)! PreFLMR is a general-purpose retriever that was pre-trained on more than ten million multi-modal retrieval data and achieved strong performance across a wide range of knowledge-intensive tasks. It can also serve as a strong foundation retrieval model that can be fine-tuned to fit any downstream retrieval tasks. We will release the model through huggingface-transformers very soon, which allows quick deployment in minutes.
- [31/01/2024] 🔥We are happy to announce that the training and testing code for FLMR is now released! For the legacy RAVQA-v1 and the code for FVQA, please checkout to `legacy_v1` or tag `v1.0`. We are also preparing a new FLMR implementation for Huggingface transformers, which will be released as plug-in-and-play models.🔥
- [03/10/2023] Our follow-up work "Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering" has been accepted to appear at NeurIPS 2023! The paper can be found here [here](https://arxiv.org/abs/2309.17133). If you prefer a 3-minute technical summary, look at this [post](https://jinghong-chen.ghost.io/fined-grained-late-interaction-multimodal-retrieval-flmr/). The code will be released in this repository soon. We are happy to announce that we have made a major change to our code framework such that experiment management and data processing are more flexible.
- [01/05/2023] FVQA 2.0 is released [here](FVQA2.0.md).
- [08/02/2023] Our work for creating adversarial samples for the FVQA dataset is accepted to appear at EACL 2023. The dataset and codes will be released here soon.
- [01/01/2023] We released an initial version of our work. The framework supports:
    - RA-VQA-NoDPR (T5 baseline)
    - RA-VQA-FrDPR (DPR retriever + T5 reader)
    - RA-VQA (joint training of DPR + T5)
    - TRiG (Our replication of TRiG)
    - Datasets: OK-VQA and F-VQA
- [19/12/2022] We plan to release the code within Dec, 2022. The author is currently overwhelmed by internship work. Thanks for waiting!
- [12/12/2022] We plan to release the code of our reproduced TRiG system as well.
  

## Table of Content
- [Retrieval-augmented Visual Question Answering with Fine-grained Late-interaction Multi-modal Retrieval](#retrieval-augmented-visual-question-answering-with-fine-grained-late-interaction-multi-modal-retrieval)
- [🔥🔥News](#news)
  - [Table of Content](#table-of-content)
- [Benchmarks](#benchmarks)
  - [Benchmark Results for PreFLMR in the dedicated FLMR codebase](#benchmark-results-for-preflmr-in-the-dedicated-flmr-codebase)
  - [Models and Benchmark Results for the M2KR Amd M2KR-CN (the Chinese split of the M2KR)](#models-and-benchmark-results-for-the-m2kr-amd-m2kr-cn-the-chinese-split-of-the-m2kr)
  - [Benchmark Results for FLMR in this codebase](#benchmark-results-for-flmr-in-this-codebase)
- [Resources](#resources)
- [Detailed Instructions](#detailed-instructions)
  - [Overview](#overview)
    - [Structure](#structure)
    - [Configs](#configs)
    - [ModuleParser](#moduleparser)
    - [MetricsProcessor](#metricsprocessor)
    - [WANDB Logging](#wandb-logging)
    - [Useful Command-line Arguments](#useful-command-line-arguments)
      - [Universal](#universal)
      - [Training](#training)
      - [Testing](#testing)
  - [Environments](#environments)
  - [ElasticSearch](#elasticsearch)
  - [Download Datasets](#download-datasets)
    - [COCO images](#coco-images)
    - [OKVQA Dataset](#okvqa-dataset)
    - [Google Search Corpus](#google-search-corpus)
  - [Feature Extraction](#feature-extraction)
    - [VinVL Features (object detection/attributes/relations)](#vinvl-features-object-detectionattributesrelations)
      - [Step 1: Install environments](#step-1-install-environments)
      - [Step 2: Generating OKVQA datasets](#step-2-generating-okvqa-datasets)
      - [Step 3: Download pre-trained models](#step-3-download-pre-trained-models)
      - [Step 4: Running models](#step-4-running-models)
      - [Step 5: Recommended Save Path](#step-5-recommended-save-path)
    - [Oscar+ Features (image captioning)](#oscar-features-image-captioning)
      - [Step 1: Download data](#step-1-download-data)
      - [Step 2: Download the pre-trained model](#step-2-download-the-pre-trained-model)
      - [Step 3: Running the inference](#step-3-running-the-inference)
      - [Step 4: Recommended Save Path](#step-4-recommended-save-path)
    - [Google OCR Features](#google-ocr-features)
  - [Fine-grained Late-interaction Multi-modal Retrieval](#fine-grained-late-interaction-multi-modal-retrieval)
    - [Pretraining the mapping network with WIT](#pretraining-the-mapping-network-with-wit)
    - [Finetuning FLMR on the GoogleSearch corpus](#finetuning-flmr-on-the-googlesearch-corpus)
    - [Generating static retrieval results for inspection and inference](#generating-static-retrieval-results-for-inspection-and-inference)
  - [BLIP2 with FLMR](#blip2-with-flmr)
- [Some Notes](#some-notes)
- [Citation](#citation)



# Benchmarks

## Benchmark Results for PreFLMR in the dedicated [FLMR](https://github.com/LinWeizheDragon/FLMR) codebase

| Model         | WIT Recall@10 | IGLUE Recall@1 | KVQA Recall@5 | MSMARCO Recall@5 | OVEN Recall@5 | LLaVA Recall@1 | EVQA Recall@5 | EVQA Pseudo Recall@5 | OKVQA Recall@5 | OKVQA Pseudo Recall@5 | Infoseek Recall@5 | Infoseek Pseudo Recall@5 |
|---------------|---------------|----------------|---------------|------------------|---------------|----------------|---------------|----------------------|----------------|-----------------------|-------------------|--------------------------|
| [LinWeizheDragon/PreFLMR_ViT-G🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) | 0.619         | 0.718          | 0.419         | 0.783            | 0.643         | 0.726          | 0.625         | 0.721                | 0.302          | 0.674                 | 0.392             | 0.577                    |
| [LinWeizheDragon/PreFLMR_ViT-L🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) | 0.605         | 0.699          | 0.440         | 0.779            | 0.608         | 0.729          | 0.609         | 0.708                | 0.314          | 0.690                 | 0.374             | 0.578                    |
| [LinWeizheDragon/PreFLMR_ViT-B🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-B) | 0.427         | 0.574          | 0.294         | 0.786            | 0.468         | 0.673          | 0.550         | 0.663                | 0.272          | 0.658                 | 0.260             | 0.496                    |

## Models and Benchmark Results for the M2KR Amd M2KR-CN (the Chinese split of the M2KR)
|                            Model                             | WIT(EN) Recall@10 | WIT(CN) Recall@10 | KVQA(EN) Recall@5 | KVQA(EN) Recall@5 | MSMARCO(EN) Recall@5 | MSMARCO(CN) Recall@5 | OVEN(EN) Recall@5 | OVEN(CN) Recall@5 | LLaVA(EN) Recall@1 | LLaVA(CN) Recall@1 | EVQA(EN) Recall@5 | EVQA(CN) Recall@5 | OKVQA(EN) Recall@5 | OKVQA(CN) Recall@5 | Infoseek(EN) Recall@5 | Infoseek(CN) Recall@5 |
| :----------------------------------------------------------: | :---------------: | :---------------: | :---------------: | :---------------: | :------------------: | :------------------: | :---------------: | :---------------: | :----------------: | :----------------: | :---------------: | :---------------: | :----------------: | :----------------: | :-------------------: | :-------------------: |
| [LinWeizheDragon/PreFLMR_ViT-L🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) |       60.5        |       10.9        |       43.6        |        3.2        |         78.7         |         10.3         |       59.8        |        6.6        |        71.8        |        3.2         |       70.8        |        2.8        |        68.5        |        2.1         |         57.9          |          7.9          |
| [LinWeizheDragon/PreFLMR_ViT-L_ENCN🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L_ENCN) |       60.8        |       83.4        |       41.1       |       37.3       |         82.6         |        82.3         |       60.8        |       58.8       |       71.1        |       58.9        |       58. 0       |       46.6       |       13.9        |       13.3        |         41.9         |         39.7         |

**Note:** We converted the checkpoints from PyTorch to Huggingface-transformers, whose benchmark results differ from the numbers reported in the original paper slightly. You can reproduce the results in the above paper by referring to the instructions in [this document](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md).

The benchmark dataset corpus can be found [here](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR) and the images can be found [here](https://huggingface.co/datasets/BByrneLab/M2KR_Images) 

## Benchmark Results for FLMR in this codebase
Using the provided codebase, it is expected to obtain the following results.

| Model  | Recall@5 | Notes |
|--------|-----------|-------|
| FLMR (9 ROIs) | 89.20     |       |
| FLMR (9 ROIs) | 89.28     |  Using the pretrained ckpt     |

| Model  | VQA Score | Notes |
|--------|-----------|-------|
| RA-VQA | 54.51     | In the previous paper     |
| RA-VQA-v2 | 61.86     | with FLMR     |

Since we refactored the codebase significantly in clean-up, these numbers may not match exactly to what were reported in the paper.

# Resources
We host the data required for running this system in [Huggingface](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data/tree/main) and Baidu Cloud (coming soon). 

The data contains:
- Packed pre-extracted data for OK-VQA (including OCR features, VinVL object detection features, Oscar captioning features)
- FLMR with the mapping network pretrained on WIT (batch size 30, in-batch negative sampling, 1 GPU, grad accumulation 4)
- FLMR pretrained on OK-VQA and Google Search dataset (batch size 30, in-batch negative sampling, 1 GPU, grad accumulation 4)

You can download these resources from Huggingface altogether: [Combined Download on Huggingface](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data/blob/main/RAVQA_v2_data.tar.gz). 
```
wget https://huggingface.co/datasets/BByrneLab/RAVQAV2Data/resolve/main/RAVQA_v2_data.tar.gz?download=true
```

After downloading and extracting the `tar.gz`, you need to unzip all `.zip` files under `okvqa` folder  and `okvqa/pre-extracted/OCR.zip`. 

After otaining all these resources, you should:
- Change the data paths in `configs/okvqa/okvqa_data_config.libsonnet`
- Change the paths to `TokenizerModelVersion` in `configs/okvqa/FLMR_with_ROI.jsonnet`
- Change the paths to `EncoderModelVersion` and `TokenizerModelVersion` in `configs/okvqa/FLMR_base_preload_vision_features.jsonnet`

By downloading the provided OK-VQA data, you must comply with the [OK-VQA license](https://okvqa.allenai.org/download.html) and [MS COCO license](https://cocodataset.org/#termsofuse).

# Detailed Instructions

## Overview
The framework was designed and implemented by Weizhe Lin, University of Cambridge. All rights are reserved. Use with research purposes is allowed. This framework is designed for **research purpose**, with flexibility for extension. It is not a perfect framework for production, of course.

The training and testing are backboned by pytorch-lightning. The pre-trained Transformer models are from Huggingface-transformers. The training platform is Pytorch.

In this release, we designed a new framework that wraps the data processing/training/testing utilities - [Runway For ML](https://github.com/EriChen0615/runway_for_ml/tree/kbvqa_dev). It is a highly efficient framework that enables flexible experimentation and data processing. Data processing is formulated as a Directional Acyclic Graph, on which the framework traverses through nodes to prepare data. This framework enables efficient data processing at million scale. For more details, please refer to the [README](https://github.com/EriChen0615/runway_for_ml/tree/kbvqa_dev) of the framework. 
When cloning this repository, please use the `kbvqa_dev` branch.

The indexing and searching of FLMR is supported by [FAISS](https://github.com/facebookresearch/faiss) and [ColBERT](https://github.com/stanford-futuredata/ColBERT). The ColBERT engine is plugged into this project as a third-party package. We fixed many errors in this package following [LI-RAGE](https://github.com/amazon-science/robust-tableqa).

### Structure
The framework consists of:

1. **main.py**: the main program. It loads a config file and override some entries with command-line arguments. It initialises a `RunwayExperiment` instance to execute training and testing.
2. **Data Ops**: it loads the data according to configs specified in `data_pipeline`. The details of this feature can be found in [here](https://github.com/EriChen0615/runway_for_ml/tree/kbvqa_dev?tab=readme-ov-file#data-preprocessing-1)
3. **Datasets**: they are automatically loaded by the data loader wrapper. `.collate_fn` is defined to collate the data. An decorator class `ModuleParser` is used to help generate the training inputs. This decorator class generates input dict according to configs (`config.model_config.input_modules/decorder_input_modules/output_modules`).
4. **Model Executors**: a pytorch-lightning `LightningModule` instance. It defines training/testing behaviors (training steps, optimizers, schedulers, logging, checkpointing, and so on). It initialises the model being trained at `self.model`.
5. **Models**: pytorch `nn.Modules` models.

### Configs
The configuration is achieved with `jsonnet`. It enables inheritance of config files. For example, `configs/okvqa/FLMR_with_ROI.jsonnet` override its configs to `configs/okvqa/FLMR_base_preload_vision_features.jsonnet`.

By including the corresponding key:value pair in the config file, overriding can be easily performed.

### ModuleParser
A decorator class that helps to parse data into features that are used by models.

An example is shown below:
```
"input_modules": {
    "module_list":[
    {"type": "QuestionInput",  "option": "default", 
                "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},  
    {"type": "TextBasedVisionInput",  "option": "caption",
                "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
    {"type": "TextBasedVisionInput",  "option": "object", 
                "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
    ],
    "postprocess_module_list": [
    {"type": "PostProcessInputTokenization", "option": "default"},
    ],
},
"decoder_input_modules": {
    "module_list":[],
    "postprocess_module_list": [],
},
"output_modules": {
    "module_list":[
    {"type": "GenerationOutput", "option": "default"},
    ],
    "postprocess_module_list": [
    {"type": "PostProcessOutputTokenization", "option": "default"},
    ],
},
```
which first generates text_sequences:
```
<BOQ> Question <EOQ> <BOC> Caption <EOC> <BOV> obj1 attr1 attr2 <SOV> obj2 ... [OCR results] <EOV>
```
in the order defined in `input_modules`, and then the postprocessing unit `PostProcessInputTokenization` is used to tokenize the input into `input_ids` and `input_attention_masks`.

By defining new functions in `ModuleParser`, e.g. `self.TextBasedVisionInput`, a new behavior can be easily introduced to transform modules into training features.

### MetricsProcessor
The following entries in config file `test.metrics` define the metrics to compute in validation and testing. Each module uploads `log_dict` with `metrics_name: metrics_value` which can be processed in trainers conveniently.
```
"metrics": [
    {'name': 'compute_exact_match'},
    {'name': 'compute_retrieval_metrics'},
    {'name': 'compute_okvqa_scores'},
],
```

### WANDB Logging
We use WANDB for logging in this framework. You will need to register a WANDB account, and change the WANDB config in `meta_configs/hpc_meta_config.libsonnet`:
```
  "WANDB": {
    "CACHE_DIR":  wandb_cache_dir,
    "entity": "your account/project account",
    "project": "your project",
    "tags": ["FVQA"], // default tags
  },
```


### Useful Command-line Arguments
Some general cli arguments. For more details, please read the code / directly look at how they are used in training/evaluation of specific models.

#### Universal
- All trainer parameters supported by pytorch-lightning, such as :
```
--opts train.trainer_paras.accelerator=auto \
             train.trainer_paras.devices=auto \
             train.trainer_paras.strategy=ddp_find_unused_parameters_true \
             train.trainer_paras.num_sanity_val_steps=2 \
             train.trainer_paras.max_epochs=10000 \
             train.trainer_paras.val_check_interval=1000 \
             train.trainer_paras.accumulate_grad_batches=2 \
```
- `--experiment_name EXPERIMENT_NAME` the name of the experiment. Will be used as the name of the folder as well as the run name on WANDB
- `--mode [prepare_data/train/test]` indicate the mode for running. prepare_data only runs the data preprocessing pipeline.
- `--modules module1 module2 module3 ...` list of modules that will be used. They will be saved to `self.config.model_config.modules` so that they are accessible anywhere in the framework.
- `--log_prediction_tables`: logs validation/test model outputs to WANDB.
- `--tags tag_a tag_b tag_c`: adds tags to the WANDB run.

#### Training

- `--opts [list of configurations]` used at the end of the cli command. `self.config` will be overwritten by the configurations here. For example:

  - `train.batch_size=30` batch size
  - `train.optimizer_config.scheduler=linear` currently supports none/linear
  - `train.trainer_paras.max_epochs=10000`
  - `train.optimizer_config.optimizer_params.lr=0.00001` learning rate
  - `train.trainer_paras.accumulate_grad_batches=2` 
  - `train.optimizer_config.scheduler_params.num_warmup_steps=0`
  - `train.early_stopping_callback_paras.patience=10`
  - `train.model_checkpoint_callback_paras.save_top_k=1` how many best checkpoints are saved
  - `valid.batch_size=4`
  - `train.trainer_paras.val_check_interval=1000` how many steps between validation

#### Testing

- `--test_suffix XXX` this will creates a folder under the experiment folder (indicated by `--experiment_name`) and save everything there. Also, in the WANDB run (run name indicated by `--experiment_name`), a new section with this name (`XXX`) will be created, and the evaluation scores will be logged into this section.
- `--opts test.batch_size=16` 
- `--opts test.load_epoch=6` which checkpoint to load. Note that you need to have the same experiment name


## Environments
Create virtualenv:
```
conda create -n RAVQA python=3.8
conda activate RAVQA
```
Install Pytorch:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install other libraries:
```
pip install transformers==4.28.1
conda install -c pytorch faiss-gpu -y
pip install setuptools==59.5.0
pip install wandb pytorch-lightning==2.0.4 jsonnetbin easydict pandas scipy opencv-python fuzzywuzzy scikit-image matplotlib timm scikit-learn sentencepiece tensorboard datasets
pip install ujson evaluate GPUtil easydict peft==0.4.0
pip install bitarray spacy ujson gitpython ninja absl-py openai sacrebleu
cd third_party/ColBERT
pip install -e .
```

## ElasticSearch
To speed up training and inference, this codebase supports pre-computing image features (including ROI features). These features can be overwhelming if saved to the local disk as individual files. Thus, we install ElasticSearch to index all images and their extracted features.

1. Download [ElasticSearch](https://www.elastic.co/downloads/elasticsearch) and unzip

2. Run ElasticSearch at a separate thread and keep it running in the background:
```
./bin/elasticsearch
```

3. In the first launch, note down the password

4. Before running data processing scripts, set the environment variables:
```
export ELASTIC_CA_CERTS="/path/to/elasticsearch-8.7.0/config/certs/http_ca.crt"
export ELASTIC_PASSWORD="YOUR PASSWORD"
```

## Download Datasets
Note that we provide a zip file containing all data here: [Resources](#resources)

### COCO images
`data/ok-vqa/train2014`: [Train images](http://images.cocodataset.org/zips/train2014.zip)

`data/ok-vqa/val2014`: [Test images](http://images.cocodataset.org/zips/val2014.zip)

### OKVQA Dataset
`data/ok-vqa/mscoco_train2014_annotations.json`: [Training annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)

`data/ok-vqa/mscoco_val2014_annotations.json`: [Testing annotations](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip)

`data/ok-vqa/OpenEnded_mscoco_train2014_questions.json`: [Training questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)

`data/ok-vqa/OpenEnded_mscoco_val2014_questions.json`: [Testing questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip)

### Google Search Corpus
[Official download link](https://drive.google.com/drive/folders/15uWx33RY5UmR_ZmLO6Ve1wyzbXsLxV6o?usp=sharing)

Data can be saved to `data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv`.



## Feature Extraction
We provide the pre-extracted features for OK-VQA dataset. If you want to re-extract the features or extract features for other datasets, please follow the instructions below.
### VinVL Features (object detection/attributes/relations)
#### Step 1: Install environments
VinVL needs a separate env.

Refer to [Offical installation guide](https://github.com/microsoft/scene_graph_benchmark/blob/main/INSTALL.md)

Since HPC uses A-100, which requires a higher version of CUDA, the recommended environment with CUDA 10.1 does not work.

```
conda create --name sg_benchmark python=3.7 -y
conda activate sg_benchmark
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install ipython h5py nltk joblib jupyter pandas scipy -y
pip install ninja yacs>=0.1.8 cython matplotlib tqdm opencv-python numpy>=1.19.5 
python -m pip install cityscapesscripts
pip install pycocotools scikit-image timm einops
cd materials/scene_graph_benchmark
python setup.py build develop
```


#### Step 2: Generating OKVQA datasets
```
cd materials/scene_graph_benchmark
python tools/prepare_data_for_okvqa.py
```
This command generates trainset/testset of OKVQA datasets to `datasets/okvqa/`, which will be used in object detection.

#### Step 3: Download pre-trained models
```
mkdir models
mkdir models/vinvl
/path/to/azcopy copy https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth ./models/vinvl/
```

#### Step 4: Running models
`vinvl_vg_x152c4` is a pre-trained model with object and attribute detection:
For OKVQA dataset:
```
python tools/test_sg_net.py \
    --config-file sgg_configs/vgattr/vinvl_x152c4_okvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_x152c4_okvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```

`vinvl_large` is a pre-trained model with **only** object detection. But it was pre-trained on more object detection datasets!
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_large_okvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_large.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_large_okvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_large.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```

#### Step 5: Recommended Save Path
The object/attribute data can be saved to `data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv`.

### Oscar+ Features (image captioning)
#### Step 1: Download data
We can download COCO-caption data with azcopy:
```
cd materials/Oscar
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_caption' ./oscar_dataset --recursive
```
Reference: [offical download page](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md)

#### Step 2: Download the pre-trained model
We can download [COCO captioning large](https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/image_captioning/coco_captioning_large_scst.zip) here, or refer to the [official download page](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md#Image-Captioning-on-COCO) for the model checkpoints.

Save the pre-trained model to `pretrained_models/coco_captioning_large_scst`.

#### Step 3: Running the inference
```
python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml oscar_dataset/coco_caption/[train/val/test].yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --output_prediction_path './output/[train/val/test]_predictions.json' \
    --eval_model_dir pretrained_models/coco_captioning_large_scst/checkpoint-4-50000
```

Note that in the script, `transformer` is renamed to `transformer2` such that it won't conflict with existing `transformer` package in your environment.

#### Step 4: Recommended Save Path
The data can be saved to `data\ok-vqa\pre-extracted_features\captions\train_predictions.json`.


### Google OCR Features
First, enable Google OCR APIs; download the key file to `google_ocr_key.json`. This is **not** free! Ask me for the already generated features.
```
cd src
python ocr.py
```
The detected features will be saved to `data/ok-vqa/pre-extracted_features/OCR`.

## Fine-grained Late-interaction Multi-modal Retrieval
**IMPORTANT NOTE**: In the following sections, first you need to run the provided scripts with `--mode train` changed to `--mode prepare_data`. This runs the data preprocessing and save the processing results to the cache folder. Then, you will be able to reuse these cache files in later runs. If you want to re-run some of the data nodes, open the data config file (e.g. `configs/okvqa/okvqa_data_config.libsonnet`) and change `regenerate=False` to `True`, and then rerun the script with `--mode prepare_data`. You will see that the nodes with `regenerate=True`, along with their downstream nodes,  are re-generated.

### Pretraining the mapping network with WIT

```
python src/main.py \
    --experiment_name "FLMR_Pretraining(WIT)_MappingNetwork(32)" \
    --config "configs/wit/FLMR_WIT_pretraining.jsonnet" \
    --reset --override \
    --mode train \
    --opts train.trainer_paras.accelerator=auto \
             train.trainer_paras.devices=auto \
             train.trainer_paras.strategy=ddp_find_unused_parameters_true \
             train.trainer_paras.num_sanity_val_steps=2 \
             train.trainer_paras.max_epochs=10000 \
             train.batch_size=30 \
             train.trainer_paras.val_check_interval=1000 \
             valid.batch_size=16 \
             train.trainer_paras.accumulate_grad_batches=2 \
             train.early_stopping_callback_paras.patience=10 \
             train.optimizer_config.optimizer_params.lr=0.00001 \
             train.optimizer_config.scheduler=none \
```
A pretrained checkpoint is provided earlier in this document. You don't have to run the pretraining on your own.

### Finetuning FLMR on the GoogleSearch corpus
```
python src/main.py \
    --experiment_name "OKVQA_FLMR_9ROI_with_text_based_vision_fix_lens" \
    --config "configs/okvqa/FLMR_with_ROI.jsonnet" \
    --reset --override \
    --mode train \
    --opts train.trainer_paras.accelerator=auto \
             train.trainer_paras.devices=auto \
             train.trainer_paras.strategy=ddp_find_unused_parameters_true \
             train.trainer_paras.num_sanity_val_steps=2 \
             train.trainer_paras.max_epochs=10000 \
             train.batch_size=30 \
             train.trainer_paras.val_check_interval=1000 \
             valid.batch_size=16 \
             train.trainer_paras.accumulate_grad_batches=2 \
             train.early_stopping_callback_paras.patience=10 \
             train.optimizer_config.optimizer_params.lr=0.00001 \
             train.optimizer_config.scheduler=none \
             model_config.num_ROIs=9 \
             train.load_model_path="checkpoint_path" \
```
`checkpoint_path` is the path to either the checkpoint saved during the pretraining in the previous step, or the pretrained checkpoint `WIT_pretrained_ckpt.ckpt`.


### Generating static retrieval results for inspection and inference
```
python src/main.py \
    --experiment_name "OKVQA_FLMR_9ROI_with_text_based_vision_generate_index" \
    --config "configs/okvqa/FLMR_with_ROI.jsonnet" \
    --reset --override \
    --test_suffix generate_index \
    --mode test \
    --opts test.trainer_paras.accelerator=auto \
             test.trainer_paras.devices=auto \
             test.trainer_paras.strategy=ddp_find_unused_parameters_true \
             test.batch_size=16 \
             model_config.num_ROIs=10 \
             train.load_model_path="checkpoitn_path" \
             data_pipeline.transforms.input:LoadGoogleSearchAnnotations.setup_kwargs.use_all_samples=1 \
```
## BLIP2 with FLMR
The static results are generated in the previous step:
```
"/path/to/experiments/OKVQA_FLMR_9ROI_with_text_based_vision_generate_index/test/generate_index/generate_index_test_OKVQADatasetForDPR.test_predictions_rank_0.json",
"/path/to/experiments/OKVQA_FLMR_9ROI_with_text_based_vision_generate_index/test/generate_index/generate_index_test_OKVQADatasetForDPR.train_predictions_rank_0.json",
```
Change the config file `configs/rag/okvqa/RAG_BLIP2_with_FLMR.jsonnet`
```
local index_files = {
  "index_path": "",
  "embedding_path": "",
  "static_results": [
    "/path/to/experiments/OKVQA_FLMR_9ROI_with_text_based_vision_generate_index/test/generate_index/generate_index_test_OKVQADatasetForDPR.test_predictions_rank_0.json",
    "/path/to/experiments/OKVQA_FLMR_9ROI_with_text_based_vision_generate_index/test/generate_index/generate_index_test_OKVQADatasetForDPR.train_predictions_rank_0.json",
  ],
};
```
Note: this framework also supports retrieving passages dynamically. Due to time constraints, we are not able to provide a hit-to-run instruction for that feature. Users are encouraged to explore this feature if they are interested in using RAVQA-v2 with joint training (similar to RAVQA-v1).


Now you can run training as follows:
```
python src/main.py \
    --experiment_name "OKVQA_RAG_BLIP2(t5-xl)_FLMR(10ROI)" \
    --config "configs/rag/okvqa/RAG_BLIP2_with_FLMR.jsonnet" \
    --modules static_retrieval force_existence \
    --reset --override \
    --mode train \
    --opts train.trainer_paras.accelerator=auto \
             train.trainer_paras.devices=auto \
             train.trainer_paras.strategy=ddp_find_unused_parameters_true \
             train.trainer_paras.num_sanity_val_steps=2 \
             train.trainer_paras.max_epochs=9999999 \
             train.trainer_paras.precision="bf16" \
             train.batch_size=1 \
             train.trainer_paras.val_check_interval=500 \
             valid.batch_size=16 \
             train.trainer_paras.accumulate_grad_batches=16 \
             train.early_stopping_callback_paras.patience=5 \
             train.optimizer_config.optimizer_params.lr=0.0001 \
             train.optimizer_config.scheduler=none \
             train.model_checkpoint_callback_paras.save_top_k=1 \
             model_config.num_beams=2 \
             model_config.num_knowledge_passages=5 \
             model_config.num_knowledge_passages_in_training=5 \
```
If you encounter GPU OOM errors, try reducing `num_knowledge_passages_in_training` to reduce the passages used in each forward pass. If `num_knowledge_passages_in_training < num_knowledge_passages (K)`, random passages will be drawn from top-K retrieved documents.


# Some Notes
- This publication version was made in a rush due to intensive workload that the author currently have. We will add follow-up patches to make codes more readible and ensure reproducibility. (of course, the speed depends on the number of people who are interested in using this framework.)
- Before applying the system to your own task, you may find it useful to read the author's note in `third_party/ColBERT/colbert/search/index_storage.py: Line 67`.

# Citation

If our work (including the software provided) helped your research, please kindly cite our paper at NeurIPS 2023 and EMNLP 2022:
```
@inproceedings{lin-etal-2024-preflmr,
    title = "{P}re{FLMR}: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers",
    author = "Lin, Weizhe  and
      Mei, Jingbiao  and
      Chen, Jinghong  and
      Byrne, Bill",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.289",
    pages = "5294--5316",
    abstract = "Large Multimodal Models (LMMs) excel in natural language and visual understanding but are challenged by exacting tasks such as Knowledge-based Visual Question Answering (KB-VQA) which involve the retrieval of relevant information from document collections to use in shaping answers to questions. We present an extensive training and evaluation framework, M2KR, for KB-VQA. M2KR contains a collection of vision and language tasks which we have incorporated into a single suite of benchmark tasks for training and evaluating general-purpose multi-modal retrievers. We use M2KR to develop PreFLMR, a pre-trained version of the recently developed Fine-grained Late-interaction Multi-modal Retriever (FLMR) approach to KB-VQA, and we report new state-of-the-art results across a range of tasks. We also present investigations into the scaling behaviors of PreFLMR intended to be useful in future developments in general-purpose multi-modal retrievers.",
}
```
```
@inproceedings{
  lin2023finegrained,
  title={Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering},
  author={Weizhe Lin and Jinghong Chen and Jingbiao Mei and Alexandru Coca and Bill Byrne},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=IWWWulAX7g}
}
```
```
@inproceedings{lin-etal-2023-fvqa,
    title = "{FVQA} 2.0: Introducing Adversarial Samples into Fact-based Visual Question Answering",
    author = "Lin, Weizhe  and
      Wang, Zhilin  and
      Byrne, Bill",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.11",
    pages = "149--157",
    abstract = "The widely used Fact-based Visual Question Answering (FVQA) dataset contains visually-grounded questions that require information retrieval using common sense knowledge graphs to answer. It has been observed that the original dataset is highly imbalanced and concentrated on a small portion of its associated knowledge graph. We introduce FVQA 2.0 which contains adversarial variants of test questions to address this imbalance. We show that systems trained with the original FVQA train sets can be vulnerable to adversarial samples and we demonstrate an augmentation scheme to reduce this vulnerability without human annotations.",
}
```
```
@inproceedings{lin-byrne-2022-retrieval,
    title = "Retrieval Augmented Visual Question Answering with Outside Knowledge",
    author = "Lin, Weizhe  and
      Byrne, Bill",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.772",
    doi = "10.18653/v1/2022.emnlp-main.772",
    pages = "11238--11254",
    abstract = "Outside-Knowledge Visual Question Answering (OK-VQA) is a challenging VQA task that requires retrieval of external knowledge to answer questions about images. Recent OK-VQA systems use Dense Passage Retrieval (DPR) to retrieve documents from external knowledge bases, such as Wikipedia, but with DPR trained separately from answer generation, introducing a potential limit on the overall system performance. Instead, we propose a joint training scheme which includes differentiable DPR integrated with answer generation so that the system can be trained in an end-to-end fashion. Our experiments show that our scheme outperforms recent OK-VQA systems with strong DPR for retrieval. We also introduce new diagnostic metrics to analyze how retrieval and generation interact. The strong retrieval ability of our model significantly reduces the number of retrieved documents needed in training, yielding significant benefits in answer quality and computation required for training.",
}
```
