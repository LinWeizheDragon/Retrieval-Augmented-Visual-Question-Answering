# Retrieval Augmented Visual Question Answering
This is the official repository of the Retrieval Augmented Visual Question Answering (RAVQA) project.

If our work (including the software provided) helped your research, please kindly cite our paper at EMNLP 2022:

```
Weizhe Lin and Bill Byrne. 2022. Retrieval Augmented Visual Question Answering with Outside Knowledge. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11238–11254, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
```

If you use the TRiG model, please additionally cite the TRiG paper at CVPR 2022:
```
Gao, Feng, et al. "Transform-Retrieve-Generate: Natural Language-Centric Outside-Knowledge Visual Question Answering." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
```

## Table of Content
<!-- TOC -->

- [Retrieval Augmented Visual Question Answering](#retrieval-augmented-visual-question-answering)
  - [Table of Content](#table-of-content)
- [News](#news)
- [Benchmarks](#benchmarks)
- [Resources](#resources)
- [Detail Instructions](#detail-instructions)
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
  - [Download Datasets](#download-datasets)
    - [COCO images](#coco-images)
    - [OKVQA Dataset](#okvqa-dataset)
    - [Google Search Corpus](#google-search-corpus)
    - [F-VQA Dataset](#f-vqa-dataset)
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
  - [Dense Passage Retrieval](#dense-passage-retrieval)
    - [Training](#training-1)
      - [OK-VQA](#ok-vqa)
      - [F-VQA](#f-vqa)
    - [Generating Static Retrieval Results by Testing](#generating-static-retrieval-results-by-testing)
      - [OK-VQA](#ok-vqa-1)
      - [F-VQA](#f-vqa-1)
    - [Prepare FAISS index files for dynamic DPR retrieval](#prepare-faiss-index-files-for-dynamic-dpr-retrieval)
      - [OK-VQA](#ok-vqa-2)
      - [F-VQA](#f-vqa-2)
  - [Baseline models without DPR for retrieval](#baseline-models-without-dpr-for-retrieval)
    - [RA-VQA-NoDPR (T5 baseline)](#ra-vqa-nodpr-t5-baseline)
      - [OK-VQA](#ok-vqa-3)
      - [F-VQA](#f-vqa-3)
  - [Baseline models with DPR](#baseline-models-with-dpr)
    - [TRiG](#trig)
      - [OK-VQA](#ok-vqa-4)
      - [F-VQA](#f-vqa-4)
  - [RAVQA framework](#ravqa-framework)
    - [RA-VQA-FrDPR](#ra-vqa-frdpr)
    - [RA-VQA-NoPR](#ra-vqa-nopr)
    - [RA-VQA](#ra-vqa)
    - [RA-VQA-NoCT](#ra-vqa-noct)
    - [RA-VQA on Wikipedia](#ra-vqa-on-wikipedia)
  - [Some Notes](#some-notes)

<!-- /TOC -->

# News
- [08/02/2023] Our work for creating adversarial samples for the FVQA dataset is accepted to appear at EACL 2023. The dataset and codes will be released here soon.
- [01/01/2023] We released an initial version of our work. The framework supports:
    - RA-VQA-NoDPR (T5 baseline)
    - RA-VQA-FrDPR (DPR retriever + T5 reader)
    - RA-VQA (joint training of DPR + T5)
    - TRiG (Our replication of TRiG)
    - Datasets: OK-VQA and F-VQA
- [19/12/2022] We plan to release the code within Dec, 2022. The author is currently overwhelmed by internship work. Thanks for waiting!
- [12/12/2022] We plan to release the code of our reproduced TRiG system as well.

# Benchmarks
Using the provided codebase, it is expected to obtain the following results.

| Model  | VQA Score | Exact Match | Notes |
|--------|-----------|-------------|-------|
| TRiG   | 50.44     |             |       |
| RA-VQA | 54.51     | 59.65       |       |

Since we refactored the codebase to use pytorch-lightining, these numbers may not match exactly to what were reported in the paper. The author is currently too busy to run all replications. We will add them soon. For now, you can refer to our paper for reported numbers.

# Resources

Packed pre-extracted data for both OK-VQA and F-VQA (including OCR features, VinVL object detection features, Oscar captioning features): [Google Drive](https://drive.google.com/file/d/1fDsoZDVtN0mXeWCKvGA9ITZo2GULu_La/view?usp=share_link)

Pre-trained DPR checkpoint: 
- DPR pretrained on OK-VQA and Google Search dataset (batch size 30, in-batch negative sampling, 1 GPU, grad accumulation 4) [Google Drive](https://drive.google.com/file/d/1Nwx-7e0aZVyXL3GxLvFIink0khbyE0QY/view?usp=share_link)

# Detailed Instructions

## Overview
The framework was designed and implemented by Weizhe Lin, University of Cambridge. All rights are reserved. Use with research purposes is allowed. This framework is designed for **research purpose**, with flexibility for extension. It is not a perfect framework for production, of course.

The training and testing are backboned by pytorch-lightning. The pre-trained Transformer models are from Huggingface-transformers. The training platform is Pytorch.

### Structure
The framework consists of:

1. **main.py**: the main program. It loads a config file and override some entries with command-line arguments. It initialises a data loader wrapper, a model trainer, and a pytorch-lightning trainer to execute training and testing.
2. **Data Loader Wrapper**: it loads the data according to `data_modules` defined in config files. `.set_dataloader()` is called after data loading is finished. `.train_dataloader` and `.test_dataloader` are loaded.
3. **Datasets**: they are automatically loaded by the data loader wrapper. `.collate_fn` is defined to collate the data. An decorator class `ModuleParser` is used to help generate the training inputs. This decorator class generates input dict according to configs (`config.model_config.input_modules/decorder_input_modules/output_modules`).
4. **Model Trainers**: a pytorch-lightning `LightningModule` instance. It defines training/testing behaviors (training steps, optimizers, schedulers, logging, checkpointing, and so on). It initialises the model being trained at `self.model`.
5. **Models**: pytorch `nn.Modules` models.

### Configs
The configuration is achieved with `jsonnet`. It enables inheritance of config files. For example, `RAVQA.jsonnet` override its configs to `RAVQA_base.jsonnet`, which again inherits from `base_env.jsonnet` where most of important paths are defined.

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
We use WANDB for logging in this framework. You will need to register a WANDB account, and change the WANDB config in `base_env.jsonnet`:
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
- All trainer parameters supported by pytorch-lightning, such as `--accelerator gpu --devices 8 --strategy ddp --num_sanity_val_steps 2`
- `--experiment_name EXPERIMENT_NAME` the name of the experiment. Will be used as the name of the folder as well as the run name on WANDB
- `--mode [train/test]` indicate the mode for running. create_data and run are used for Computron runs
- `--modules module1 module2 module3 ...` list of modules that will be used. They will be saved to `self.config.model_config.modules` so that they are accessible anywhere in the framework.
- `--log_prediction_tables`: logs validation/test model outputs to WANDB.
- `--tags tag_a tag_b tag_c`: adds tags to the WANDB run.

#### Training

- `--opts [list of configurations]` used at the end of the cli command. `self.config` will be overwritten by the configurations here. For example:

  - `train.batch_size=1` batch size
  - `train.scheduler=linear` currently supports none/linear
  - `train.epochs=20`
  - `train.lr=0.00002`
  - `train.retriever_lr=0.00001`
  - `train.additional.gradient_accumulation_steps=4` 
  - `train.additional.warmup_steps=0`
  - `train.additional.early_stop_patience=7`
  - `train.additional.save_top_k=1`
  - `valid.step_size=1`
  - `valid.batch_size=4`
  - `data_loader.additional.num_knowledge_passages=5`: an example of how you can change `K` in RAVQA training

#### Testing

- `--test_evaluation_name test_set` this will creates a folder under the experiment folder (indicated by `--experiment_name`) and save everything there. Also, in the WANDB run (run name indicated by `--experiment_name`), a new section with this name (`test_set`) will be created, and the evaluation scores will be logged into this section.
- `--opts test.batch_size=32` 
- `--opts test.load_epoch=6` which checkpoint to load. Note that you need to have the same experiment name


## Environments
Create virtualenv:
```
conda create -n RAVQA python=3.8
conda activate RAVQA
```
Install Pytorch:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Install other libraries:
```
pip install transformers==4.12.5
conda install -c pytorch faiss-gpu -y
pip install wandb pytorch-lightning jsonnet easydict pandas scipy opencv-python fuzzywuzzy scikit-image matplotlib timm scikit-learn sentencepiece tensorboard
pip install setuptools==59.5.0
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

### F-VQA Dataset
[Official repository](https://github.com/wangpengnorman/FVQA)

Data can be saved to `data/fvqa/`:
```
├── Name_Lists
├── all_fact_triples_release.json
├── all_qs_dict_release.json
├── images
├── kg_surface_facts.csv
```


## Feature Extraction
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
For FVQA dataset:
```
python tools/test_sg_net.py \
    --config-file sgg_configs/vgattr/vinvl_x152c4_fvqa_testset.yaml  \
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
    --config-file sgg_configs/vgattr/vinvl_x152c4_fvqa_trainset.yaml  \
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
For FVQA dataset
```
python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml ../scene_graph_benchmark/datasets/fvqa_for_oscar/test.yaml \
    --per_gpu_eval_batch_size 16 \
    --num_beams 5 \
    --max_gen_length 20 \
    --output_prediction_path './output/test_predictions.json' \
    --eval_model_dir /mnt/e/projects/Oscar/pretrained_models/coco_captioning_large_scst/checkpoint-4-50000
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

## Dense Passage Retrieval
### Training
We have extended DPR training to support multiple GPUs using DDP.
#### OK-VQA
This checkpoint has been provided earlier in this document.
```
python main.py ../configs/okvqa/DPR.jsonnet \
    --mode train \
    --experiment_name OKVQA_DPR_FullCorpus  \
    --accelerator auto --devices auto  \
    --strategy ddp \
    --modules exhaustive_search_in_testing \
    --opts train.epochs=10 \
            train.batch_size=8 \
            valid.step_size=1 \
            valid.batch_size=32 \
            train.additional.gradient_accumulation_steps=4 \
            train.lr=0.00001
```
#### F-VQA
This checkpoint is not provided since there are 5 models for 5 splits respectively. You can easily train your own DPR models, and pick the checkpoints you want to load. We recommend running jobs in parallel using slurm/bash.
```
python main.py ../configs/fvqa/DPR.jsonnet \
     --mode train \
     --experiment_name FVQA_DPR_split_0 \
     --accelerator auto --devices auto \
     --opts train.epochs=10 \
             train.batch_size=8 \
             valid.step_size=1 \
             valid.batch_size=32 \
             train.additional.gradient_accumulation_steps=4 \
             train.lr=0.00001 \
             train.additional.early_stop_patience=5 \
             data_loader.dataset_modules.module_dict.LoadFVQAData.config.use_split="0"
```
### Generating Static Retrieval Results by Testing
#### OK-VQA
This checkpoint has been provided earlier in this document.

Testing set:
```
python main.py ../configs/okvqa/DPR.jsonnet \
    --mode test \
    --experiment_name OKVQA_DPR_FullCorpus \
    --accelerator auto --devices 1 \
    --test_evaluation_name generate_test_set \
    --opts train.batch_size=64 \
            valid.batch_size=64 \
            test.load_epoch=06
```
Training set:
```
python main.py ../configs/okvqa/DPR.jsonnet \
    --mode test \
    --experiment_name OKVQA_DPR_FullCorpus \
    --accelerator auto --devices 1 \
    --test_evaluation_name generate_train_set \
    --opts train.batch_size=64 \
            valid.batch_size=64 \
            test.load_epoch=06 \
            data_loader.use_dataset=train
```
#### F-VQA
You need to modify `test.load_epoch=x` to the epoch you want to load. We use epoch0 for example.

Testing set:
```
CUDA_VISIBLE_DEVICES=5 python main.py ../configs/fvqa/DPR.jsonnet \
    --mode test \
    --experiment_name FVQA_DPR_split_0 \
    --accelerator auto --devices 1 \
    --test_evaluation_name generate_test_set \
    --opts train.batch_size=64 \
            valid.batch_size=64 \
            test.load_epoch=0 \
            data_loader.dataset_modules.module_dict.LoadFVQAData.config.use_split="0"
```
Training set:
```
CUDA_VISIBLE_DEVICES=5 python main.py ../configs/fvqa/DPR.jsonnet \
    --mode test \
    --experiment_name FVQA_DPR_split_0 \
    --accelerator auto --devices 1 \
    --test_evaluation_name generate_train_set \
    --opts train.batch_size=64 \
            valid.batch_size=64 \
            test.load_epoch=0 \
            data_loader.use_dataset=train \
            data_loader.dataset_modules.module_dict.LoadFVQAData.config.use_split="0"
```

### Prepare FAISS index files for dynamic DPR retrieval
#### OK-VQA
```
python tools/prepare_faiss_index.py  \
    --csv_path ../data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus_title.csv \
    --output_dir  ../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-new-framework \
    --dpr_ctx_encoder_model_name /path/to/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/item_encoder \
    --dpr_ctx_encoder_tokenizer_name /path/to/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/item_encoder_tokenizer \
```
#### F-VQA
```
python tools/prepare_faiss_index.py  \
    --csv_path ../data/fvqa/kg_surface_facts.csv \
    --output_dir  ../data/fvqa/pre-extracted_features/faiss/fvqa-passages-full \
    --dpr_ctx_encoder_model_name /additional_data/projects/RAVQA/Experiments/FVQA_DPR_split_0/train/saved_model/epoch0/item_encoder \
    --dpr_ctx_encoder_tokenizer_name /additional_data/projects/RAVQA/Experiments/FVQA_DPR_split_0/train/saved_model/epoch0/item_encoder_tokenizer
```

## Baseline models without DPR for retrieval
Note: the OK-VQA evaluation script does not support partial evaluation (it throws an error when the number of questions to be evaluated does not match to the total number of questions in the dataset), you may want to write some additional codes to gather model predictions from other GPUs to RANK 0. In our experiments, one single A100 GPU was sufficient, and thus we did not put effort in this extension. We may consider adding support for this feature in our later updates.

### RA-VQA-NoDPR (T5 baseline)

#### OK-VQA
```
python main.py ../configs/okvqa/baseline_T5.jsonnet \
    --mode train \
    --experiment_name OKVQA_RA-VQA-NoDPR  \
    --accelerator auto --devices 1  \
    --opts train.epochs=10  \
            train.batch_size=1  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.scheduler=linear
```
#### F-VQA
```
python main.py ../configs/fvqa/baseline_T5.jsonnet \
    --mode train \
    --experiment_name FVQA_RA-VQA-NoDPR_split_0  \
    --accelerator auto --devices 1  \
    --opts train.epochs=20  \
            train.batch_size=1  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00008  \
            train.scheduler=linear \
            data_loader.dataset_modules.module_dict.LoadFVQAData.config.use_split="0"
```

## Baseline models with DPR
For models using static DPR outputs, pre-trained DPR features (derived from "Generating Static Retrieval Results by Testing") should be configured at the config file.
Can override `data_loader.dataset_modules.module_dict.LoadPretrainedDPROutputForGoogleSearchPassage.config.pretrained_dpr_outputs` or simply change the path in `base_env.jsonnet`:
```
local pretrained_dpr_features = {
  "train": "/path/to/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/test/test_evaluation/train_predictions.json",
  "test": "/path/to/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/test/test_evaluation/test_predictions.json",
};
```
Then run the training script.

### TRiG
#### OK-VQA
```
python main.py ../configs/okvqa/TRiG.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_TRiG  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10 \
            train.batch_size=1 \
            valid.step_size=1 \
            valid.batch_size=32 \
            train.additional.gradient_accumulation_steps=32 \
            train.lr=0.00006 \
            train.retriever_lr=0.00001 \
            train.scheduler=linear \
            data_loader.additional.num_knowledge_passages=5
```
#### F-VQA
```
python main.py ../configs/fvqa/TRiG.jsonnet  \
    --mode train  \
    --experiment_name FVQA_TRiG_split_0  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10 \
            train.batch_size=1 \
            valid.step_size=1 \
            valid.batch_size=32 \
            train.additional.gradient_accumulation_steps=32 \
            train.lr=0.00006 \
            train.retriever_lr=0.00001 \
            train.scheduler=linear \
            data_loader.additional.num_knowledge_passages=5 \
            data_loader.dataset_modules.module_dict.LoadFVQAData.config.use_split="0" \
            data_loader.dataset_modules.module_dict.LoadPretrainedDPROutputForGoogleSearchPassage.config.pretrained_dpr_outputs.train="../Experiments/FVQA_DPR_split_0/test/generate_train_set/generate_train_set_predictions.json" \
            data_loader.dataset_modules.module_dict.LoadPretrainedDPROutputForGoogleSearchPassage.config.pretrained_dpr_outputs.test="../Experiments/FVQA_DPR_split_0/test/generate_test_set/generate_test_set_predictions.json"
```

## RAVQA framework
Here, we load the index file to dynamically retrieve documents in training with the fast search of FAISS. You should specify some paths in the config file `RAVQA_base.jsonnet`:

- Which query encoder to load? It must be a huggingface transformer model (saved by `.save_pretrained()`). We generate the DPR checkpoints during training, which you can directly use here.
```
"QueryEncoderModelVersion": "/path/to/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/query_encoder",
```
- Which index file to use? These files will be generated when you run the steps in [Prepare FAISS index files for dynamic DPR retrieval](#prepare-faiss-index-files-for-dynamic-dpr-retrieval)
```
// data configuration
local RAG_data_config_full = {
  "index_passages_path": "../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset",
  "index_path": "../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset_hnsw_index.faiss",
};
```

### RA-VQA-FrDPR
DPR is frozen during training
```
python main.py ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA-FrDPR_FullCorpus  \
    --accelerator auto --devices 1  \
    --modules freeze_question_encoder force_existence  \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5
```
For FVQA dataset:
```
python main.py ../configs/fvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name FVQA_RA-VQA-FrDPR-split-0  \
    --accelerator auto --devices 1  \
    --tags RA-VQA-FrDPR \
    --modules freeze_question_encoder force_existence  \
    --opts train.epochs=20  \
            train.batch_size=2  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.additional.early_stop_patience=3 \
            train.lr=0.00008  \
            train.retriever_lr=0.00001  \
            train.scheduler=none  \
            data_loader.additional.num_knowledge_passages=5 \
            model_config.QueryEncoderModelVersion=/additional_data/projects/RAVQA/Experiments/FVQA_DPR_split_0/train/saved_model/epoch0/query_encoder \
            data_loader.index_files.index_passages_path=../data/fvqa/pre-extracted_features/faiss/fvqa-passages-split-0/my_knowledge_dataset \
            data_loader.index_files.index_path=../data/fvqa/pre-extracted_features/faiss/fvqa-passages-split-0/my_knowledge_dataset_hnsw_index.faiss
```

### RA-VQA-NoPR
Only model predictions are used to train the retriever:
```
python main.py ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name RA-VQA-NoPR  \
    --accelerator auto --devices 1  \
    --modules force_existence  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=NoPR  \
            data_loader.additional.num_knowledge_passages=5
```

### RA-VQA
Training with both Pseudo Relevance Labels and Model Predictions:
```
python main.py ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA_FullCorpus  \
    --accelerator auto --devices 1  \
    --modules force_existence  \
    --opts train.epochs=10  \
            train.batch_size=1  \
            valid.step_size=1  \
            valid.batch_size=4  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=Approach6  \
            data_loader.additional.num_knowledge_passages=5
```
Testing Example:
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode test  \
    --experiment_name OKVQA_RA-VQA_FullCorpus  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts data_loader.additional.num_knowledge_passages=5  \
            test.load_model_path=../Experiments/OKVQA_RA-VQA_FullCorpus/train/saved_model/epoch_06.ckpt
```

### RA-VQA-NoCT
Customized Targets are not used to improve answer generation:
```
python main.py ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name RA-VQA-NoCT  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=Approach6  \
            data_loader.additional.num_knowledge_passages=5
```

### RA-VQA on Wikipedia
Train RA-VQA with Wikipedia passages; The embeddings of Wikipedia passages are generated by the DPR paper.
```
python main.py ../configs/okvqa/RAVQA_wikipedia.jsonnet  \
    --mode train  \
    --experiment_name RA-VQA_Wikipedia  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=Approach6  \
            data_loader.additional.num_knowledge_passages=5
```

## Some Notes
- For your convenience, we refactored the codebase to incorporate pytorch-lightning as the backbone. This makes model performance different from what we reported in our paper. But you should be able to obtain performance that is close enough. You can tune the hyperparameters on your own, as long as you choose the checkpoints for evaluation fairly.
- There are no validation sets in both datasets. We evaluated systems per epoch and reported reasonable performance. In fact, you may obtain higher performance by setting a shorter validation interval.
- This publication version was made in a rush due to intensive workload that the author currently have. We will add follow-up patches to make codes more readible and ensure reproducibility. (of course, the speed depends on the number of people who are interested in using this framework.)

