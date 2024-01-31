local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'wit_data_config.libsonnet';
local wit_data = data.wit_data_pipeline;

local pretrained_ckpt_path = "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/checkpoints/colbertv2.0";

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DocTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
};
local feature_extractor_config = {
};
local image_processor_config = {
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": "openai/clip-vit-base-patch32",
  },
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////
local data_loader = {
  transforms: {
    'process:ExtractImageFeaturesWithViT': {
      input_node: [
        'process:PrepareImagesForWITData',
      ],
      transform_name: 'ExtractImageFeaturesWithViTv3',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        input_column: "images",
        index_name: "encoded_wit_image_features",
        image_processor_config: image_processor_config,
        vit_model_config: {
          "VisionModelConfigClass": "CLIPVisionConfig",
          "VisionModelClass": "CLIPVisionModel",
          "VisionModelVersion": "openai/clip-vit-base-patch32",
        },
        batch_size: 32,
      },
    },
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'process:PrepareWITDataForRetrieval',
      ],
      regenerate: true,
      cache: false,
      setup_kwargs: {
        output_keys: ["wit_data"],
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:SplitWITPassagesForLargeScaleTraining',
        'process:PrepareImagesForWITData',
        // 'process:PrepareWITDataForRetrieval',
        'process:WrapOutputIntoKeys',
        'process:ExtractImageFeaturesWithViT',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: {
          "passages": "train_passages",
          "images": "images",
          "image_dataset_with_embeddings": "image_dataset_with_embeddings",
        },
        pass_columns: {
          "valid_passages": "valid_passages",
          "test_passages": "test_passages",
          "vqa_data_with_dpr_output": "wit_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'train',
              use_column: 'wit_data',
            },
          ],
          valid: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'valid',
              use_column: 'wit_data',
            },
          ],
          test: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'valid',
              use_column: 'wit_data',
            },
          ],
        },
        tokenizer_config: tokenizer_config,
        feature_extractor_config: feature_extractor_config,
        image_processor_config: image_processor_config,
      },
    },
  },
};

local data_pipeline = std.mergePatch(wit_data, data_loader);

{
    experiment_name: 'default_DPR',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: data_pipeline,
    model_config: {
        "base_model": "FLMR",
        "ModelClass": "FLMRForPretraining",
        "EncoderModelVersion": pretrained_ckpt_path,
        "VisionModelConfigClass": "CLIPVisionConfig",
        "VisionModelClass": "CLIPVisionModel",
        "VisionModelVersion": "openai/clip-vit-base-patch32",
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
        ],
        "Ks": [5, 10, 20, 50, 80, 100, 500],
        "nbits": 8,
        "num_negative_samples": 1,
        "max_source_length":512,
        "max_decoder_source_length": 512,
        "full_corpus_in_training": true,
        "full_corpus_in_testing": false,
        "mapping_network_prefix_length": 32,
        "vision_embedding_size": 768,
        "lm_embedding_size": 128,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_embeddings"},
                {"type": "EmptyTextInput",  "option": "default"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputFromEmbeddings", "option": "default"},
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '', 'end': ''}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
            ],
        },
        "output_modules": {
            "module_list":[
                {"type": "SimilarityOutput", "option": "default"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessConcatenateLabels", "option": "default"},
            ],
        },
    },
    executor: {
        ExecutorClass: 'FLMRVisionPretrainingExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
        },
    },
    train: {
        batch_size: 8,
        num_dataloader_workers: 0,
        trainer_paras: {
            max_epochs: 100,
            accumulate_grad_batches: 4,
            check_val_every_n_epoch: null,
            val_check_interval: 10,
            log_every_n_steps: 10,
        },
        model_checkpoint_callback_paras: {
            monitor: 'valid/WITDatasetForDPR.valid/pos_item_ids_recall_at_10',
            save_top_k: 3,
            mode: "max",
            filename: 'model_step_{step}',
            save_last: true,
            verbose: true,
            auto_insert_metric_name: false,
            save_on_train_epoch_end: false,
        },
        early_stopping_callback_paras: {
            patience: 3,
            verbose: true,
            mode: "max",
        },
        optimizer_config: {
            optimizer_name: "AdamW",
            optimizer_params: {
                lr: 0.00001,
                eps: 1e-08,
            },
            scheduler: "none",
            scheduler_params: {
                num_warmup_steps: 0,
            },
        },
    },
    valid: {
        batch_size: 64,
        num_dataloader_workers: 0,
    },
    test: {
        checkpoint_name: "",
        load_model_path: "",
        load_best_model: false,
        trainer_paras: {},
        batch_size: 64,
        num_dataloader_workers: 0,
    },
    eval: {
        'eval_op_name': 'Your eval op name'
    },
    "metrics": [
        {'name': 'compute_DPR_scores_with_pos_ids'},
    ],
}
