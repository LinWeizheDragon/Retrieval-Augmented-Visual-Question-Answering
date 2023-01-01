local base_env = import 'base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local retriever_lr = 1e-5;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;



local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "DPR",
    "ModelClass": "RetrieverDPR",
    "QueryEncoderModelClass": "DPRQuestionEncoder",
    "QueryEncoderConfigClass": "DPRConfig",
    "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    // "QueryEncoderModelVersion": "/home/wl356/rds/rds-wjb31-nmt2020/wl356/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/train/saved_model/epoch6/query_encoder",
    "ItemEncoderModelClass": "DPRContextEncoder",
    "ItemEncoderConfigClass": "DPRConfig",
    "ItemEncoderModelVersion": "facebook/dpr-ctx_encoder-single-nq-base",
    // "ItemEncoderModelVersion": "/home/wl356/rds/rds-wjb31-nmt2020/wl356/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/train/saved_model/epoch6/item_encoder",
    "TokenizerClass": "DPRQuestionEncoderTokenizer",
    "TokenizerModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "DecoderTokenizerClass": "DPRContextEncoderTokenizer",
    "DecoderTokenizerModelVersion": "facebook/dpr-ctx_encoder-single-nq-base",
    "pretrained": 1,
    "modules": [
      'separate_query_and_item_encoders',
    ],
    "Ks": [1, 5, 10, 20, 50, 80, 100],
    "num_negative_samples": 1,
    "prepend_tokens": {
      "query_encoder": "",
      "item_encoder": "",
    },
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
    "DECODER_SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
        {"type": "TextBasedVisionInput",  "option": "caption",
                  "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
        {"type": "TextBasedVisionInput",  "option": "object", 
                  "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 0,
                  "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessInputTokenization", "option": "default"},
      ],
    },
    "decoder_input_modules": {
      "module_list":[
        {"type": "KnowledgeInput",  "option": "default",
                  "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessDecoderInputTokenization", "option": "default"},
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
  "cache":{
    "regenerate":{
      "vinvl_feature_preprocessed": 0,
      "ocr_feature_preprocessed": 0,
      "train_data_preprocessed": 1,
      "test_data_preprocessed": 1,
    },
  },
  "data_loader": {
    "type": "DataLoaderFVQAWithKnowledge",
    "dataset_type": "FVQADatasetForDPR",
    "dummy_dataloader": 0,
    "use_dataset": "test", // which dataset split (questions) is evaluated in validation/test
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'full_corpus_in_training': true,
      'full_corpus_in_testing': true,
    },
    "dataset_modules": {
      "module_list": [
        "LoadVinVLFeatures",
        "LoadOscarCaptionFeatures",
        "LoadFVQAData",
        "LoadFVQAPassageData",
        "LoadFVQAAnnotations",
      ],
      "module_dict":{
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "DPRExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "save_interval":save_interval,
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
        "save_top_k_metric": "test/recall_at_5",
        "plugins": [],
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "break_interval": break_interval,
    "additional": {
      "save_HF_model": true,
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "additional": {
      "save_HF_model": false,
    },
  },
  "metrics": [
    {'name': 'compute_DPR_scores'},
  ],
};

std.mergePatch(base_env, override)
