// This is the base environment file
// It serves as default values for all other jsonnet config files
// Please override these values dirrectly in corresponding config files


// Default values for training control
local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;

// data path configuration
local wandb_cache_dir = ''; //'/home/wl356/rds/rds-wjb31-nmt2020/wl356/wandb_cache';
local default_cache_folder = '../data/fvqa/cache';
local vqa_data = {
  "question_files": {
    "full": '../data/fvqa/all_qs_dict_release.json',
  },
  "kg_files": {
    "full": '../data/fvqa/all_fact_triples_release.json',
  },
  "split_files": {
    "full": "../data/fvqa/Name_Lists",
  }
};
local VinVL_features = {
  "train": "../data/fvqa/pre-extracted_features/vinvl_output/vinvl_fvqa_trainset/inference/vinvl_vg_x152c4/predictions.tsv",
  "test": "../data/fvqa/pre-extracted_features/vinvl_output/vinvl_fvqa_testset/inference/vinvl_vg_x152c4/predictions.tsv",
};
local img_data = {
  "full": "../data/fvqa/images",
};
local caption_features = {
  "train": "../data/fvqa/pre-extracted_features/captions/train_predictions.json",
  "test": "../data/fvqa/pre-extracted_features/captions/test_predictions.json",
};


{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "WANDB": {
    "CACHE_DIR":  wandb_cache_dir,
    "entity": "weizhelin",
    "project": "VQA",
    "tags": ["FVQA"],
  },
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "RAG",
    "pretrained": 1,
    "modules": [],
    "input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "rag_modules": {
      "module_list":[],
    },
    "decoder_input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "output_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
  },
  "cache":{
    "default_folder": default_cache_folder,
    "regenerate":{
      "vinvl_feature_preprocessed": 0,
      "ocr_feature_preprocessed": 0,
      "train_data_preprocessed": 0,
      "test_data_preprocessed": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQAWithKnowledge",
    "dataset_type": "OKVQADataset",
    "dummy_dataloader": 0,
    "additional":{},
    "dataset_modules": {
      "module_list": [],
      "module_dict":{   // all available modules
        "LoadVinVLFeatures":{
          "type": "LoadVinVLFeatures", "option": "default", 
          "config": VinVL_features,
        },
        "LoadOscarCaptionFeatures": {
          "type": "LoadOscarCaptionFeatures", "option": "default",
          "config": caption_features,
        },
        "LoadFVQAData": {
          "type": "LoadFVQAData", "option": "default",
          "config": {
            "use_split": "0",
            "vqa_data_path": vqa_data,
            "image_data_path": img_data,
          },
        },
        "LoadFVQAPassageData": {
          "type": "LoadFVQAPassageData", "option": "default",
          "config": {
            "use_split": "0",
            "vqa_data_path": vqa_data,
          },
        },
        "LoadFVQAAnnotations": {
          "type": "LoadFVQAAnnotations", "option": "default",
          "config": {
            "use_split": "0",
            "vqa_data_path": vqa_data,
          },
        },
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "RagExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch":-1,
    "save_interval":save_interval,
    "load_model_path": "",
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
        "plugins": [],
        "save_top_k": 1,
        "save_top_k_metric": "test/accuracy",
        "save_top_k_mode": "max",
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "load_model_path": "",
    "additional": {
        "multiprocessing": 4,
    },
  }
}