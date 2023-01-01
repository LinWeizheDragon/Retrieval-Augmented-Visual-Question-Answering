local base_env = import 'RAVQA_base.jsonnet';

// data configuration
local RAG_data_config_full = {
  "index_passages_path": "../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset",
  "index_path": "../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset_hnsw_index.faiss",
};

local override = {
  "model_config": {
    "modules": [
    ],
    "RAVQA_loss_type": "Approach6", // NoPR, Approach[1-6]
    "loss_ratio":{
      "nll_loss": 1,
      "additional_loss": 0,
      "rag_loss": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQAWithKnowledge",
    "dataset_type": "OKVQADataset",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'max_target_length':10,
      'num_knowledge_passages': 5,
    },
    "dataset_modules": {
      "module_list": [
        "LoadVinVLFeatures",
        "LoadGoogleOCRFeatures",
        "LoadOscarCaptionFeatures",
        "LoadOKVQAData",
        "LoadGoogleSearchPassageData",
      ],
      "module_dict":{
      },
    },
    "index_files": RAG_data_config_full,
  },
};

std.mergePatch(base_env, override)
