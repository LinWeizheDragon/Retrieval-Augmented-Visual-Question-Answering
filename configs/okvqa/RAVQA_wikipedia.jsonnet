local base_env = import 'RAVQA_base.jsonnet';


// data configuration
local index_passages_path = '';
local index_path = "";
local index_dataset = 'wiki_dpr';
local index_dataset_split = 'train';
local index_name = 'exact';
local index_dummy = 0;
local RAG_data_config_wiki = {
  "index_passages_path": index_passages_path,
  "index_path": index_path,
  "index_dataset": index_dataset,
  "index_dataset_split": index_dataset_split,
  "index_name": index_name,
  "index_dummy": index_dummy,
};


local override = {
  "model_config": {
    "modules": [
    ],
    "RAVQA_loss_type": "Approach6", // NoPR, Approach[1-6]
    "loss_ratio":{
      "nll_loss": 1,
      "additional_loss": 1,
      "rag_loss": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQA",
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
      ],
      "module_dict":{
      },
    },
    "index_files": RAG_data_config_wiki,
  },
};

std.mergePatch(base_env, override)
