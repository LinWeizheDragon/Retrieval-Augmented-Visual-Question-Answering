local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'okvqa_data_config.libsonnet';
local base = import 'FLMR_base_preload_vision_features.jsonnet';

local okvqa_data = data.okvqa_data_pipeline;

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/checkpoints/colbertv2.0",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DocTokenizer",
    "TokenizerModelVersion": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/checkpoints/colbertv2.0",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
};
local feature_extractor_config = {
};
local image_processor_config = {
  // "vit_image_processor": {
  //   "ImageProcessorClass": "AutoImageProcessor",
  //   "ImageProcessorModelVersion": "google/vit-base-patch16-224-in21k",
  // },
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": "openai/clip-vit-base-patch32",
  },
};

local data_loader = {
  transforms: {
    'process:LoadOKVQAData': {
      regenerate: false,
      setup_kwargs: {
        add_images: false,
      },
    },
    'process:CropRegionOfInterestImages': {
      input_node: [
        'process:LoadOKVQAData',
      ],
      transform_name: 'CropRegionOfInterestImages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        max_objects: 9,
      },
    },
    'process:ExtractImageFeaturesWithViT': {
      input_node: [
        'process:CropRegionOfInterestImages',
      ],
      transform_name: 'ExtractImageFeaturesWithViTv3',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        input_column: "images",
        index_name: "okvqa_encoded_image_features",
        image_processor_config: image_processor_config,
        vit_model_config: {
          "VisionModelConfigClass": "CLIPVisionConfig",
          "VisionModelClass": "CLIPVisionModel",
          "VisionModelVersion": "openai/clip-vit-base-patch32",
        },
        batch_size: 32,
        _num_proc: 4,
        _num_gpu_proc: null,
      },
      // transform_name: 'ExtractImageFeaturesWithViT',
      // regenerate: false,
      // cache: true,
      // setup_kwargs: {
      //   input_column: "images",
      //   image_processor_config: image_processor_config,
      //   vit_model_config: {
      //     "VisionModelConfigClass": "CLIPVisionConfig",
      //     "VisionModelClass": "CLIPVisionModel",
      //     "VisionModelVersion": "openai/clip-vit-base-patch32",
      //   },
      //   batch_size: 32,
      // },
    },
    'input:LoadGoogleSearchAnnotations': {
      input_node: [
        'input:LoadGoogleSearchPassageData',
        'process:CropRegionOfInterestImages',
      ],
      regenerate: false,
    },
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'input:LoadGoogleSearchAnnotations',
      ],
      regenerate: true,
      cache: false,
      setup_kwargs: {
        output_keys: ["okvqa_data"],
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:PrepareGoogleSearchPassages',
        'process:WrapOutputIntoKeys',
        'process:CropRegionOfInterestImages',
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
          // "train_passages": "train_passages",
          "valid_passages": "valid_passages",
          "test_passages": "test_passages",
          "vqa_data_with_dpr_output": "okvqa_data",
          // "vqa_data": "okvqa_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'train',
              use_column: 'okvqa_data',
            },
          ],
          valid: [
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'test',
              use_column: 'okvqa_data',
            },
          ],
          test: [
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'train',
              use_column: 'okvqa_data',
            },
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'test',
              use_column: 'okvqa_data',
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


local okvqa_data_pipeline = std.mergePatch(okvqa_data, data_loader);

local override = {
    data_pipeline: okvqa_data_pipeline,
    executor: {
      init_kwargs: {
        // validation_indexing_source: validation_indexing_source,
      },
    },
    model_config: {
      "vision_embedding_size": 768,
      "num_ROIs": 9,
      "input_modules": {
          "module_list":[
              {"type": "VisionInput",  "option": "from_embeddings", "use_ROI": true},
              // {"type": "EmptyTextInput",  "option": "default"},
              {"type": "QuestionInput",  "option": "default", 
                      "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
              {"type": "TextBasedVisionInput",  "option": "caption",
                      "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
              {"type": "TextBasedVisionInput",  "option": "object", 
                      "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                      "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
          ],
          "postprocess_module_list": [
              {"type": "PostProcessVisionInputFromEmbeddings", "option": "default"},
              {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
          ],
      },
      "decoder_input_modules": {
          "module_list":[
              {"type": "KnowledgeInput",  "option": "default",
                      "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
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
};

std.mergePatch(base, override)
