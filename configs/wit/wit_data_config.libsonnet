local wit_data_paths = {
  "image_data_path": "/home/wl356/cvnlp_rds/shared_space/vqa_data/KBVQA_data/wit/images",
  "train": [
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00000-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00001-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00002-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00003-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00004-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00005-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00006-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00007-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00008-of-00010.tsv",
    // "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00009-of-00010.tsv",
  ],
  "valid": [
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00000-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00001-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00002-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00003-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00004-of-00005.tsv",
  ],
  "test": [
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00000-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00001-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00002-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00003-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00004-of-00005.tsv",
  ],
};

local wit_data_pipeline = {
  name: 'WITDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadWITData': {
      transform_name: 'LoadWITData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_paths: wit_data_paths,
        only_main_image: false,
      },
    },
    'process:PrepareImagesForWITData': {
      input_node: "input:LoadWITData",
      transform_name: 'PrepareImagesForWITData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_paths: wit_data_paths,
        _fetch_images: false,
      },
    },
    'process:LoadWITPassages': {
      input_node: "process:PrepareImagesForWITData",
      transform_name: 'LoadWITPassages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
    'process:PrepareWITDataForRetrieval': {
      input_node: [
        // "input:LoadWITData",
        "process:PrepareImagesForWITData",
        "process:LoadWITPassages",
      ],
      transform_name: 'PrepareWITDataForRetrieval',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        // iglue_test_file: "/data/wit/iglue_test_en.jsonl"
      },
    },
    'process:SplitWITPassagesForLargeScaleTraining': {
      input_node: [
        "process:PrepareWITDataForRetrieval",
        "process:LoadWITPassages",
      ],
      transform_name: 'SplitWITPassagesForLargeScaleTraining',
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
  },
};

{
  wit_data_pipeline: wit_data_pipeline,
}
