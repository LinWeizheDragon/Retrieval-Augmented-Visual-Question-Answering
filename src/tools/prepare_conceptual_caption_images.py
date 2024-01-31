
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import requests


USER_AGENT = get_datasets_user_agent()

num_threads = 128


def fetch_single_image(image_url, timeout=10, retries=0):
    for _ in range(retries + 1):
        try:
            response = requests.get(image_url, stream=True, timeout=timeout)
            if response:
                image = PIL.Image.open(response.raw)
                break
            else:
                image = None
        except Exception:
            image = None
    return image


def get_images(batch, num_threads, timeout=10, retries=0):
    fetch_single_image_with_args = partial(
        fetch_single_image, timeout=timeout, retries=retries
    )
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch_images = list(
            executor.map(fetch_single_image_with_args, batch["image_url"])
        )

    batch["images"] = batch_images

    return batch



con_caps = load_dataset(
    "parquet",
    data_files={
        "train": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/conceptual_captions/pre-extracted-features/conceptual_captions_ViT-L_14@336px_train.parquet",
        # "val": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/conceptual_captions/pre-extracted-features/conceptual_captions_ViT-L_14@336px_validation.parquet",
    },
)

con_caps['train'] = con_caps['train'].remove_columns(['clip_embeddings'])
con_caps['train'] = con_caps['train'].map(
    get_images,
    batched=True,
    batch_size=512,
    fn_kwargs={
        "num_threads": num_threads,
    },
)
out_path = '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/conceptual_captions/pre-extracted-features/conceptual_captions_train.parquet'
print(f"Writing output to {out_path}...")
con_caps['train'].to_parquet(out_path)


