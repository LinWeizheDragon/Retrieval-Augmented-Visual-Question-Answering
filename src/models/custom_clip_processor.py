
from transformers import CLIPImageProcessor
import numpy as np
from typing import Dict, List, Optional, Union
from transformers.image_utils import ChannelDimension, ImageInput, PILImageResampling, is_batched, to_numpy_array, valid_images
from transformers.image_transforms import (
    center_crop,
    convert_to_rgb,
    get_resize_output_image_size,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)

class CustomCLIPImageProcessor(CLIPImageProcessor):
    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.
        Args:
            image (`np.ndarray`):
                Image to normalize.
            image_mean (`float` or `List[float]`):
                Image mean.
            image_std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        print(image.shape, mean)
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)
