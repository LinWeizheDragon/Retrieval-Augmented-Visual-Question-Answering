import sys
from subprocess import call
import torch
import logging
logger = logging.getLogger(__name__)

def print_cuda_statistics():

    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    try:
        call(["nvidia-smi", "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    except Exception as e:
        logger.info(e)
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))