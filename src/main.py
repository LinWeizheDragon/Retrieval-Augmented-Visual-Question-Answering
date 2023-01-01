"""
main.py:  
    Start functions
        - Read json/jsonnet config files
        - Parse args and override parameters in config files
        - Find selected data loader and initialize
        - Run Trainer to perform training and testing
"""

import os
import argparse
from tabnanny import verbose
import torch
import wandb
import json
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
logger = logging.getLogger(__name__)

from utils.config_system import process_config
from utils.dirs import *
from utils.cuda_stats import print_cuda_statistics
from utils.seed import set_seed
from utils.metrics_log_callback import MetricsHistoryLogger

from data_loader_manager import *
from trainers import *



def get_checkpoint_model_path(saved_model_path, load_epoch=-1, load_best_model=False, load_model_path=""):

    if load_model_path:
        path_save_model = load_model_path
        if not os.path.exists(path_save_model):
            raise FileNotFoundError("Model file not found: {}".format(path_save_model))
    else:
        if load_best_model:
            file_name = "best.ckpt"
        else:
            if load_epoch == -1:
                file_name = "last.ckpt"
            else:
                file_name = "model_{}.ckpt".format(load_epoch)

        path_save_model = os.path.join(saved_model_path, file_name)
        if not os.path.exists(path_save_model):
            logger.warning("No checkpoint exists from '{}'. Skipping...".format(path_save_model))
            logger.info("**First time to train**")
            return '' # return empty string to indicate that no model is loaded
        else:
            logger.info("Loading checkpoint from '{}'".format(path_save_model))
    return path_save_model


def main(config):
    pprint(config)
    if config.seed:
        set_seed(config.seed)
        seed_everything(config.seed, workers=True)
        # sets seeds for numpy, torch and python.random.
        logger.info(f'All seeds have been set to {config.seed}')
    
    DataLoaderWrapper = globals()[config.data_loader.type]
    if DataLoaderWrapper is not None:
        # init data loader
        data_loader_manager = DataLoaderWrapper(config)
    else:
        raise ValueError(f"Data loader {config.data_loader.type} not found")

    # Default logger
    tb_logger = TensorBoardLogger(
        save_dir=config.tensorboard_path,
        name=config.experiment_name
    )
    # Wandb logger
    logger.info('init wandb logger with the following settings: {}'.format(config.WANDB))
    wandb_logger = WandbLogger(config=config, **config.WANDB)
    
    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.saved_model_path,
        every_n_epochs=config.train.save_interval,
        save_top_k=config.train.additional.save_top_k,
        monitor=config.train.additional.save_top_k_metric if 'save_top_k_metric' in config.train.additional.keys() else None,
        mode=config.train.additional.save_top_k_mode,
        filename='model_{epoch}',
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    metrics_history_logger = MetricsHistoryLogger()

    # Get plugins
    plugin_names = config.train.additional.plugins
    plugins = [globals()[plugin_name]() for plugin_name in plugin_names]

    additional_args = {
        'accumulate_grad_batches': config.train.additional.gradient_accumulation_steps,
        "default_root_dir": config.saved_model_path,
        'max_epochs': config.train.epochs,
        # 'limit_train_batches': 2 if config.data_loader.dummy_dataloader else 1.0,
        # 'limit_val_batches': 2 if config.data_loader.dummy_dataloader else 1.0,
        # 'limit_test_batches': 2 if config.data_loader.dummy_dataloader else 1.0,
        'logger': [tb_logger, wandb_logger, metrics_history_logger],
        'callbacks': [checkpoint_callback],
        'plugins': plugins,
        'log_every_n_steps': 10,
        # 'accelerator': "cpu", 
        # 'strategy': "ddp",
        # 'devices': 2,
    }

    trainer = Trainer.from_argparse_args(args, **additional_args)
    logger.info(f"arguments passed to trainer: {str(args)}")
    logger.info(f"additional arguments passed to trainer: {str(additional_args)}")
    
    # Find checkpoints in saved_model_path
    if config.mode == 'train':
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=config.saved_model_path,
            load_model_path=config.train.load_model_path, 
            load_epoch=config.train.load_epoch, 
            load_best_model=config.train.load_best_model
        )
    else:
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=config.saved_model_path,
            load_model_path=config.test.load_model_path, 
            load_epoch=config.test.load_epoch, 
            load_best_model=config.test.load_best_model
        )
        if not checkpoint_to_load:
            raise FileNotFoundError("No checkpoint found. Please check your config file.")

    
    # init data loader manager
    data_loader_manager.build_dataset()
    data_loader_manager.set_dataloader()


    if config.mode == 'train':
        # init train excecutor
        Train_Executor = globals()[config.train.type]
        executor = Train_Executor(config, data_loader_manager)
        # After Initialization, save config files
        with open(os.path.join(config.experiment_path, "config.jsonnet"), 'w') as config_file:
            save_config = config.copy()
            # save_config.pop('device') # Not serialisable
            json.dump(save_config, config_file, indent=4)
            logger.info(f'config file was successfully saved to {config.experiment_path} for future use.')
        
        # Start training
        trainer.fit(
            executor,
            ckpt_path=checkpoint_to_load,
        )
    
    else:
        # init train excecutor
        Train_Executor = globals()[config.train.type]
        executor = Train_Executor(config, data_loader_manager)
        # Start testing
        trainer.test(
            executor,
            ckpt_path=checkpoint_to_load,
        )




def initialization(args):
    assert args.mode in ['train', 'test', 'run']
    # ===== Process Config =======
    config = process_config(args)

    print(config)
    if config is None:
        return None
    # Create Dirs
    dirs = [
        config.log_path,
    ]
    if config.mode == 'train':
        dirs += [
            config.saved_model_path,
            config.imgs_path,
            config.tensorboard_path
        ]
    if config.mode == 'test':
        dirs += [
            config.imgs_path,
            config.results_path,
        ]

    delete_confirm = 'n'
    if config.reset and config.mode == "train":
        # Reset all the folders
        print("You are deleting following dirs: ", dirs, "input y to continue")
        delete_confirm = input()
        if delete_confirm == 'y':
            for dir in dirs:
                try:
                    delete_dir(dir)
                except Exception as e:
                    print(e)
            # Reset load epoch after reset
            config.train.load_epoch = 0
        else:
            print("reset cancelled.")

    create_dirs(dirs)
    print(dirs)

    # ====== Set Logger =====
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s (in %(pathname)s:%(lineno)d)"
    log_console_format = "[%(levelname)s] - %(name)s : %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))
    from utils.color_logging import CustomFormatter
    custom_output_formatter = CustomFormatter(custom_format=log_console_format)
    console_handler.setFormatter(custom_output_formatter)

    info_file_handler = RotatingFileHandler(os.path.join(config.log_path, 'info.log'), maxBytes=10 ** 6,
                                            backupCount=5)
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(Formatter(log_file_format))

    exp_file_handler = RotatingFileHandler(os.path.join(config.log_path, 'debug.log'), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(os.path.join(config.log_path, 'error.log'), maxBytes=10 ** 6,
                                                    backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(info_file_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

    # setup a hook to log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            if wandb.run is not None:
                logger.error(f"Attempting to stop the wandb run {wandb.run}")
                wandb.finish() # stop wandb if keyboard interrupt is raised
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            
        logger.error(f"Uncaught exception: {exc_type} --> {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception

    # setup wandb
    WANDB_CACHE_DIR = config.WANDB.pop('CACHE_DIR')
    if WANDB_CACHE_DIR:
        os.environ['WANDB_CACHE_DIR'] = WANDB_CACHE_DIR
    
    # add base_model as a tag
    config.WANDB.tags.append(config.model_config.base_model)
    # add modules as tags
    config.WANDB.tags.extend(config.model_config.modules)

    all_runs = wandb.Api(timeout=19).runs(path=f'{config.WANDB.entity}/{config.WANDB.project}',  filters={"config.experiment_name": config.experiment_name})
    if config.reset and config.mode == "train" and delete_confirm == 'y':
        for run in all_runs:
            logger.info(f'Deleting wandb run: {run}')
            run.delete()
        config.WANDB.name=config.experiment_name
    else:
        if len(all_runs) > 0:
            config.WANDB.id=all_runs[0].id
            config.WANDB.resume="must"
            config.WANDB.name=config.experiment_name
        else:
            config.WANDB.name=config.experiment_name
    
    # if wandb.run.resumed:
    #     logger.info('Resuming the wandb experiment...')
    # else:
    #     logger.info('Creating new wandb experiment...')
    
    # if config.args.accelerator is None:
    #     # By default, use all GPUs available! (args.accelerator not set)
    #     # This lists all gpu devices being used
    #     config.cuda_device = [i for i in range(torch.cuda.device_count())]
    #     # Enable GPU training
    #     config.args.devices="auto"
    #     config.args.accelerator="auto"
    #     logger.info(f'Instruct to use {config.args.devices} GPUs')

    # # set cuda flag
    # is_cuda = torch.cuda.is_available()
    # if is_cuda and not config.cuda:
    #     logger.warning("WARNING: You have a CUDA device, so you should probably enable CUDA")

    # cuda = is_cuda & config.cuda

    # if cuda:
    #     # Using GPUs
    #     # This is the main GPU device
    #     config.device = torch.device("cuda:0")
    #     # torch.cuda.set_device(config.gpu_device)

    #     # This lists all gpu devices being used
    #     config.cuda_device = [i for i in range(torch.cuda.device_count())]
    #     if len(config.cuda_device) > 1:
    #         # Multiple GPUs
    #         config.using_data_parallel = True
    #     else:
    #         config.using_data_parallel = False
        
    #     logger.info("Program will run on *****GPU-CUDA***** ")
    #     print_cuda_statistics()
    # else:
    #     config.cuda_device = []
    #     config.device = torch.device("cpu")
    #     config.using_data_parallel = False
    #     logger.info("Program will run on *****CPU*****\n")

    logger.info(f'Initialization done with the config: {str(config)}')
    return config

def parse_args_sys(args_list=None):
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    arg_parser.add_argument('--DATA_FOLDER', type=str, default='', help='The path to data.')
    arg_parser.add_argument('--EXPERIMENT_FOLDER', type=str, default='', help='The path to save experiments.')
    
    arg_parser.add_argument('--mode', type=str, default='', help='train/test')
    arg_parser.add_argument('--reset', action='store_true', default=False, help='Reset the corresponding folder under the experiment_name')
    
    arg_parser.add_argument('--experiment_name', type=str, default='', help='Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.')
    arg_parser.add_argument("--tags", nargs='*', default=[], help="Add tags to the wandb logger")
    arg_parser.add_argument('--modules', type=str, nargs="+", default=[], help='Select modules for models. See training scripts for examples.')
    arg_parser.add_argument('--log_prediction_tables', action='store_true', default=False, help='Log prediction tables.')

    # ===== Testing Configuration ===== #
    arg_parser.add_argument('--test_batch_size', type=int, default=-1)
    arg_parser.add_argument('--test_evaluation_name', type=str, default="")
    
    
    arg_parser = Trainer.add_argparse_args(arg_parser)

    arg_parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    return args

if __name__ == '__main__':
    args = parse_args_sys()
    print(args)
    config = initialization(args)
    if config is None:
        exit(0)
    main(config)