import sys
sys.path.append('.')

from runway_for_ml.utils import config_system as rw_conf
from runway_for_ml.configs import configuration as rw_cfg
from runway_for_ml.data_module.data_pipeline import DataPipeline
from runway_for_ml.utils.global_variables import Executor_Registry
from runway_for_ml.experiment import RunwayExperiment
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from utils.seed import set_seed
from easydict import EasyDict
from pprint import pprint
import argparse

def parse_sys_args(args_list=None):
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        '--config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format'
    )
    arg_parser.add_argument('--experiment_name', type=str, default='', help='Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.')
    arg_parser.add_argument('--from_experiment', type=str, default='', help="The Experiment name from which the new experiment inherits/overwrites config")
    arg_parser.add_argument('--test_suffix', type=str, default='', help='Tests will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$/test/$test_suffix$.')
    
    arg_parser.add_argument('--mode', type=str, default='prepare_data', help='prepare_data/train/test')
    arg_parser.add_argument('--reset', action='store_true', default=False, help='Reset the corresponding folder under the experiment_name')
    arg_parser.add_argument('--override', action='store_true', default=False, help='Danger. Force yes for reset=1')

    arg_parser.add_argument("--tags", nargs='*', default=[], help="Add tags to the wandb logger")
    arg_parser.add_argument('--modules', type=str, nargs="+", default=[], help='Select modules for models. See training scripts for examples.')
    arg_parser.add_argument('--log_prediction_tables', action='store_true', default=False, help='Log prediction tables.')
    arg_parser.add_argument(
        "--log_prediction_tables_with_images",
        action="store_true",
        default=False,
        help="Use WANDB artifacts to logged WANDB tables.",
    )
    arg_parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        default=False,
        help="Whether to enable dummy data mode",
    )
    arg_parser.add_argument(
        "--wandb_artifacts",
        type=str,
        default="weizhelin/vqa-images-open/VQAv2-Images:v0",
        help="Which WANDB artifacts to use",
    )

    # arg_parser = Trainer.add_argparse_args(arg_parser)
    
    arg_parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if args_list is None:
        sys_args = arg_parser.parse_args()
    else:
        sys_args = arg_parser.parse_args(args_list)

    return sys_args


def _process_sys_args(config_dict, sys_args):
    # for key in vars(sys_args):
    #     if key == 'opts': continue
    #     value = getattr(sys_args, key)
    #     config_dict[key] = value
    config_dict.reset = sys_args.reset
    config_dict.override = sys_args.override
    config_dict.from_experiment = sys_args.from_experiment
    config_dict.test_suffix = sys_args.test_suffix
    config_dict.mode = sys_args.mode
    if sys_args.experiment_name != '':
        config_dict.experiment_name = sys_args.experiment_name
    
    config_dict.meta.WANDB.tags = config_dict.meta.WANDB.tags + sys_args.tags
    config_dict.use_dummy_data = sys_args.use_dummy_data
    
    # Custom
    config_dict.log_prediction_tables = sys_args.log_prediction_tables
    config_dict.model_config.modules += sys_args.modules

    # change args to dict, and save to config
    def namespace_to_dict(namespace):
        return EasyDict({
            k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
            for k, v in vars(namespace).items()
        })
    config_dict.args = namespace_to_dict(sys_args)

    _process_optional_args(config_dict, sys_args.opts)
        
def _process_optional_args(config_dict, opts):
    if opts is None: return
    for opt in opts:
        splited = opt.split('=')
        path, value = splited[0], '='.join(splited[1:])
        try:
            value = eval(value)
        except:
            value = str(value)
            print('input value {} is not a number, parse to string.')
        config_key_list = path.split('.')
        item = config_dict
        for key in config_key_list[:-1]:
            # assert key in item, f"Optional args error: {opt} does not exists. Error with key={key}"
            if key not in item:
                item[key] = EasyDict() 
            item = item[key]
        item[config_key_list[-1]] = value
            

def parse_config(config_path, sys_args):
    config_dict = rw_conf.get_config_from_json(config_path)
    _process_sys_args(config_dict, sys_args)
    return config_dict    

def prepare_data_main(config_dict):
    meta_config = rw_cfg.MetaConfig.from_config(config_dict)
    # dp_config = rw_cfg.DataPipelineConfig.from_config(config_dict, meta_config)
    dp_config = config_dict.data_pipeline
    dp = DataPipeline(dp_config, global_config=config_dict)
    dp_config_dict = config_dict.data_pipeline
    output_data = {}
    if 'out_ops' in dp_config_dict:
        output_data = dp.get_data(dp_config_dict['out_ops'])
    else:
        output_data = dp.apply_transforms()

    
def train_main(config_dict):
    print("Runway Training...")
    rw_experiment = RunwayExperiment(config_dict)
    rw_experiment.train()

def test_main(config_dict):
    print("Runway Testing...")
    rw_experiment = RunwayExperiment(config_dict)
    rw_experiment.test()

def eval_main(config_dict):
    print("Runway Evaluating...")
    rw_experiment = RunwayExperiment(config_dict)
    rw_experiment.eval()

if __name__ == '__main__':
    
    print("Runway main started.")
    sys_args = parse_sys_args()
    config_dict = parse_config(sys_args.config, sys_args)
    print("Configuration Loaded.")
    pprint(config_dict)
    rw_conf.import_user_modules()
    print("User modules imported")
    mode = sys_args.mode
    if mode == 'prepare_data':
        prepare_data_main(config_dict)
    elif mode == 'train':
        train_main(config_dict)
    elif mode == 'test':
        test_main(config_dict)
    elif mode == 'eval':
        eval_main(config_dict)




    