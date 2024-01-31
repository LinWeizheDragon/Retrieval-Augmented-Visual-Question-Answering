import collections

from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

class MetricsHistoryLogger(Logger):
    """
    This is a logger that logs the metrics history, since PyTorch Lightning does not directly support this feature
    It logs the metrics history in `self.history`, as long as the metric name does not end with '_auto_max' or '_auto_min'
    Refer to trainer.log_max_and_min_metrics() for automatically adding the max and min metrics
    """
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list) # copy not necessary here  
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "metrics_log_collector"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if metric_name.endswith('_auto_max') or metric_name.endswith('_auto_min'):
                # these are added automatically by this class
                continue
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else: # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (not len(self.history['epoch']) or    # len == 0:
                    not self.history['epoch'][-1] == metric_value) : # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value)
                else:
                    pass
        return

    def log_hyperparams(self, params):
        pass
