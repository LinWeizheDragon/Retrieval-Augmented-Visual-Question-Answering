'''
Logging module for framework
'''
import os
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter


class LoggerSystem():
    def __init__(self):
        print('loggerSystem Created.')

    def init_logger(self, config):
        self.config = config

        log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
        log_console_format = "[%(levelname)s]: %(message)s"

        # Main logger
        main_logger = logging.getLogger()
        main_logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(Formatter(log_console_format))

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
        self.main_logger = main_logger

    def print(self, content, *args, **kwargs):
        if 'mode' in kwargs.keys():
            mode = kwargs['mode']
        else:
            mode = 'info'
        text = '{}  '.format(content)
        for arg in args:
            text += '{}   '.format(arg)
        if mode =='info':
            self.main_logger.info(text)
        elif mode =='debug':
            self.main_logger.debug(text)
        elif mode == 'warning':
            self.main_logger.warning(text)
        elif mode == 'error':
            self.main_logger.error(text)

logger = LoggerSystem()