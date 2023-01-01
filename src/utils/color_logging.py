from email.policy import default
import logging

class CustomFormatter(logging.Formatter):
    
    default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    def __init__(self, custom_format=default_format, **kwargs):
        
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        self.FORMATS = {
            logging.DEBUG: grey + custom_format + reset,
            logging.INFO: grey + custom_format + reset,
            logging.WARNING: yellow + custom_format + reset,
            logging.ERROR: red + custom_format + reset,
            logging.CRITICAL: bold_red + custom_format + reset
        }
        super().__init__(**kwargs)

    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)