import os
import yaml
import logging

def load_config(yaml_file):
    f = open(yaml_file, 'r')
    config = yaml.safe_load(f)
    return config

def check_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

import logging

class Logger():
    def __init__(self, log_file=None, log_level=logging.INFO):
        self.log_file = log_file
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self._setup_stream_handler()
        if self.log_file is not None:
            self._setup_file_handler()
    
    def set_output_file(self, log_file):
        self.log_file = log_file
        self._setup_file_handler()

    def _setup_stream_handler(self):
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.log_level)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def _setup_file_handler(self):
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def critical(self, msg):
        self.logger.critical(msg)
    
    def exception(self, msg):
        self.logger.exception(msg)
    
    def set_level(self, level):
        self.logger.setLevel(level)
        self.stream_handler.setLevel(level)
        if self.log_file is not None:
            self.file_handler.setLevel(level)