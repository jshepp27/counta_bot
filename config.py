import json
import os
import logging

basedir = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, *args):
        self._config_items = {}

        for arg in args:
            logger.info(f"Adding {arg.name} to configuration setup.")
            self._config_items[arg.name] = arg
        self._config_file_data = None

    def load_config(self, file_name="config.json"):
        with open(file_name) as file:
            logger.debug(f"Loading configuration from {file_name},")
            self._config_file_data = json.load(file)