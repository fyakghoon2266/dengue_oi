# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import configparser
from function_process import function_process

from function_process import function_process

logger.info('start')

def load_data(self):

    data_ovtrip = function_process(config=config)

    logger.info('======== loading data .... ========')
