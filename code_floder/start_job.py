# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

from xgboost_dengue_ODI import *


class Strat_job():

    def __init__(self, config):
        self.config=config

    def start(self):
        try:


