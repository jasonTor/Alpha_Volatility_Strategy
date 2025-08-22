import numpy as np
from scipy.stats import norm
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class BaseStrategy(ABC):

    def __init__(self, market_data):
        self.market_data = market_data

    @abstractmethod
    def generate_alpha(self, straddle_row):
        pass

    @abstractmethod
    def should_trade(self, straddle_row):
        pass