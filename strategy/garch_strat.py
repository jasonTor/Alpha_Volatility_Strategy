import pandas as pd
import numpy as np

from strategy.base_strat import BaseStrategy
from arch import arch_model

class Garch_strat(BaseStrategy):

    def __init__(self, market_data):
        super().__init__(market_data)
        self.df_straddle = pd.concat([self.market_data.df_train[['Date','IV']], self.market_data.df_validation[['Date','IV']], self.market_data.df_test[['Date','IV']]], axis=0)
    
    def generate_alpha(self, straddle_row):

        id_date = self.market_data.df_price['Date'].tolist().index(straddle_row['Date'])
        df_filtered_price = self.market_data.df_price[:id_date]
        df_straddle_filtered = self.df_straddle[self.df_straddle['Date'] == straddle_row['Date']]

 
        log_returns = np.log(df_filtered_price['Price'] / df_filtered_price['Price'].shift(1)).dropna() * 100  # in %
        model = arch_model(log_returns, vol='Garch', p=1, q=1) # We instantiate the GARCH class.
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=20)
        vol_forecast_garch = np.mean( np.sqrt(forecast.variance.values[-1]) /100) * np.sqrt(252) # It's a float number corresponding to the annualized volatility predicted by the Garch model

        return vol_forecast_garch


    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row) # equal to the predicted annualized volatility
        df_straddle_filtered = self.df_straddle[self.df_straddle['Date'] == straddle_row['Date']]
        condition = all(alpha > df_straddle_filtered['IV']) or all(alpha < df_straddle_filtered['IV'])
        return condition


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        df_straddle_filtered = self.df_straddle[self.df_straddle['Date'] == straddle_row['Date']]
        if (all(alpha < df_straddle_filtered['IV'])) & (straddle_row['IV'] < np.percentile(df_straddle_filtered['IV'], 25)) : 
            return 'LONG'
        elif all(alpha > df_straddle_filtered['IV']) & (straddle_row['IV'] > np.percentile(df_straddle_filtered['IV'], 75)): 
            return 'SHORT' 
