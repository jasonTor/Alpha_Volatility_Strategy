import pandas as pd
import numpy as np

from strategy.base_strat import BaseStrategy
from arch import arch_model

class Garch_strat(BaseStrategy):

    def __init__(self, market_data):
        super().__init__(market_data)
        self.df_option = pd.concat([self.market_data.df_validation[['Date','IV']], self.market_data.df_test[['Date','IV']] ]) # df_validation + df_test
        self.dict_vol_garch = self.set_vol_garch()


    def set_vol_garch(self):
        df_ess = self.df_option.copy()
        vol_garch = []
        for date in df_ess['Date'].unique().tolist():
            id_date = self.market_data.df_price['Date'].tolist().index(date)
            df_filtered_price = self.market_data.df_price[:id_date]
            log_returns = np.log(df_filtered_price['Price'] / df_filtered_price['Price'].shift(1)).dropna() * 100  # in %

            model = arch_model(log_returns, vol='Garch', p=1, q=1) # We instantiate the GARCH class.
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=30)
            vol_forecast_garch = np.mean( np.sqrt(forecast.variance.values[-1]) /100) * np.sqrt(252) # Annualized
            vol_garch.append(vol_forecast_garch)

        df_garch = pd.DataFrame({'Date': df_ess['Date'].unique().tolist(), 'vol_garch': vol_garch})
        df_merged_ess = df_ess.merge(df_garch, on='Date', how='left')
        df_merged_ess_unique = df_merged_ess[['Date','vol_garch']].drop_duplicates()
        dict_vol_garch = dict(zip(df_merged_ess['Date'],df_merged_ess['vol_garch']))

        return dict_vol_garch


    
    def generate_alpha(self, straddle_row):
        return self.dict_vol_garch[straddle_row['Date']]


    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row) # equal to the predicted annualized volatility
        df_filtered = self.df_option[self.df_option['Date'] == straddle_row['Date']]
        condition1 = alpha < np.min(df_filtered['IV'])
        condition2 = alpha > np.max(df_filtered['IV'])*1.2
        return condition1 or condition2


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row) # equal to the predicted annualized volatility
        df_filtered = self.df_option[self.df_option['Date'] == straddle_row['Date']]
        condition1 = alpha < np.min(df_filtered['IV'])
        condition2 = alpha > np.max(df_filtered['IV'])*1.2
        if condition1 : 
            return 'SHORT'
        elif condition2: 
            return 'LONG' 
