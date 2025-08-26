import pandas as pd
import numpy as np

from strategy.base_strat import BaseStrategy

class Regime_switching_modelfree(BaseStrategy):

    def __init__(self, market_data):
        super().__init__(market_data)
        self.dic_vol_10_train, self.dic_vol_1_train, self.dic_vol_10_validation, self.dic_vol_1_validation = self.set_vol_lookahead()

    def set_vol_lookahead(self):
        def compute_vols(df_subset):
            rv_10, rv_1 = [], []
            for _, row in df_subset.iterrows():
                id_date = self.market_data.df_price['Date'].tolist().index(row['Date'])
                prices_10 = self.market_data.df_price['Price'].iloc[id_date-10:id_date+1]
                prices_2 = self.market_data.df_price['Price'].iloc[id_date-1:id_date+1]

                vol10 = np.sqrt(np.mean(np.diff(np.log(prices_10))**2) * 252)
                vol  = np.sqrt(np.mean(np.diff(np.log(prices_2))**2) * 252)

                rv_10.append(vol10)
                rv_1.append(vol)

            df_subset['rv_lookahead_10d'] = rv_10
            df_subset['rv_lookahead_1d'] = rv_1
            return df_subset

        df_train_sub = self.market_data.df_train[['Date','Price']].drop_duplicates()
        df_train_sub = df_train_sub[self.market_data.df_train['Date'] > " 2016-03-01"].copy() 

        df_validation_sub = self.market_data.df_validation[['Date','Price']].drop_duplicates().copy()

        # df_subset_train = compute_vols(df_train_sub)
        # df_subset_validation = compute_vols(df_validation_sub)

        dic_vol_10_train = dict(zip(df_train_sub['Date'], compute_vols(df_train_sub)['rv_lookahead_10d']))
        dic_vol_1_train = dict(zip(df_train_sub['Date'], compute_vols(df_train_sub)['rv_lookahead_1d']))

        dic_vol_10_validation = dict(zip(df_validation_sub['Date'],compute_vols(df_validation_sub)['rv_lookahead_10d']))
        dic_vol_1_validation = dict(zip(df_validation_sub['Date'],compute_vols(df_validation_sub)['rv_lookahead_1d']))        

        return dic_vol_10_train, dic_vol_1_train, dic_vol_10_validation, dic_vol_1_validation

    
    def generate_alpha(self, straddle_row):
        if straddle_row['Date'] in self.dic_vol_1_train.keys():
            return self.dic_vol_1_train[straddle_row['Date']], self.dic_vol_10_train[straddle_row['Date']]
        elif straddle_row['Date'] in self.dic_vol_1_validation.keys():
            return self.dic_vol_1_validation[straddle_row['Date']], self.dic_vol_10_validation[straddle_row['Date']]


    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = alpha[0] > 2.5*alpha[1] # LONG
        condition2 = alpha[0] < 0.2*alpha[1] # SHORT
        return condition1 or condition2


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = alpha[0] > 2.5*alpha[1] # LONG if the instant vol is suddenly high (threshold 3.1 has been calibrated on exploratory_analysis file)
        condition2 = alpha[0] < 0.2*alpha[1] # SHORT if the instant vol is suddenly low (threshold 0.2 has been calibrated on exploratory_analysis file)
        if condition1 : 
            return 'LONG'
        elif condition2: 
            return 'SHORT' 
