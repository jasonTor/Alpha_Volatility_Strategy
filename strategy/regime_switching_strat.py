import pandas as pd
import numpy as np

from strategy.base_strat import BaseStrategy

from hmmlearn import hmm

class Regime_switching_modelfree(BaseStrategy):

    def __init__(self, market_data):
        super().__init__(market_data)
        self.dic_vol_10_train, self.dic_vol_1_train, self.dic_vol_10_validation, self.dic_vol_1_validation = self.set_vol_historical()

    def set_vol_historical(self):
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

            df_subset['rv_historical_10d'] = rv_10
            df_subset['rv_historical_1d'] = rv_1
            return df_subset

        df_train_sub = self.market_data.df_train[['Date','Price']].drop_duplicates()
        df_train_sub = df_train_sub[self.market_data.df_train['Date'] > " 2016-03-01"].copy() 

        df_validation_sub = self.market_data.df_validation[['Date','Price']].drop_duplicates().copy()

        # df_subset_train = compute_vols(df_train_sub)
        # df_subset_validation = compute_vols(df_validation_sub)

        dic_vol_10_train = dict(zip(df_train_sub['Date'], compute_vols(df_train_sub)['rv_historical_10d']))
        dic_vol_1_train = dict(zip(df_train_sub['Date'], compute_vols(df_train_sub)['rv_historical_1d']))

        dic_vol_10_validation = dict(zip(df_validation_sub['Date'],compute_vols(df_validation_sub)['rv_historical_10d']))
        dic_vol_1_validation = dict(zip(df_validation_sub['Date'],compute_vols(df_validation_sub)['rv_historical_1d']))        

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



class Regime_switching_HMM(BaseStrategy):

    def __init__(self, market_data):
        super().__init__(market_data)
    
    def df_filtered_price(self, straddle_row):
        '''Return the dataframe of all prices until the date of straddle_row, with
        log_return include
        '''
        df_filtered = self.market_data.df_price.copy()
        df_filtered['log_returns'] = np.log(df_filtered['Price'] / df_filtered['Price'].shift(1))
        rv_5d = []
        for i in range(len(df_filtered)):
            if i < 5:
                rv_5d.append(np.nan)
            else:
                vol = np.sqrt(np.mean(df_filtered['log_returns'].iloc[i-4:i+1]**2) * 252)
                rv_5d.append(vol)
        df_filtered['rv_5d'] = rv_5d

        df_filtered = df_filtered[(df_filtered['Date'] > " 2016-03-01") & (df_filtered['Date'] <= straddle_row['Date'])]

        return df_filtered


    def generate_alpha(self, straddle_row):
        if straddle_row['Date'] in self.market_data.df_train['Date'].tolist(): # if it's a straddle in df_train
            df_filtered = self.df_filtered_price(self.market_data.df_train.iloc[-1])   
        else : # if it's a straddle in df_validation
            df_filtered = self.df_filtered_price(straddle_row)  

        log_returns = df_filtered['log_returns']
        features = pd.concat([log_returns], axis=1).dropna()  

        mean = features.mean()
        std = features.std()
        features_scaled = (features - mean) / std

        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42) 
        model.fit(features_scaled)

        hidden_states = model.predict(features_scaled) # probale states
        proba_states = model.predict_proba(features_scaled) # probability of being in a given state

        # calculate the variance of log_return for each state
        state_variances = [np.var(log_returns[hidden_states == i]) for i in range(model.n_components)]
        sorted_states = np.argsort(state_variances)  
        low_vol_index = sorted_states[0] # Low regime
        mid_vol_index = sorted_states[1] # Middle regime
        high_vol_index = sorted_states[2] # High regime

        df_states = pd.DataFrame({
            "Date": df_filtered['Date'],
            "log_return": log_returns.values,
            "rv_5d": df_filtered['rv_5d'],
            "state": hidden_states,
            "proba_lowregime": proba_states[:,low_vol_index],
            "proba_midregime": proba_states[:,mid_vol_index],
            "proba_highregime": proba_states[:,high_vol_index]
        })

        dic_vol_date = dict(zip(df_states['Date'],df_states['rv_5d']))
        dic_prob_low = dict(zip(df_states['Date'],df_states['proba_lowregime']))
        dic_prob_high = dict(zip(df_states['Date'],df_states['proba_highregime']))


        return dic_vol_date[straddle_row['Date']], dic_prob_low[straddle_row['Date']], dic_prob_high[straddle_row['Date']] # historical vol 5d, proba low_reg, proba high regime




    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = (alpha[0] < straddle_row['IV']) and (alpha[1] > 0.86) # SHORT
        condition2 = (alpha[0] > straddle_row['IV']) and (alpha[2] > 0.7) # LONG
        return condition1 or condition2



    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = (alpha[0] < straddle_row['IV']) and (alpha[1] > 0.86) # SHORT
        condition2 = (alpha[0] > straddle_row['IV']) and (alpha[2] > 0.7) # LONG
        if condition1 : 
            return 'SHORT'
        elif condition2:
            return 'LONG'