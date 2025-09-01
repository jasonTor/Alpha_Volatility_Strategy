import numpy as np
from scipy.stats import norm
import pandas as pd

from sklearn.linear_model import LinearRegression

from strategy.base_strat import BaseStrategy


class Statistical_strat1(BaseStrategy):

    def __init__(self, market_data):
        super().__init__(market_data)
        self.dic_vol_3_historical, self.dic_IV_3_historical = self.set_vol_historical()

    def set_vol_historical(self):
        d1 = {}
        d2 = {} 
        df_filtered = pd.concat([self.market_data.df_train, self.market_data.df_validation, self.market_data.df_test], ignore_index=True)
        df_filtered = df_filtered[df_filtered['Date'] > " 2016-03-01"]
        list_date = df_filtered['Date'].unique().tolist() 
        for d in list_date:
            id_date = self.market_data.df_price['Date'].tolist().index(d)
            d1[d] = self.market_data.df_price['Price'].iloc[id_date - 3:id_date+1]
            id_date = df_filtered.groupby('Date')['IV'].mean().index.tolist().index(d)
            d2[d] = df_filtered.groupby('Date')['IV'].mean()[id_date - 2:id_date+1]
        
        for d in d1.keys():
            d1[d] = np.abs(np.log(d1[d]/d1[d].shift(1)).dropna())

        return d1,d2

    def is_sorted(self,lst):
        return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

    def is_sorted_desc(self,lst):
        return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))
    

    def generate_alpha(self, straddle_row):
        return self.dic_vol_3_historical[straddle_row['Date']].tolist(), self.dic_IV_3_historical[straddle_row['Date']].tolist()

    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = self.is_sorted(alpha[0]) and self.is_sorted(alpha[1]) # LONG
        condition2 = self.is_sorted_desc(alpha[0]) and self.is_sorted_desc(alpha[1]) # SHORT
        return condition1 or condition2


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = self.is_sorted(alpha[0]) and self.is_sorted(alpha[1]) # LONG
        condition2 = self.is_sorted_desc(alpha[0]) and self.is_sorted_desc(alpha[1]) # SHORT
        if condition1 : 
            return 'LONG'
        elif condition2: 
            return 'SHORT'



class Statistical_strat2(BaseStrategy):
    ''' TERM STRUCTURE STRAT
    '''
    def __init__(self, market_data):
        super().__init__(market_data)

    def is_sorted(self,lst):
        return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

    def is_sorted_desc(self,lst):
        return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))
    
    def generate_alpha(self, straddle_row):
        df_filtered = pd.concat([self.market_data.df_train, self.market_data.df_validation, self.market_data.df_test], ignore_index=True)
        df_date = df_filtered[df_filtered['Date'] == straddle_row['Date']].sort_values(by='day_to_maturity', ascending=True)
        list_iv = df_date['IV'].tolist()
        return list_iv

    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = self.is_sorted(alpha)  # SHORT
        return condition1


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = self.is_sorted(alpha) # SHORT
        if condition1 : 
            return 'SHORT'


class Statistical_strat3(BaseStrategy):
    ''' based-spread STRAT
    '''

    def __init__(self, market_data):
        super().__init__(market_data)
        self.quantile_80_positif_train, self.quantile_20_negatif_train, self.quantile_80_positif_train_validation, self.quantile_20_negatif_train_validation = self.get_quantile()

    def get_quantile(self):
        df = pd.concat([self.market_data.df_train, self.market_data.df_validation], ignore_index=True)
        
        df_train_filtered = self.market_data.df_train[self.market_data.df_train['Date'] > " 2016-03-01"]
        spread_list_train = []
        for _,row in df_train_filtered.iterrows():
            spread_list_train.append(self.market_data.get_realized_volatility(self.market_data.get_prices_list_historical(row['Date'],int(row['day_to_maturity']))) - row['IV'])
        
        df_train_validation = df[df['Date'] > " 2016-03-01"]
        spread_list_train_validation = []
        for _,row in df_train_validation.iterrows():
            spread_list_train_validation.append(self.market_data.get_realized_volatility(self.market_data.get_prices_list_historical(row['Date'],int(row['day_to_maturity']))) - row['IV'])

        list_spread_train_positif = [i for i in spread_list_train if i>0]
        list_spread_train_negatif = [i for i in spread_list_train if i<0]

        list_spread_train_validation_positif = [i for i in spread_list_train_validation if i>0]
        list_spread_train_validation_negatif = [i for i in spread_list_train_validation if i<0]

        quantile_80_positif_train = np.percentile(list_spread_train_positif,80)
        quantile_20_negatif_train = np.percentile(list_spread_train_negatif,20)

        quantile_80_positif_train_validation = np.percentile(list_spread_train_validation_positif,80)
        quantile_20_negatif_train_validation = np.percentile(list_spread_train_validation_negatif,20)

        return quantile_80_positif_train, quantile_20_negatif_train, quantile_80_positif_train_validation, quantile_20_negatif_train_validation


    
    def generate_alpha(self, straddle_row):
        spread = self.market_data.get_realized_volatility(self.market_data.get_prices_list_historical(straddle_row['Date'],int(straddle_row['day_to_maturity']))) - straddle_row['IV']
        return spread

    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        if (straddle_row['Date'] in self.market_data.df_train['Date'].unique()) or (straddle_row['Date'] in self.market_data.df_validation['Date'].unique()):
            condition1 = alpha > self.quantile_80_positif_train # SHORT
            condition2 = alpha < self.quantile_20_negatif_train # LONG
        elif straddle_row['Date'] in self.market_data.df_test['Date'].unique():
            condition1 = alpha > self.quantile_80_positif_train_validation # SHORT
            condition2 = alpha < self.quantile_20_negatif_train_validation # LONG            
        return condition1 or condition2


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        if (straddle_row['Date'] in self.market_data.df_train['Date'].unique()) or (straddle_row['Date'] in self.market_data.df_validation['Date'].unique()):
            condition1 = alpha > self.quantile_80_positif_train # SHORT
            condition2 = alpha < self.quantile_20_negatif_train # LONG
        elif straddle_row['Date'] in self.market_data.df_test['Date'].unique():
            condition1 = alpha > self.quantile_80_positif_train_validation # SHORT
            condition2 = alpha < self.quantile_20_negatif_train_validation # LONG  
        if condition1 : 
            return 'SHORT'
        elif condition2 :
            return 'LONG'


class Statistical_strat4(BaseStrategy):
    ''' RISK REVERSAL 25
    '''

    def __init__(self, market_data):
        super().__init__(market_data)
        self.dict_rr25, self.dict_iv = self.get_rr25_iv()

    def preprocessing_option(self):
        self.market_data.df_option[' [C_IV]'] = pd.to_numeric(self.market_data.df_option[' [C_IV]'], errors='coerce')
        self.market_data.df_option[' [P_IV]'] = pd.to_numeric(self.market_data.df_option[' [P_IV]'], errors='coerce')

        self.market_data.df_option[' [C_DELTA]'] = pd.to_numeric(self.market_data.df_option[' [C_DELTA]'], errors='coerce')
        self.market_data.df_option[' [P_DELTA]'] = pd.to_numeric(self.market_data.df_option[' [P_DELTA]'], errors='coerce')

        self.market_data.df_option['IV'] = ((self.market_data.df_option[' [C_IV]'] + self.market_data.df_option[' [P_IV]']) / 2).round(5)
        self.market_data.df_option = self.market_data.df_option.dropna(subset=['IV'])

        self.market_data.df_option = self.market_data.df_option[['Date','day_to_maturity',' [C_IV]',' [C_DELTA]',' [P_IV]',' [P_DELTA]','IV','moneyness']].copy()

        self.market_data.df_option = self.market_data.df_option[self.market_data.df_option['day_to_maturity'] <= 21]
        self.market_data.df_option = self.market_data.df_option[self.market_data.df_option['day_to_maturity'] >= 12]

        self.market_data.df_option['tau'] = self.market_data.df_option['day_to_maturity']/252
        

    def get_rr25_iv(self):
        self.preprocessing_option()
        df_filtered_option = self.market_data.df_option.copy()

        list_date = df_filtered_option['Date'].unique().tolist() # ALL THE DIFFERENT DATE FROM df_filtered_option (NOT df_train + df_validation + df_test that had been filtered only with ATM)
        historical_rr25 = []
        for d in list_date:
            df_filtered = df_filtered_option[df_filtered_option['Date'] == d]
            idx_c = (df_filtered[' [C_DELTA]'] - 0.25).abs().idxmin() # Retrieve the index of delta 25 call
            idx_p = (df_filtered[' [P_DELTA]'] + 0.25).abs().idxmin() # Retrieve the index of delta 25 put
            historical_rr25.append(df_filtered.loc[idx_p][' [P_IV]'] - df_filtered.loc[idx_c][' [C_IV]'])
        historical_rr25 = pd.Series(historical_rr25,list_date)
        
        df_all = pd.concat([self.market_data.df_train,self.market_data.df_validation,self.market_data.df_test], ignore_index = True)

        iv_train = [df_all[df_all['Date'] == d].groupby('Date')['IV'].mean()[0] for d in df_all['Date'].unique().tolist()]
        iv_train = pd.Series(iv_train, df_all['Date'].unique())

        dict_rr25 = {}
        df_filtered = df_all[df_all['Date'] > ' 2016-03-01']
        for d in df_filtered['Date'].unique():
            id_date = historical_rr25.index.get_loc(d)
            dict_rr25[d] = historical_rr25[id_date-9:id_date+1]
        
        dict_iv = {}
        for d in df_all['Date'].unique():
            id_date = iv_train.index.get_loc(d)
            dict_iv[d] = iv_train[id_date-9:id_date+1]
        
        return dict_rr25, dict_iv
        
    def is_sorted(self,lst):
        return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

    def is_sorted_desc(self,lst):
        return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))
    
    def generate_alpha(self, straddle_row):
        rr25 = self.dict_rr25[straddle_row['Date']]
        return rr25[-1]

    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row) # rr25
        #print((self.is_sorted_desc(self.dict_rr25[straddle_row['Date']][-3:])))
        #print((alpha - self.dict_rr25[straddle_row['Date']][-2] > -0.005))
        #print((self.dict_rr25[straddle_row['Date']][-1] - np.mean(self.dict_rr25[straddle_row['Date']][-4:-1]) > 0.015))
        #print((self.is_sorted_desc(self.dict_iv[straddle_row['Date']][-3:])))
        condition1 = (self.is_sorted_desc(self.dict_rr25[straddle_row['Date']][-3:])) and (alpha - self.dict_rr25[straddle_row['Date']][-2] > -0.005) # SHORT   
        condition2 = (self.dict_rr25[straddle_row['Date']][-1] - np.mean(self.dict_rr25[straddle_row['Date']][-4:-1]) > 0.015) and (self.is_sorted_desc(self.dict_iv[straddle_row['Date']][-3:])) # LONG
        return condition1 or condition2


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row) # rr25
        condition1 = (self.is_sorted_desc(self.dict_rr25[straddle_row['Date']][-3:])) and (alpha - self.dict_rr25[straddle_row['Date']][-2] > -0.005) # SHORT   
        condition2 = (self.dict_rr25[straddle_row['Date']][-1] - np.mean(self.dict_rr25[straddle_row['Date']][-4:-1]) > 0.015) and (self.is_sorted_desc(self.dict_iv[straddle_row['Date']][-3:])) # LONG        
        if condition1 : 
            return 'SHORT'
        elif condition2 :
            return 'LONG'