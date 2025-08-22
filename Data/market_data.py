import numpy as np
from scipy.stats import norm
import pandas as pd


class Market_data:

    def __init__(self, df_train, df_validation, df_test, df_price, df_option):
        self.df_train = df_train.copy()
        self.df_validation = df_validation.copy()
        self.df_test = df_test.copy()
        self.df_price = df_price.copy()
        self.df_option = df_option.copy()

    
    def get_dataset(self, name):
        if name == "train":
            return self.df_train
        elif name == "validation":
            return self.df_validation
        elif name == "test":
            return self.df_test
        else:
            raise ValueError(f"Unknown dataset: {name}")


    def get_price_df(self):
        return self.df_price
    

    def get_row_price_date(self, date):
        ''' This function retrieves from the df_price DataFrame the row corresponding 
        to the date provided as a parameter.

        RETURNS
        -------
        type : pandas.core.series.Series
        '''
        rows = self.df_price[self.df_price["Date"] == date]
        if rows.empty:
            raise KeyError(f"Date {date} not found")
        return rows.iloc[0]


    def get_rows_price_date(self,date1,date2):
        '''This function retrieves from the DataFrame the spot prices for the 
        dates between date1 and date2.
        
        RETURNS
        -------
        type : pandas.core.frame.DataFrame
        '''
        filtered = self.df_price[(self.df_price['Date'] >= date1) & (self.df_price['Date'] <= date2)]
        return filtered

    def get_prices_date_list(self,date1,date2):
        '''This function returns the list of spot prices for dates between date1 and date2.

        RETURNS
        -------
        type : np.array
        '''
        filtered = self.get_rows_price_date(date1,date2)
        return filtered['Price'].to_numpy()


    def get_prices_list_lookahead(self, date, nb_period):
        ''' This function returns the list of spot prices for the nb_period preceding the date 
        provided as a parameter.

        RETURNS
        -------
        type : list
        '''
        list_date = self.df_price['Date'].tolist()
        id_date = list_date.index(date)
        filtered = self.df_price.iloc[id_date - nb_period : id_date + 1]
        return filtered['Price'].tolist()

    def get_realized_volatility(self,prices):
        '''This function calculates the annualized realized volatility for a 
        given list of stock prices provided as the parameter prices.

        RETURNS
        -------
        type : float
        '''
        log_returns = np.diff(np.log(prices))
        realized_vol = np.sqrt(np.mean(log_returns ** 2) * 252)
        return realized_vol

    
    def get_list_realized_volatility(self,date,nb_period):
        '''This function returns the list of realized volatilities, 
        each computed over nb_period days, for each day preceding the date 
        provided as a parameter.
        In other words : it returns the evolution of the realized volatility
        over the previous nb_period date preceding date

        RETURNS
        -------
        type : list of float
        '''
        vol = []
        list_date = self.df_price['Date'].tolist()
        id_date = list_date.index(date)
        list_date_lookahead = list_date[id_date - nb_period : id_date + 1] # list of nb_period date preceding id_date
        for d in list_date_lookahead:
            real_vol = self.get_realized_volatility(self.get_prices_list_lookahead(d,nb_period))
            vol.append(real_vol)
        return vol

    def get_list_IV(self,date, nb_period):
        '''This function returns the list of implied volatilities (IV) for 
        the nb_period dates preceding the given date. For each date, the IV 
        is computed as the average of the IVs of the options available on 
        that date.

        RETURNS
        -------
        type : list of float
        '''
        IV = []
        list_date = self.df_price['Date'].tolist()
        id_date = list_date.index(date)
        list_date_lookahead = list_date[id_date - nb_period : id_date + 1] # id_date is included, so the iv of the current straddle is included 

        df_train_filtered = self.df_train[self.df_train['Date'].isin(list_date_lookahead)]
        df_validation_filtered = self.df_validation[self.df_validation['Date'].isin(list_date_lookahead)]

        
        df_filtered_date = pd.concat([df_train_filtered, df_validation_filtered], ignore_index=True)

        mean_iv_by_date = df_filtered_date.groupby('Date')['IV'].mean()
        mean_iv_list = mean_iv_by_date.tolist()

        return mean_iv_list