import numpy as np
from scipy.stats import norm
import pandas as pd

from sklearn.linear_model import LinearRegression

from strategy.base_strat import BaseStrategy

class Regression_realvol(BaseStrategy):
    
    def __init__(self, market_data, nb_period):
        super().__init__(market_data)
        self.nb_period = nb_period
        self.quantile_high = np.quantile(self.compute_slope_train(), 0.75)
        self.quantile_low = np.quantile(self.compute_slope_train(), 0.25)

    def compute_slope_train(self):
        slope_train = []
        df_filtered = self.market_data.df_train[self.market_data.df_train['Date'] > " 2016-03-01"]
        for _, row in df_filtered.iterrows():
            Y = self.market_data.get_list_realized_volatility(row['Date'],self.nb_period)
            X = [i for i in range(1,len(Y)+1)]

            X = np.array(X).reshape(-1, 1)
            Y = np.array(Y)

            model = LinearRegression()
            model.fit(X, Y)

            slope_train.append(model.coef_[0])
        return slope_train


    def generate_alpha(self, straddle_row):
        Y = self.market_data.get_list_realized_volatility(straddle_row['Date'],self.nb_period)
        X = [i for i in range(1,len(Y)+1)]

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y)

        model = LinearRegression()
        model.fit(X, Y)

        '''
        print(f"Slope : {model.coef_[0]}")
        print(f"Intercept : {model.intercept_}")
        print(f"R² : {model.score(X, Y)}")

        plt.scatter(X, Y, color='blue', label='Données')
    
        # Tracé de la droite de régression
        Y_pred = model.predict(X)
        plt.plot(X, Y_pred, color='red', label='Droite de régression')
        
        plt.xlabel("Index")
        plt.ylabel("Volatilité réalisée")
        plt.title("Régression linéaire de la volatilité réalisée")
        plt.legend()
        plt.show()
        '''

        return model.coef_[0]

        

    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        return (alpha > self.quantile_high) or (alpha < self.quantile_low)

    
    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        if alpha > self.quantile_high: # If RV is over estimated
            return 'SHORT'
        elif alpha < self.quantile_low: # If RV is under estimated
            return 'LONG'


class Regression_IV(BaseStrategy):
    
    def __init__(self, market_data, nb_period):
        super().__init__(market_data)
        self.nb_period = nb_period
        self.quantile_high = np.quantile(self.compute_slope_train(), 0.75)
        self.quantile_low = np.quantile(self.compute_slope_train(), 0.25)


    def compute_slope_train(self):
        slope_train = []
        df_filtered = self.market_data.df_train[self.market_data.df_train['Date'] > " 2016-03-01"]
        for _, row in df_filtered.iterrows():
            Y = self.market_data.get_list_IV(row['Date'],self.nb_period)
            X = [i for i in range(1,len(Y)+1)]

            X = np.array(X).reshape(-1, 1)
            Y = np.array(Y)

            model = LinearRegression()
            model.fit(X, Y)

            slope_train.append(model.coef_[0])
        return slope_train


    def generate_alpha(self, straddle_row):
        Y = self.market_data.get_list_IV(straddle_row['Date'],self.nb_period)
        X = [i for i in range(1,len(Y)+1)]

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y)

        model = LinearRegression()
        model.fit(X, Y)

        '''
        print(f"Slope : {model.coef_[0]}")
        print(f"Intercept : {model.intercept_}")
        print(f"R² : {model.score(X, Y)}")

        
        plt.scatter(X, Y, color='blue', label='Données')
    
        
        # Tracé de la droite de régression
        Y_pred = model.predict(X)
        plt.plot(X, Y_pred, color='red', label='Droite de régression')
        
        plt.xlabel("Index")
        plt.ylabel("Volatilité réalisée")
        plt.title("Régression linéaire de la volatilité réalisée")
        plt.legend()
        plt.show()
        '''

        return model.coef_[0] # return the slop


    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        return (alpha > self.quantile_high) or (alpha < self.quantile_low)

    
    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        if alpha > self.quantile_high: # If IV is over estimated
            return 'LONG'
        elif alpha < self.quantile_low: # If IV is under estimated
            return 'SHORT'


class Regression_IVvsRV(BaseStrategy):
    
    def __init__(self, strat_iv, strat_rv, market_data, nb_period):
        super().__init__(market_data)
        self.strat_iv = strat_iv
        self.strat_rv = strat_rv
        self.nb_period = nb_period

    def generate_alpha(self, straddle_row):
        alpha_iv = self.strat_iv.generate_alpha(straddle_row)
        alpha_rv = self.strat_rv.generate_alpha(straddle_row)

        return [alpha_iv,alpha_rv] # IV:alpha[0] and RV:alpha[1]


    def should_trade(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        condition1 = (alpha[0] > self.strat_iv.quantile_high) and (alpha[1] < self.strat_rv.quantile_low)
        condition2 = (alpha[0] < self.strat_iv.quantile_low) and (alpha[1] > self.strat_rv.quantile_high)
        return condition1 or condition2
        # If (IV is overestimated and RV is underestimated) or if (IV is underestimated and RV is overestimated) in terms of slope variation


    def get_signal(self, straddle_row):
        alpha = self.generate_alpha(straddle_row)
        if (alpha[0] > self.strat_iv.quantile_high) and (alpha[1] < self.strat_rv.quantile_low): 
            return 'LONG'
        elif (alpha[0] < self.strat_iv.quantile_low) and (alpha[1] > self.strat_rv.quantile_high): 
            return 'SHORT' 
