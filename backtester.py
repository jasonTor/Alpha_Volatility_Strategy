import numpy as np
from scipy.stats import norm
import pandas as pd

class Backtester:

    def __init__(self, market_data, strategy):
        '''
        PARAMETERS
        ----------
        market_data
            Class Market_data
        strategy
            Class Strategy
        '''
        self.market_data = market_data
        self.strategy = strategy
    
    # ------------------------- PRICERS AND GREEKS --------------------------------------
    def black_scholes_call_price(self, S, K, T, r, sigma):
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2)* T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def black_scholes_put_price(self, S, K, T, r, sigma):
        if T <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    def black_scholes_call_delta(self, S, K, T, r, sigma):
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)

    def black_scholes_put_delta(self, S, K, T, r, sigma):
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1 
    # --------------------------------------------------------------------------------

    
    def run_gamma_scalping(self, straddle_row):
        '''

        PARAMETERS
        ----------
        straddle_row : 
            pandas.core.series.Series
        '''
        res = {}

        PNL_evolution = [] # evolution of the pnl at each time step of the delta hedging
        hedge_gain = []

        Straddle_t0 = straddle_row['Straddle']

        df_prices_lifetime = self.market_data.get_rows_price_date(straddle_row['Date'],straddle_row[' [EXPIRE_DATE]']) # We retrieve the df of the spot prices associated to the lifetime of the straddle

        # avoid splited stock problem
        if straddle_row['Date'] <= " 2020-07-31":
            date_to_price = dict(zip(df_prices_lifetime['Date'], df_prices_lifetime['Price_unsplited'])) # We create the dictionnary of date and associated unsplited price
        elif straddle_row['Date'] >= " 2020-09-01":
            date_to_price = dict(zip(df_prices_lifetime['Date'], df_prices_lifetime['Price'])) # We create the dictionnary of date and associated price
        else:
            return {'PNL':0, 'price_strad':0, 'gross_gain':0} 


        list_date = list(date_to_price.keys())

        r = 0.03
        iv_call = straddle_row[' [C_IV]']
        iv_put = straddle_row[' [P_IV]']

        # Hedging part
        for i in range(1, len(date_to_price)):
            S_tbefore = date_to_price[list_date[i-1]]
            S_t = date_to_price[list_date[i]]

            time_to_maturity_before = (len(list_date) - (i - 1)) / 252
            time_to_maturity = (len(list_date) - i) / 252

            C_tbefore = self.black_scholes_call_price(S_tbefore, straddle_row[' [STRIKE]'], time_to_maturity_before, r, iv_call)
            C_t = self.black_scholes_call_price(S_t, straddle_row[' [STRIKE]'], time_to_maturity, r, iv_call)
            P_tbefore = self.black_scholes_put_price(S_tbefore, straddle_row[' [STRIKE]'], time_to_maturity_before, r, iv_put)
            P_t = self.black_scholes_put_price(S_t, straddle_row[' [STRIKE]'], time_to_maturity, r, iv_put)

            Straddle_before = C_tbefore + P_tbefore
            Straddle_t = C_t + P_t

            delta_call_t = self.black_scholes_call_delta(S_tbefore, straddle_row[' [STRIKE]'], time_to_maturity_before, r, iv_call)
            delta_put_t = self.black_scholes_put_delta(S_tbefore, straddle_row[' [STRIKE]'], time_to_maturity_before, r, iv_put)

            hedge_value = (-delta_call_t * (S_t - S_tbefore) +
                        (delta_call_t * S_tbefore - C_tbefore) * (np.exp(r * (1/252)) - 1)
                        - delta_put_t * (S_t - S_tbefore) +
                        (delta_put_t * S_tbefore - P_tbefore) * (np.exp(r * (1/252)) - 1))

            hedge_gain.append(hedge_value)

            pnl_total_t = (Straddle_t - Straddle_t0) + np.sum(hedge_gain[:i])
            PNL_evolution.append(pnl_total_t)
        
        res['PNL'] = PNL_evolution[-1]
        res['price_strad'] = Straddle_t0
        res['gross_gain'] = PNL_evolution[-1] + Straddle_t0

        return res





    # ------------------------------------------- IV vs RV Backtesting ---------------------------------------------------------------
    def run_backtest_IVvsRV(self):
        '''This function backtests the predictive power of the strategy by computing the proportion
        of correct predictions between implied volatility (IV) and realized volatility (RV) on 
        df_train and df_validation
        '''
        df_filtered = self.market_data.df_train[self.market_data.df_train['Date'] > " 2016-03-01"]

        mask_trade = df_filtered.apply(self.strategy.should_trade, axis=1)
        mask_trade2 = self.market_data.df_validation.apply(self.strategy.should_trade, axis=1)

        df_trades = df_filtered[mask_trade].copy()
        df_trades2 = self.market_data.df_validation[mask_trade2].copy()

        df_trades["signal"] = df_trades.apply(self.strategy.get_signal, axis=1)
        df_trades2["signal"] = df_trades2.apply(self.strategy.get_signal, axis=1)

        df_trades["success"] = np.where(
            ((df_trades["signal"] == "LONG")  & (df_trades["vol_real"] > df_trades["IV"])) |
            ((df_trades["signal"] == "SHORT") & (df_trades["vol_real"] < df_trades["IV"])),
            1, 0
        )

        df_trades2["success"] = np.where(
            ((df_trades2["signal"] == "LONG")  & (df_trades2["vol_real"] > df_trades2["IV"])) |
            ((df_trades2["signal"] == "SHORT") & (df_trades2["vol_real"] < df_trades2["IV"])),
            1, 0
        )

        long_trades_train = df_trades[df_trades["signal"] == "LONG"]
        short_trades_train = df_trades[df_trades["signal"] == "SHORT"]
        
        success_rate_long_train = long_trades_train["success"].mean() if not long_trades_train.empty else 0.0
        success_rate_short_train = short_trades_train["success"].mean() if not short_trades_train.empty else 0.0
        success_rate_total_train = df_trades["success"].mean() if not df_trades.empty else 0.0
        
        long_trades_validation = df_trades2[df_trades2["signal"] == "LONG"]
        short_trades_validation = df_trades2[df_trades2["signal"] == "SHORT"]
        
        success_rate_long_validation = long_trades_validation["success"].mean() if not long_trades_validation.empty else 0.0
        success_rate_short_validation = short_trades_validation["success"].mean() if not short_trades_validation.empty else 0.0
        success_rate_total_validation = df_trades2["success"].mean() if not df_trades2.empty else 0.0

        # Detail
        print("=" * 80)
        print("IV vs RV BACKTEST RESULTS - TRAIN SET")
        print("=" * 80)
        print(f"Total trades: {len(df_trades)}")
        print(f"LONG trades: {len(long_trades_train)} - (RV > IV) Success rate: {success_rate_long_train:.2%} versus 38% in df_train")
        print(f"SHORT trades: {len(short_trades_train)} - (RV < IV) Success rate: {success_rate_short_train:.2%} versus 62% in df_train")
        print(f"OVERALL success rate: {success_rate_total_train:.2%}")
        print()
        
        print("=" * 80)
        print("IV vs RV BACKTEST RESULTS - VALIDATION SET")
        print("=" * 80)
        print(f"Total trades: {len(df_trades2)}")
        print(f"LONG trades: {len(long_trades_validation)} - (RV > IV) Success rate: {success_rate_long_validation:.2%} versus 25% in df_validation")
        print(f"SHORT trades: {len(short_trades_validation)} - (RV < IV) Success rate: {success_rate_short_validation:.2%} versus 75% in df_validation")
        print(f"OVERALL success rate: {success_rate_total_validation:.2%}")
        print("=" * 80)

        return None
    # --------------------------------------------------------------------------------------------------------------------------------      
        



    # ------------------------------------------- PNL generation backtesting ---------------------------------------------------------
    def run_row(self, straddle_row):
        trade_signal = "SKIP"
        res = {}
        if self.strategy.should_trade(straddle_row): # return a bool
            signal = self.strategy.get_signal(straddle_row)
            scalp = self.run_gamma_scalping(straddle_row)
            if signal == 'LONG' :
                res['PNL'] = scalp['PNL']
                res['trade_signal'] = 'LONG'
            elif signal == 'SHORT' : # If signal == 'SHORT'
                res['PNL'] = -scalp['PNL']
                res['trade_signal'] = 'SHORT'
            res['price_strad'] = scalp['price_strad']
        else :
            res = {'PNL' : 0,  'trade_signal' : trade_signal, 'price_strad' : 0}
        #print(res)
        return res
 

    def run_backtest_train(self):
        PNL = 0
        Investment_capital = 0
        df_filtered = self.market_data.df_train[self.market_data.df_train['Date'] > " 2016-03-01"]
        for _, row in df_filtered.iterrows():
            res = self.run_row(row)
            PNL += res['PNL']
            Investment_capital += res['price_strad']
            #print(f"Date:{row['Date']}, PNL:{PNL}, Decision to trade:{res['trade_signal']}, iv:{row['IV']}, rv:{row['vol_real']}")
        ROI = PNL / Investment_capital if Investment_capital != 0 else 0
        
        print(f"Result on df_train : PNL:{PNL}, ROI:{ROI*100} %")
        return None
    
    def run_backtest_validation(self):
        PNL = 0
        Investment_capital = 0
        for _,row in self.market_data.df_validation.iterrows():
            res = self.run_row(row)
            PNL += res['PNL']
            Investment_capital += res['price_strad']
            #print(f"Date:{row['Date']}, PNL:{PNL}, Decision to trade:{res['trade_signal']}, iv:{row['IV']}, rv:{row['vol_real']}")
        ROI = PNL / Investment_capital if Investment_capital != 0 else 0
        
        print(f"Result on df_validation : PNL:{PNL}, ROI:{ROI*100} %")
        return None
    # -----------------------------------------------------------------------------------------------------------------------------
    
