# **Volatility Trading Project** - (***ongoing project***)

### **1. Project Context and Goals**
This project was carried out independently in  autonomous. My initial goal was to design volatility trading strategies using statistical and machine learning tools. The main idea was to predict, for each market option, whether its implied volatility was correctly priced. In this framework, I aimed to generate alpha on volatility by comparing the realized volatility of the underlying with the implied volatility of the quoted option, then taking a position based on this comparison. The quadratic variations of the spot were then exploited through **Gamma**, maintaining delta-hedging until maturity to generate a PnL. It is important to note that the strategies developed here are solely my own and reflect a personal and creative approach. They are not based on any principle guaranteeing immediate profit and should not be considered as arbitrage or guaranteed trading.

### **2. Data and Simulation Setup**

The dataset used in this project comes from two main sources.
First, option data on **Apple** stock retrieved from Kaggle, covering the period between 2016-04-01 and 2023-03-31. Each observation includes, among other features, the quotation date, the last price of calls and puts, and their respective implied volatilities.
Second, the underlying daily Apple stock prices were collected using the Yahoo Finance API, covering the same time span until the last option maturity. These two datasets are combined to simulate and evaluate volatility trading strategies.

The option dataset was split chronologically into three disjoint subsets:

- Training set (70%): used to design and train models or statistical strategies.
- Validation set (15%): used for backtesting and simulating strategies.
- Test set (15%): kept strictly out-of-sample, treated as an unseen “future” dataset to evaluate final strategies.

This setup ensures a clear separation between model development, simulation, and unbiased performance assessment.

### **3. Repository structure**

This repository contains the following files:
1. **Files related to simulation prototypes, preprocessing and experiments** :
   - ***gamma_scalp_simulation.ipynb***: Notebook illustrating the theory of gamma scalping (RV vs IV) through simulations. For a long/short position on a straddle priced with a given IV under Black-Scholes assumptions, this notebook shows how the PnL evolves based on simulated spot trajectories (geometric Brownian motion) and realized volatility. The simulation results perfectly match the theory, as seen in the last three PnL histograms from Monte-Carlo simulations.
   - ***data_preprocessing.ipynb***: Notebook for studying and cleaning the data to make it reproducible. The code may look a bit messy as it is just to clean data.
   - ***exploratory_analysis.ipynb***: This notebook aims to perform exploratory analyses. It serves as a space for prototyping and experimentation. The code here is not meant for production use but is intended to inspire and guide me in designing strategies.
2. **Simulation architecture coded from scratch**:
   - ***Data/market_data.py***
   - ***strategy/*** : Folder containing all files related to strategy implementation.
   - ***backtester.py*** : Prototype for backtesting the implemented strategies.
   - ***main.py*** : Main file where backtests are run.
 
