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

 
