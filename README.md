# **Volatility Trading Project** - (***ongoing project***)
---

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
 
## **4. Theoretical Framework and Motivation**

### 1. **Choice of pricer Framework**

   I chose to evaluate option prices and calculate the Greeks using the basic assumptions of the Black-Scholes model.
   Recall that under Black-Scholes Assumption, in risk neutral measure with **r** the risk free rate, the market is composed of :
   
   $$dS_t = S_t \mu dt + S_t \sigma dW_t$$ (***risky asset***)
   
   $$dS^0_t = S^0_t r dt$$ (***risk-free asset***)
   
   $$\Longrightarrow S_t = S_0 e^{(\mu - \frac{\sigma^2}{2})t + \sigma \sqrt{t} \mathcal{N}(0,1)}$$
   
   $$\Longrightarrow S^0_t=e^{rt}$$

Call and Put prices:

$$
C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
$$

$$
P = K e^{-rT} \Phi(-d_2) - S_0 \Phi(-d_1)
$$

with

$$
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2) T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

Call and Put deltas:

$$
\Delta_{\text{call}} = \Phi(d_1), \quad
\Delta_{\text{put}} = \Phi(d_1) - 1
$$

### 2. **Hedging Process : In a perfect world**

**Market Assumptions**:

- No transaction costs  
- No market impact  
- Perfect liquidity  
- No arbitrage opportunities  


When we buy a **call** (strike K, maturity T) at time $t_1$, the bank uses cash from its internal capital to pay the call price:

$$
C_{t_1} = \Delta_{t_1} S_{t_1} - K e^{-r(T-t_1)} N(d_2)
$$

Then it hedges by shorting $\Delta_{t_1} S_{t_1}$ dollars of the stock, receiving $\Delta_{t_1} S_{t_1}$ dollars. From this amount, the bank invests the part 

$$
\Delta_{t_1} S_{t_1} - C_{t_1}
$$

at the risk-free rate, leaving $C_{t_1}$ in cash, which is returned to its capital to cover the initial cash used to buy the call.

Thus, the total portfolio value is

$$
P_{t_1} = C_{t_1} + V_{t_1} = 0\ at\ time\ t_1
$$

where $V_{t_1}$ is the hedging portfolio, which can be written as

$$
V_t = \zeta_t S_t + \zeta^0_t S^0_t
$$

with

$$
\zeta_{t_1} = -\Delta_{t_1} \quad \text{and} \quad \zeta^0_{t_1} = \frac{\Delta_{t_1} S_{t_1} - C_{t_1}}{S^0_{t_1}}
$$

$\zeta_t$ and $\zeta^0_t$ represent the amounts of risky and risk-free assets held in the replication (hedging) portfolio, respectively.

We impose the **self-financing condition** on the portfolio, which ensures that no additional cash is needed to maintain the position over time. The positions $\zeta_t$ and $\zeta^0_t$ in the portfolio are balanced by selling one asset to buy the other, and vice versa.

$$
dV_t = \zeta_t dS_t + \zeta^0_t dS^0_t \ \ \  (self-financing\ condition)
$$

In the theoretical framework of perfect hedging, we must maintain 

$$
P_t = C_t + V_t
$$ 

for all $t \in [t_1, T]$.  

This implies that, continuously for all $t \in [t_1, T]$, we hold

$$
\zeta_t = -\Delta_t \text{ shares of the stock}.
$$

**Property**: This perfect continuous hedging strategy produces no loss and no profit: PnL = 0.

**PROOF**:

consider the infinitesimal variation of the portfolio:

$$
dP_t = dC_t + dV_t
$$

Using Itô's formula, we have

$$
dP_t = \frac{\partial C}{\partial t} dt + \frac{\partial C}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 C}{\partial S^2} d\langle S,S \rangle_t -\Delta_t dS_t + \frac{\Delta_t S_t - C_t}{S^0_t} dS^0_t
$$


Replacing $d\langle S,S \rangle_t$ by $\sigma^2 S_t^2 dt$ and $dS^0_t$ by $S^0_t r dt$, we get


$$
dP_t = \Theta_t dt + \Delta_t dS_t + \frac{1}{2} \Gamma_t \sigma^2 S_t^2 dt - \Delta_t dS_t + (\Delta_t S_t - C_t) r dt
$$

The terms cancel $\Delta_t dS_t$ out.

$$
dP_t = \Theta_t dt + \frac{1}{2} \Gamma_t \sigma^2 S_t^2 dt  + (\Delta_t S_t - C_t) r dt
$$

Recall the Black–Scholes PDE in Greeks form:

$$
\Theta_t + r S_t \Delta_t + \tfrac12 \sigma^2 S_t^2 \Gamma_t = r C_t
$$

This implies

$$
r \big(C_t - \Delta_t S_t\big) = \Theta_t + \tfrac12 \sigma^2 S_t^2 \Gamma_t
$$

Plugging this identity into $dP_t$


gives

$$
dP_t = 0
$$

**PNL FORMULA :**

Finally, the PnL of the hedged portfolio is given by

$$
\text{PnL} = \int_{t_1}^T dP_t = 0
$$

**Conclusion** : It shows that, under perfect continuous hedging and the Black–Scholes assumptions, **the portfolio generates no profit and no loss**.

### 3. **Hedging Process : In reality**

**Market Assumptions still**:

- No transaction costs  
- No market impact  
- Perfect liquidity  
- No arbitrage opportunities



Since continuous hedging is impossible in practice (for obvious reasons), hedging is done on a discrete grid of times.  
This introduces **hedging errors** between two rebalancing times $t_1$ and $t_2$.

Between $t_1$ and $t_2$, the PnL is

$$
\text{PnL} = P_{t_2} - P_{t_1} = \int_{t_1}^{t_2} dC_t + dV_t
$$

which can be expanded as

$$
\text{PnL} = \int_{t_1}^{t_2} \Theta_t dt + \Delta_t dS_t + \tfrac12 \Gamma_t \sigma^2 S_t^2 dt - \Delta_{t_1} dS_t + (\Delta_{t_1} S_{t_1} - C_{t_1}) r dt
$$

From the Black–Scholes PDE, we know that

$$
\Theta_t = -\tfrac12 \Gamma_t \sigma^2_{iv} S_t^2 - (\Delta_t S_t - C_t) r
$$

and also

$$
(dS_t)^2 = \sigma^2_{rv} S_t^2 dt
$$

so the PnL becomes

$$
\text{PnL} = \int_{t_1}^{t_2} (\Delta_t - \Delta_{t_1}) dS_t + \int_{t_1}^{t_2} \tfrac12 \Gamma_t S_t^2 \big(\sigma^2_{rv} - \sigma^2_{iv}\big) dt + \int_{t_1}^{t_2} \big((\Delta_{t_1} S_{t_1} - C_{t_1}) - (\Delta_t S_t - C_t)\big) r dt
$$

Heuristically : the first integral term, representing the PnL from the change in delta, is locally zero but positive on average, regardless of the direction of the stock trajectory between the rebalancing times. The third integral term, associated with the variations in the risk-free investment, is also locally zero and negative, partially cancels out the PnL from the first part.

Finally : 

$$
\text{PnL} = \int_{t_1}^{t_2} \tfrac12 \Gamma_t S_t^2 \big(\sigma^2_{rv} - \sigma^2_{iv}\big) dt 
$$

**Conclusion**, the PnL depends on Gamma and on the difference between realized volatility (RV) and implied volatility (IV). It is positive when RV > IV and negative when IV < RV (cf. ***gamma_scalp_simulation.ipynb*** file ). To maximize this gain, one needs a sufficiently large Gamma, which is highest for **at-the-money options**, and a large quadratic variation of the spot price between the rebalancing times.

In this project, we will focus on trading straddles (at-the-money call + put) to maximize Gamma to optimally capture the differences between realized volatility (RV) and implied volatility (IV)

---

### **REMARK** : Volatility risk premium and structural option bias
It is important to highlight that implied volatility (IV) is, on average, higher than realized volatility (RV). This gap reflects a volatility risk premium, indeed option sellers are compensated for bearing exposure to extreme risks (rare but violent market moves), while option buyers benefit from the convexity of their positions. As a result, most options are structurally “expensive” relative to the volatility that is eventually realized.

This explains why short volatility strategies are statistically favored and way easier to catch, while long volatility strategies face a structural disadvantage. In my df_train dataset, about **63%** of straddles are overpriced versus **37%** underpriced. The objective is therefore twofold:

- Build a long signal that performs better than 63% success rate.

- Build a short signal that performs better than 37% success rate.







   
















