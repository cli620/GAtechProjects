# Develop and describe at least 3 and at most 5 technical indicators. You may find our lecture on time series processing to be helpful. For each indicator you should create a single, compelling chart that illustrates the indicator.
#
# As an example, you might create a chart that shows the price history of the stock, along with "helper data" (such as upper and lower bollinger bands) and the value of the indicator itself. Another example: If you were using price/SMA as an indicator you would want to create a chart with 3 lines: Price, SMA, Price/SMA. In order to facilitate visualization of the indicator you might normalize the data to 1.0 at the start of the date range (i.e. divide price[t] by price[0]).
#
# Your report description of each indicator should enable someone to reproduce it just by reading the description. We want a written detailed description here, not code, however, it is OK to augment your written description with a pseudocode figure. Do NOT copy/paste code parts here as a description.
#
# At least one of the indicators you use should be completely different from the ones presented in our lectures. (i.e. something other than SMA, Bollinger Bands, RSI).
#
# N.B. Be careful of which indicators you end up with. We will require you to reuse the same ones on a future assignment.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
import sys

def author():
    return 'cli620'

def bb_value(price, N=5, genplot=False):
    # rollingSTD = pd.rolling_std(price, N)
    rollingSTD = price.rolling(window=N, center=False).std()
    # rollingAVG = pd.rolling_mean(price, N)
    rollingAVG = price.rolling(window=N, center=False).mean()
    bb_upper = rollingAVG + 2.0*rollingSTD
    bb_lower = rollingAVG - 2.0*rollingSTD
    bb = (price - rollingAVG) / (2 * rollingSTD)
    bb = bb.rename('bb')
    if genplot:
        # plot
        rollingAVG = rollingAVG.rename('rollingAVG')
        rollingSTD = rollingSTD.rename('rollingSTD')
        bb_upper = bb_upper.rename('bb_upper')
        bb_lower = bb_lower.rename('bb_lower')
        price = price.rename('JPM prices')

        all_bb = pd.concat([rollingAVG, rollingSTD, bb_upper, bb_lower,price], axis=1)
        plot_data(all_bb, title='Bollinger Bands', xlabel='times', ylabel='prices')
        # plt.savefig('bollinger_bands.png')

        plot_data(bb, title='Normalized BB', xlabel='times', ylabel='BBP')
        # plt.savefig('normalized_bollinger_bands.png')
    return bb, bb_upper, bb_lower, rollingAVG

def get_volatility(prices, N = 5, daily_rf = 0.0, sv=100000,genplot=False):
    start_val = 100000
    # n = 252
    # daily_rf = 0.0

    allocs = np.ones(prices.shape[0])
    normalized = prices / prices.values[0]
    # Reallocate the allocations with normalized data
    alloced = normalized * allocs
    # Calculate the position values
    pos_vals = start_val * alloced
    # Calculate the portfolio values
    port_val = np.sum(pos_vals, axis=0)
    # Calculate daily return
    if not np.shape(port_val):
        # This mean there is just one stock
        daily_ret = (pos_vals / pos_vals.shift(1)) - 1
    else:
        daily_ret = (port_val / port_val.shift(1)) - 1

    # volatility = pd.rolling_std(daily_ret, N)
    # std_daily_ret = daily_ret.std()
    # mean_daily_ret = daily_ret.mean()
    std_vol = daily_ret.rolling(window=N, center=False).std()

    # mean_vol = pd.rolling_mean(daily_ret, N)
    mean_vol = daily_ret.rolling(window=N, center=False).mean()
    # sharpe_ratio = np.sqrt(252) * mean_vol/volatility
    # sharpe_ratio = sharpe_ratio.rename('sharpe_ratio')

    volatility = (daily_ret - mean_vol) / std_vol

    mean_vol = mean_vol.rename('average_daily_ret')
    std_vol = std_vol.rename('stdev_daily_ret')
    volatility = volatility.rename('volatility')
    daily_ret = daily_ret.rename('daily_return')

    if genplot:
        # all_volatile = pd.concat([daily_ret,mean_vol, volatility, sharpe_ratio], axis=1)
        all_volatile = pd.concat([daily_ret, mean_vol, std_vol], axis=1)
        # all_volatile = pd.concat([daily_ret, volatility], axis=1)
        plot_data(all_volatile, title='Volatility_components', xlabel='times', ylabel='return')
        plt.close()
        plot_data(volatility, title='Volatility', xlabel='times', ylabel='return')

    return volatility, daily_ret

def get_momentum(prices, N, genplot=False):
    m = []
    count = np.shape(prices)[0]
    temp = 100000
    for i in range(count):
        if i - (N-1) >= 0:
            m.append((prices[i] / prices[i-N]) - 1.0)
            if i < temp:
                first = m[i]
                temp = i
            m[i] = m[i]/first
        else:
            m.append(None)

    momentum = pd.DataFrame(data= m, index=prices._index, columns=['momentum'])
    if genplot:
        # all_momentum = pd.concat([momentum['momentum'], prices], axis=1)
        plot_data(momentum, title='Momentum', xlabel='times', ylabel='delta')
    return momentum['momentum']

def indicator(data, N=20, sv = 100000, genplot = False):
    data.fillna(method="ffill",inplace=True)
    data.fillna(method="bfill", inplace=True)
    # N = 20
    bb, bb_upper, bb_lower, sma = bb_value(data, N, genplot)
    volatile, daily_ret = get_volatility(prices=data, N= N, genplot= genplot)
    m = get_momentum(data, N, genplot)
    all_indicators = pd.concat([bb, volatile, m], axis=1)
    return all_indicators

def test_code():
    plt.show(block=False)
    training = get_data(symbols=['JPM'], dates=pd.date_range('01-01-08', '12-31-09'))
    testing = get_data(symbols=['JPM'], dates=pd.date_range('01-01-10', '12-31-11'))
    test2 = indicator(training['JPM'], genplot=True)
    test1 = indicator(testing['JPM'], genplot=False)
if __name__ == "__main__":
    test_code()

