# In ManualStrategy.py implement a set of rules using the indicators you created in Part 1 above. Devise some simple logic using your indicators to enter and exit positions in the stock.
#
# A recommended approach is to create a single logical expression that yields a -1, 0, or 1, corresponding to a "short," "out" or "long" position. Example usage this signal: If you are out of the stock, then a 1 would signal a BUY 1000 order. If you are long, a -1 would signal a SELL 2000 order. You don't have to follow this advice though, so long as you follow the trading rules outlined above.
#
# For the report we want a written description, not code, however, it is OK to augment your written description with a pseudocode figure.
#
# You should tweak your rules as best you can to get the best performance possible during the in sample period (do not peek at out of sample performance). Use your rule-based strategy to generate an trades dataframe over the in sample period, then run that dataframe through your market simulator to create a chart that includes the following components over the in sample period:
#
# Benchmark (see definition above) normalized to 1.0 at the start: Green line
# Value of the rule-based portfolio (normalized to 1.0 at the start): Red line
# Vertical blue lines indicating LONG entry points.
# Vertical black lines indicating SHORT entry points.
# We expect that your rule-based strategy should outperform the benchmark over the in sample period.
#
# Your code should implement the same API as above for theoretically optimal:
#
#    df_trades = ms.testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)

import numpy as np
import pandas as pd
from marketsimcode import compute_portvals, evaluate_port_val
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
from indicators import indicator

def author():
    return 'cli620'

class ManualStrategy():
    def __init__(self, commissions=0.0, impact=0.0):
        self.commissions = commissions
        self.impact = impact
        self.startVal = 100000
        # self.benchmark = pd.DataFrame() # need +1000 shares of JPM and hold that position.
                                        # Only Allowable positions are +1000 shares -1000 shares or 0 shares.

    def testPolicy(self, symbol = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000, window=7):
        self.startVal = sv
        # window =

        # grab all the data
        date_list = pd.date_range(sd, ed)
        all_prices = get_data([symbol], date_list)
        jpm_prices = all_prices['JPM']
        self.prices = jpm_prices
        # get all the indicators for evaluation.
        all_indicators = indicator(data=jpm_prices, N=window, genplot=False)

        # initialize the df_trades output dataframe
        self.df_trades = pd.concat([pd.Series(data=[symbol] * jpm_prices.shape[0], index=jpm_prices._index, name='Symbol'),
                               pd.Series(data=["BUY"] * jpm_prices.shape[0], index=jpm_prices._index, name='Order'),
                               pd.Series(data=[None] * jpm_prices.shape[0], index=jpm_prices._index, name='Shares')], axis=1)

        df_trades = pd.DataFrame(data=[None]*jpm_prices.shape[0], index=jpm_prices._index, columns=['values'])

        # set up the benchmark data frame
        self.benchmark = pd.concat([pd.Series(data=[symbol] * jpm_prices.shape[0], index=jpm_prices._index, name='Symbol'),
                               pd.Series(data=["BUY"] * jpm_prices.shape[0], index=jpm_prices._index, name='Order'),
                               pd.Series(data=[0.0] * jpm_prices.shape[0], index=jpm_prices._index, name='Shares')], axis=1)

        # self.benchmark["Order"][0] = "BUY"
        # self.benchmark["Shares",0] = 1000.0
        self.benchmark["Shares"].values[0] = 1000.0
        # loop through each day and make the decision to have a max short (-1000) or buy (1000) or do nothing.
        # Depending on seeing future data.
        netholding = 0.0 # Only possible netholdings are 0.0 | +1000.0 | -1000.0
        allnet = []
        oldvolatility = -9999999
        old_momentum = 0.0
        short_decisions=[]
        long_decisions=[]
        # loop through all the days
        for t in range(jpm_prices.shape[0]-1):
            # just record the current state.
            thisdate = jpm_prices._index[t]
            thisprice = jpm_prices[t]
            nextprice = jpm_prices[t+1]
            curr_indicators = all_indicators.ix[t]

            bbweight= 6.0 # Bollinger Band
            mweight= 6.0 # Momentum
            vweight= 3.0 # Volatility

            decision = 0.0
            if np.isnan(curr_indicators['bb']) and np.isnan(curr_indicators['volatility']) and np.isnan(curr_indicators['momentum']):
                decision = 0.0
            else:
                # decide on bollinger band output for this day.
                if np.isnan(curr_indicators['bb']):
                    bbflag = 0.0
                else:
                    if curr_indicators['bb'] >= 1.0:
                        bbflag = -1.0
                    elif curr_indicators['bb'] <= -1.0:
                        bbflag = 1.0
                    else:
                        bbflag = 0.0

                # decide on momentum flag
                if np.isnan(curr_indicators['momentum']):
                    mflag = 0.0
                else:
                    # Look at the delta momentum --> at low/high peaks buy/sell
                    delta_momentum = curr_indicators['momentum'] - old_momentum
                    if abs(delta_momentum) < 1.0:
                        mflag = 0.0
                    else:
                        derivative = np.diff(all_indicators.ix[t-window:t]).mean()
                        if derivative <= 0.0:
                            mflag = 1.0
                        else:
                            mflag = -1.0
                    # if curr_indicators['momentum'] > 1.0:
                    #     mflag = 1.0
                    # elif curr_indicators['momentum'] < -1.0:
                    #     mflag = -1.0
                    # else:
                    #     mflag = 0.0
                    old_momentum = curr_indicators['momentum']

                # We want to buy more volatile stocks --> we are trading very often.
                if np.isnan(curr_indicators['volatility']):
                    vflag = 0.0
                else:
                    # if curr_indicators['volatility'] > oldvolatility:
                    if abs(curr_indicators['volatility']) < 0.5: # three sigma
                        vflag = -1.0
                    else:
                        vflag = 1.0
                        # if abs(curr_indicators['volatility']) > 1.0:
                        #     vflag = 1.0
                        # else:
                        #     # vflag = -1.0
                        #     vflag = 1.0
                    oldvolatility = np.copy(curr_indicators['volatility'])


                # decision = vflag*(vweight /(vweight+mweight+bbweight)) + \
                #            mflag*(mweight /(vweight+mweight+bbweight)) + \
                #            bbflag*(bbweight /(vweight+mweight+bbweight))

                # https: // www.investopedia.com / terms / b / bollingerbands.asp
                # Squeeze --> period lo volatility --> possible trading opportunities
                #         --> decrease of volatility --> possible exiting trade.
                #         --> lo volatility bad | hi volatility good
                # Breakout --> break out top --> sell
                #          --> break out bottom --> buy
                # Momentum --> positive momentum & break out up BUY!
                #          --> negative momentum & break out up SELL! # or don't do anything
                #          --> positive momentum & break out down BUY!
                #          --> negative momentum & break out down SELL! # or don't do anything

                decision = (-2.0*bbflag + mflag)*0.5 + vflag * 0.5
                # decision = ((-1.0 * bbflag + mflag) * 2.0 - 1.0) * 0.25 + vflag * 0.75
                # decision = (vflag * mflag) * (vweight + mweight)/(vweight + mweight + bbweight) + \
                #            bbflag * (bbweight)/(vweight + mweight + bbweight)

            # possible trades --> mod of +/- 2000.0, 1000.0, 0.0
            # As long as thet net sums up to -1000.0 | 1000.0 | 0.0
            if decision >= 1.0:
                # if the netholding is 1000.0 -->
                delta = -1000.0 - netholding
                self.df_trades["Shares"][t] = abs(delta)
                if np.sign(delta) < 0.0:
                    self.df_trades["Order"][t] = "SELL"
                short_decisions.append([self.df_trades.index[t]])
                df_trades['values'][t] = delta

            elif decision <= -1.0:
                delta = 1000.0 - netholding
                self.df_trades["Shares"][t] = abs(delta)
                if np.sign(delta) < 0.0:
                    self.df_trades["Order"][t] = "SELL"
                long_decisions.append([self.df_trades.index[t]])
                df_trades['values'][t] = delta

            # elif decision == 0.0:
            else:
                delta = 0.0
                self.df_trades["Shares"][t] = 0.0
                df_trades['values'][t] = delta
            # else:
            #     print(['wtf? ']*50)

            netholding += delta
            allnet.append(netholding)
        self.df_trades["Shares"][t+1] = 0.0
        df_trades['values'][t] = 0.0

        self.long_decisions = long_decisions
        self.short_decisions = short_decisions
        # self.df_trades = df_trades

        return df_trades

    def evaluate_train(self):
        bm_trades = self.benchmark
        ms_port_vals = compute_portvals(self.df_trades, start_val=self.startVal, commission=self.commissions, impact=self.impact)
        bm_port_vals = compute_portvals(bm_trades, start_val=self.startVal, commission=self.commissions, impact=self.impact)

        # ax_p = self.prices.plot(title='Training Prices', fontsize=12)
        # ax_p.set_xlabel('dates')
        # ax_p.set_ylabel('prices')
        # plt.savefig('train_prices.png')
        # plt.tight_layout()
        # plt.close()

        # Normalize the port vals output.
        ms_port_vals = ms_port_vals / ms_port_vals[0][0]
        ms_port_vals = ms_port_vals[0].rename('Manual')
        bm_port_vals = bm_port_vals / bm_port_vals[0][0]
        bm_port_vals = bm_port_vals[0].rename('BenchMark')

        ms_cum_ret, ms_avg_daily_ret, ms_std_daily_ret, ms_sharpe_ratio = evaluate_port_val(port_val=ms_port_vals,
                                                                                            gen_plot=False)
        bm_cum_ret, bm_avg_daily_ret, bm_std_daily_ret, bm_sharpe_ratio = evaluate_port_val(port_val=bm_port_vals,
                                                                                            gen_plot=False)

        # put the port vals together for plotting
        all_port_vals = pd.concat([ms_port_vals, bm_port_vals], axis=1)

        # plot the port vals --> theoretical vs benchmark.
        ax = all_port_vals.plot(title='TRAIN Manual Vs BenchMark', fontsize=12, color=['r', 'g'])
        [ax.axvline(x=self.short_decisions[i], color='k') for i in range(len(self.short_decisions))]
        [ax.axvline(x=self.long_decisions[i], color='b') for i in range(len(self.long_decisions))]
        ax.set_xlabel('dates')
        ax.set_ylabel('Portfolio Values (Normalized $)')
        plt.tight_layout()
        plt.savefig('TRAIN_Manual_vs_Benchmark.png')
        plt.close()

        # Compare portfolio against $SPX
        print "Date Range: {} to {}".format(start_date, end_date)
        print
        print "Cumulative Return of {}: {}".format(symbol, ms_cum_ret)
        print "Cumulative Return of BenchMark : {}".format(bm_cum_ret)
        print
        print "Standard Deviation of {}: {}".format(symbol, ms_std_daily_ret)
        print "Standard Deviation of BenchMark : {}".format(bm_std_daily_ret)
        print
        print "Average Daily Return of {}: {}".format(symbol, ms_avg_daily_ret)
        print "Average Daily Return of BenchMark : {}".format(bm_avg_daily_ret)
        print
        print "Sharpe Ratio of {}: {}".format(symbol, ms_sharpe_ratio)
        print "Sharpe Ratio of BenchMark : {}".format(bm_sharpe_ratio)
        print
        print "Final Portfolio Value: {}".format(ms_port_vals[-1])
        print "Final Benchmark Value: {}".format(bm_port_vals[-1])
        print "-" * 50

    def evaluate_test(self):
        bm_trades = self.benchmark
        ms_port_vals = compute_portvals(self.df_trades, start_val=self.startVal, commission=self.commissions,
                                        impact=self.impact)
        bm_port_vals = compute_portvals(bm_trades, start_val=self.startVal, commission=self.commissions, impact=self.impact)

        # ax_p = self.prices.plot(title='Test Prices', fontsize=12)
        # ax_p.set_xlabel('dates')
        # ax_p.set_ylabel('prices')
        # plt.savefig('test_prices.png')
        # plt.tight_layout()
        # plt.close()

        # Normalize the port vals output.
        ms_port_vals = ms_port_vals / ms_port_vals[0][0]
        ms_port_vals = ms_port_vals[0].rename('Manual')
        bm_port_vals = bm_port_vals / bm_port_vals[0][0]
        bm_port_vals = bm_port_vals[0].rename('BenchMark')

        ms_cum_ret, ms_avg_daily_ret, ms_std_daily_ret, ms_sharpe_ratio = evaluate_port_val(port_val=ms_port_vals,
                                                                                            gen_plot=False)
        bm_cum_ret, bm_avg_daily_ret, bm_std_daily_ret, bm_sharpe_ratio = evaluate_port_val(port_val=bm_port_vals,
                                                                                            gen_plot=False)

        # put the port vals together for plotting
        all_port_vals = pd.concat([ms_port_vals, bm_port_vals], axis=1)

        # plot the port vals --> theoretical vs benchmark.
        ax = all_port_vals.plot(title='TEST Manual Vs BenchMark', fontsize=12, color=['r', 'g'])
        # [ax.axvline(x=self.short_decisions[i], color='k') for i in range(len(self.short_decisions))]
        # [ax.axvline(x=self.long_decisions[i], color='b') for i in range(len(self.long_decisions))]
        ax.set_xlabel('dates')
        ax.set_ylabel('Portfolio Values (Normalized $)')
        plt.tight_layout()
        plt.savefig('TEST_Manual_vs_Benchmark.png')
        plt.close()

        # Compare portfolio against $SPX
        print "Date Range: {} to {}".format(start_date, end_date)
        print
        print "Cumulative Return of {}: {}".format(symbol, ms_cum_ret)
        print "Cumulative Return of BenchMark : {}".format(bm_cum_ret)
        print
        print "Standard Deviation of {}: {}".format(symbol, ms_std_daily_ret)
        print "Standard Deviation of BenchMark : {}".format(bm_std_daily_ret)
        print
        print "Average Daily Return of {}: {}".format(symbol, ms_avg_daily_ret)
        print "Average Daily Return of BenchMark : {}".format(bm_avg_daily_ret)
        print
        print "Sharpe Ratio of {}: {}".format(symbol, ms_sharpe_ratio)
        print "Sharpe Ratio of BenchMark : {}".format(bm_sharpe_ratio)
        print
        print "Final Portfolio Value: {}".format(ms_port_vals[-1])
        print "Final Benchmark Value: {}".format(bm_port_vals[-1])


if __name__ == "__main__":

    ms = ManualStrategy(commissions=9.95, impact=0.005)
    window = 20
    start_val = 100000

    train_start_date = dt.datetime(2008, 1, 1)
    train_end_date = dt.datetime(2009, 12, 31)
    # training = get_data(symbols=['JPM'], dates=pd.date_range('01-01-08', '12-31-09'))

    test_start_date = dt.datetime(2010, 1, 1)
    test_end_date = dt.datetime(2011, 12, 31)
    # testing = get_data(symbols=['JPM'], dates=pd.date_range('01-01-10', '12-31-11'))

    # start_date = dt.datetime(2010, 1, 1)
    # end_date = dt.datetime(2011,12,31)
    symbol = "JPM"

    ### PART 3 --> Manual Rule-Based Trader
    ## TRAINING PART
    start_date = train_start_date
    end_date = train_end_date
    ms.evaluate_train()


    ### PART 4: Comparative Analysis
    ## TESTING PART
    start_date = test_start_date
    end_date = test_end_date
    df_trades = ms.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val, window=window)
    ms.evaluate_test()
