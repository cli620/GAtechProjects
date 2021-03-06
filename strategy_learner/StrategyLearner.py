"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import datetime as dt
import numpy as np
import pandas as pd
import util as ut
from QLearner import QLearner
from indicators import indicator
from marketsimcode import compute_portvals, evaluate_port_val
import random


def author(self):
    return 'cli620'


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def author(self):
        return 'cli620'


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 10000):

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        window = 20 # from the manual strategy.
        # epochs = 30 # ??? DO WE NEED EPOCHS???
        epochs = 3
        # get all the indicators for evaluation.
        all_indicators = indicator(data=prices, N=window, genplot=False)
        # using Qlearner -->
        # no dyna --> you're hallucinating about the past data.
        # num action --> -1 0 1
        # num states --> number of steps raised to the number of inputs/indicators
        steps = 10
        self.learner = QLearner(num_states=steps**len(all_indicators.columns), num_actions=3, alpha=0.0, gamma=1.35, rar=0.99, radr=0.75,dyna=0, verbose=False)
        # self.learner = QLearner(num_states=steps**len(all_indicators.columns), num_actions=3, alpha=0.5, gamma=1.3, rar=0.1, radr=0.9,dyna=0, verbose=False)
        qcut_mom = pd.qcut(x=all_indicators['momentum'], q=10, labels=range(10))
        qcut_vol = pd.qcut(x=all_indicators['volatility'], q=10, labels=range(10))
        qcut_bb = pd.qcut(x=all_indicators['bb'], q=10, labels=range(10))
        # 0 - 10 --> 5 = neutral. if Nan then make 5.
        qcut_mom = qcut_mom.fillna(5).values.codes.astype('float')
        qcut_vol = qcut_vol.fillna(5).values.codes.astype('float')
        qcut_bb = qcut_bb.fillna(5).values.codes.astype('float')
        # digitized_indicator
        # this will be our state --> 1000 possible variations.
        dig_indicator = qcut_mom*100 + qcut_vol*10 + qcut_bb

        # initiate the trade data frames --> one for
        df_trades = pd.concat(
            [pd.Series(data=[symbol] * prices.shape[0], index=prices._index, name='Symbol'),
             pd.Series(data=["BUY"] * prices.shape[0], index=prices._index, name='Order'),
             pd.Series(data=[None] * prices.shape[0], index=prices._index, name='Shares')], axis=1)

        out_df_trades = pd.DataFrame(data=[None]*prices.shape[0], index=prices._index, columns=['values'])

        count = 0
        notconverged = True
        old_cr = -99999999999999
        while count < epochs and notconverged:
            if self.verbose: print('epoch = ', count)
            total_reward = 0
            state = int(dig_indicator[0])
            action = self.learner.querysetstate(state)
            netholding = 0.0
            t = 0
            if action < 1.0:
                # short position
                delta = -1000.0 - netholding
                df_trades["Shares"][t] = abs(delta)
                if np.sign(delta)<0.0:
                    df_trades["Order"][t] = "SELL"
                # easydf = delta
            elif action > 1.0:
                # long position
                delta = 1000.0 - netholding
                df_trades["Shares"][t] = abs(delta)
                if np.sign(delta)<0.0:
                    df_trades["Order"][t] = "SELL"
                # easydf = delta
            else:
                delta = 0.0
                df_trades["Shares"][t] = 0.0
                # easydf = 0.0

            out_df_trades['values'][t]=delta
            netholding += delta
            # df_trades["Shares"][t] = 0.0
            # easydf['values'][t] = 0.0
            # thisportval = compute_portvals(orders_df=df_trades[0:t+1], start_val=sv, commission=0.0, impact=self.impact)

            while t < prices.shape[0]-1: # & money >0:
                # look at historical data
                # determine the next move.
                if self.verbose: print('we are on day: ', df_trades.index[t])
                if t == prices.shape[0]-1:
                    shit = 1

                # update state
                state = int(dig_indicator[t])

                # get this update reward?

                if t==0:
                    thisportval = compute_portvals(orders_df=df_trades.ix[0:1], start_val=sv, commission=0.0,impact=self.impact)
                    reward = sv - thisportval[0].values[0]
                else:
                    thisportval = compute_portvals(orders_df=df_trades.ix[t-1:t+1], start_val=sv, commission=0.0,impact=self.impact)
                    adjust = 1.0
                    if netholding < 0:
                        # we are shorting
                        adjust = -1.0
                    elif netholding == 0:
                        adjust = 0.0
                    reward = (thisportval[0].values[1]-thisportval[0].values[0])*adjust
                # reward =

                t += 1

                # get new action
                action = self.learner.query(s_prime=state, r=reward)
                if action < 1.0:
                    # short position
                    delta = -1000.0 - netholding
                    df_trades["Shares"][t] = abs(delta)
                    if np.sign(delta)<0.0:
                        df_trades["Order"][t] = "SELL"
                    # easydf = delta
                elif action > 1.0:
                    # long position
                    delta = 1000.0 - netholding
                    df_trades["Shares"][t] = abs(delta)
                    if np.sign(delta)<0.0:
                        df_trades["Order"][t] = "SELL"
                    # easydf = delta
                else:
                    delta = 0.0
                    df_trades["Shares"][t] = 0.0
                    # easydf = 0.0
                out_df_trades['values'][t] = delta
                netholding += delta
                total_reward += reward

            # evalute the port values
            thispv = compute_portvals(df_trades, start_val=sv, commission=0.0,impact=self.impact)
            # this port value evaluation stuff.
            cr, adr, sddr, sr = evaluate_port_val(thispv)

            if abs(cr[0] - old_cr) < 0.001:
                notconverged=False
            else:
                old_cr=cr[0]

            count += 1
        self.df_trades = df_trades
        return out_df_trades
        # example use with new colname
        # volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print volume


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        window = 20  # from the manual strategy.
        # get all the indicators for evaluation.
        all_indicators = indicator(data=prices, N=window, genplot=False)

        self.learner.rar = -1.0
        self.learner.radr = 0.0

        qcut_mom = pd.qcut(x=all_indicators['momentum'], q=10, labels=range(10))
        qcut_vol = pd.qcut(x=all_indicators['volatility'], q=10, labels=range(10))
        qcut_bb = pd.qcut(x=all_indicators['bb'], q=10, labels=range(10))
        # 0 - 10 --> 5 = neutral. if Nan then make 5.
        qcut_mom = qcut_mom.fillna(5).values.codes.astype('float')
        qcut_vol = qcut_vol.fillna(5).values.codes.astype('float')
        qcut_bb = qcut_bb.fillna(5).values.codes.astype('float')
        # digitized_indicator
        # this will be our state --> 1000 possible variations.
        dig_indicator = qcut_mom*100 + qcut_vol*10 + qcut_bb
        df_trades = pd.concat(
            [pd.Series(data=[symbol] * prices.shape[0], index=prices._index, name='Symbol'),
             pd.Series(data=["BUY"] * prices.shape[0], index=prices._index, name='Order'),
             pd.Series(data=[None] * prices.shape[0], index=prices._index, name='Shares')], axis=1)

        out_df_trades = pd.DataFrame(data=[None]*prices.shape[0], index=prices._index, columns=['values'])

        netholding = 0.0
        for t in range(prices.shape[0]):
            state = int(dig_indicator[t])
            action = self.learner.querysetstate(state)
            # action = np.argmax(self.learner.Q[state,:])
            if action < 1.0:
                # short position
                delta = -1000.0 - netholding
                df_trades["Shares"][t] = abs(delta)
                if np.sign(delta)<0.0:
                    df_trades["Order"][t] = "SELL"
                # easydf = delta
            elif action > 1.0:
                # long position
                delta = 1000.0 - netholding
                df_trades["Shares"][t] = abs(delta)
                if np.sign(delta)<0.0:
                    df_trades["Order"][t] = "SELL"
                # easydf = delta
            else:
                delta = 0.0
                df_trades["Shares"][t] = 0.0
                # easydf = 0.0
            out_df_trades['values'][t] = delta
            netholding += delta
        self.df_trades = df_trades
        # here we build a fake set of trades
        # your code should return the same sort of data
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        # trades.values[:,:] = 0 # set them all to nothing
        # trades.values[0,:] = 1000 # add a BUY at the start
        # trades.values[40,:] = -1000 # add a SELL
        # trades.values[41,:] = 1000 # add a BUY
        # trades.values[60,:] = -2000 # go short from long
        # trades.values[61,:] = 2000 # go long from short
        # trades.values[-1,:] = -1000 #exit on the last day
        # if self.verbose: print type(trades) # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print prices_all
        return out_df_trades

if __name__=="__main__":
    print "One does not simply think up a strategy"

    commish = 0.0
    sl = StrategyLearner(verbose=False, impact=0.005)
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
    traindf_trades = sl.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
    # sl.evaluate_train()

    ### PART 4: Comparative Analysis
    ## TESTING PART
    start_date = test_start_date
    end_date = test_end_date
    testdf_trades = sl.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
    testdf_trades2 = sl.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
    print('shit')
    # sl.evaluate_test()
