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
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose=False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="ML4T-220", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 12, 31), \
                    sv=100000):

        # self.learner = ql.QLearner()
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        start_val = sv

        if self.verbose: print prices

        # Calculate indicators

        # Momentum indicator
        window = int(21)
        momentum = []
        price_array = pd.DataFrame.as_matrix(prices)
        for i in range(prices.shape[0]):
            if i > window:
                momentum_value = (price_array[i] / price_array[i - window]) - 1.0
                momentum.append(momentum_value)

        pmomentum_value = prices.copy()
        pmomentum_value[:] = 0.0
        norm_momentum = (momentum - np.mean(momentum)) / np.std(momentum)
        pmomentum_value.ix[window + 1:, :] = norm_momentum[:]  # [0]
        pmomentum_value.ix[:window, :] = norm_momentum[20][0]

        # Bollinger Bands
        rolling_std = pd.rolling_std(prices, window=21)
        rolling_avg = pd.rolling_mean(prices, window=21)
        bb_upper = rolling_avg + 2.0 * rolling_std
        bb_lower = rolling_avg - 2.0 * rolling_std
        bbp_band = (prices - bb_lower) / (bb_upper - bb_lower)
        normed_bbp = (bbp_band - np.mean(bbp_band)) / np.std(bbp_band)
        normed_bbp.ix[:21, :] = pd.DataFrame.as_matrix(normed_bbp)[20]

        # SMA/Price indicator
        sma = rolling_avg / prices
        normed_sma = (sma - np.mean(sma)) / np.std(sma)
        normed_sma.ix[:21, :] = pd.DataFrame.as_matrix(normed_sma)[20]

        # Discritize indicators
        nsteps = 4
        momen_values = pmomentum_value.values  # .ix[21:,:].values
        sma_values = normed_sma.values  # .ix[21:,:].values
        bbp_values = normed_bbp.values  # ix[21:,:].values

        momen_array = np.zeros(len(momen_values))
        sma_array = np.zeros(len(sma_values))
        bbp_array = np.zeros(len(sma_values))

        for i in range(len(momen_values)):
            momen_array[i] = momen_values[i]
            sma_array[i] = sma_values[i]
            bbp_array[i] = bbp_values[i]

        mout, self.mbins = pd.qcut(momen_array, nsteps, retbins=True)
        sout, self.sbins = pd.qcut(sma_array, nsteps, retbins=True)
        bout, self.bbins = pd.qcut(bbp_array, nsteps, retbins=True)

        momen_states = pd.cut(momen_array, bins=self.mbins, labels=False, include_lowest=True)
        sma_states = pd.cut(sma_array, bins=self.sbins, labels=False, include_lowest=True)
        bbp_states = pd.cut(bbp_array, bins=self.bbins, labels=False, include_lowest=True)

        converged = False
        total_states = nsteps ** 3
        count = 0
        check_conv = 0.0

        # print "Total number of states is:", total_states

        # Initialize QLearner, SET rar to zero
        self.learner = ql.QLearner(num_states=total_states, \
                                   num_actions=3, \
                                   alpha=0.5, \
                                   gamma=0.9, \
                                   rar=0.0, \
                                   radr=0.0, \
                                   dyna=0, \
                                   verbose=False)  # initialize the learner

        while (not converged) and (count < 20):
            # Set first state to the first data point (first day)
            result = 0
            indicators = [momen_states[0], sma_states[0], bbp_states[0]]
            for kl in range(len(indicators)):
                result = result + indicators[kl] * (nsteps ** kl)
            first_state = result

            action = self.learner.querysetstate(first_state)
            total_reward = 0

            df_prices = prices.copy()
            df_prices['Cash'] = pd.Series(1.0, index=prices.index)
            df_trades = df_prices.copy()
            df_trades[:] = 0.0
            indices = df_prices.index
            holdings = 0

            # Cycle through dates
            for j in range(1, pmomentum_value.shape[0]):
                daily_ret = 0.0
                result = 0
                indicators = [momen_states[j], sma_states[j], bbp_states[j]]
                for k in range(len(indicators)):
                    result = result + indicators[k] * (nsteps ** k)
                new_combined_state = result

                # Calculate reward for previous action
                if j == 0:
                    reward = 0
                else:
                    daily_ret = ((price_array[j] - price_array[j - 1]) / price_array[j]) * 100.0
                    if holdings == 0:
                        reward = 0
                    else:
                        if action == 0:
                            reward = daily_ret
                        elif action == 1:
                            reward = -1.0 * daily_ret
                        else:
                            reward = -10

                            # Query learner with current state and reward to get action

                action = self.learner.query(new_combined_state, reward)

                # Implement action returned by learner and update portfolio
                rDates = indices[j]
                if action == 0:
                    if holdings == -200:
                        # Buy if holdings permit
                        df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 400.0 * df_prices[syms[0]].ix[
                            rDates] * -1.0
                        df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] + 400.0
                        holdings = holdings + 400.0
                    if holdings == 0:
                        df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 200.0 * df_prices[syms[0]].ix[
                            rDates] * -1.0
                        df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] + 200.0
                        holdings = holdings + 200.0
                if action == 1:
                    if holdings == 200:
                        df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 400.0 * df_prices[syms[0]].ix[
                            rDates]
                        df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] - 400.0
                        holdings = holdings - 400.0
                    if holdings == 0:
                        df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 200.0 * df_prices[syms[0]].ix[
                            rDates]
                        df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] - 200.0
                        holdings = holdings - 200.0
                total_reward += reward

            df_holdings = df_trades.copy()
            df_holdings[:] = 0.0
            df_values = df_holdings.copy()

            # Populate holdings dataframe, calculate value of holdings and portfolio value
            df_holdings['Cash'].ix[indices[0]] = start_val
            df_holdings.ix[indices[0]] = df_holdings.ix[indices[0]] + df_trades.ix[indices[0]]

            for i in range(1, len(indices)):
                hDates = indices[i - 1]
                rDates = indices[i]
                df_holdings.ix[rDates] = df_holdings.ix[hDates] + df_trades.ix[rDates]

            df_values = df_holdings * df_prices
            df_port_val = df_values.sum(axis=1)
            period_end = df_port_val.ix[-1]
            commul = df_port_val.pct_change()
            cum_ret = (period_end - start_val) / start_val
            count += 1

            if abs((check_conv - cum_ret) * 100.0) < 0.00001:
                converged = True
            else:
                check_conv = cum_ret

        return df_trades

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="ML4T-220", \
                   sd=dt.datetime(2010, 1, 1), \
                   ed=dt.datetime(2011, 12, 31), \
                   sv=100000):

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        start_val = sv

        # Calculate indicators

        # Momentum indicator
        window = int(21)
        momentum = []
        price_array = pd.DataFrame.as_matrix(prices)
        for i in range(prices.shape[0]):
            if i > window:
                momentum_value = (price_array[i] / price_array[i - window]) - 1.0
                momentum.append(momentum_value)

        pmomentum_value = prices.copy()
        pmomentum_value[:] = 0.0
        norm_momentum = (momentum - np.mean(momentum)) / np.std(momentum)
        pmomentum_value.ix[window + 1:, :] = norm_momentum
        pmomentum_value.ix[:window, :] = norm_momentum[20][0]

        # Bollinger Bands
        rolling_std = pd.rolling_std(prices, window=21)
        rolling_avg = pd.rolling_mean(prices, window=21)
        bb_upper = rolling_avg + 2.0 * rolling_std
        bb_lower = rolling_avg - 2.0 * rolling_std
        bbp_band = (prices - bb_lower) / (bb_upper - bb_lower)
        normed_bbp = (bbp_band - np.mean(bbp_band)) / np.std(bbp_band)
        normed_bbp.ix[:21, :] = pd.DataFrame.as_matrix(normed_bbp)[20]

        # SMA/Price indicator
        sma = rolling_avg / prices
        normed_sma = (sma - np.mean(sma)) / np.std(sma)
        normed_sma.ix[:21, :] = pd.DataFrame.as_matrix(normed_sma)[20]

        nsteps = 4
        momen_values = pmomentum_value.values  # .ix[21:,:].values
        sma_values = normed_sma.values  # .ix[21:,:].values
        bbp_values = normed_bbp.values  # ix[21:,:].values

        momen_array = np.zeros(len(momen_values))
        sma_array = np.zeros(len(sma_values))
        bbp_array = np.zeros(len(sma_values))

        for i in range(len(momen_values)):
            momen_array[i] = momen_values[i]
            sma_array[i] = sma_values[i]
            bbp_array[i] = bbp_values[i]

        momen_states = pd.cut(momen_array, bins=self.mbins, labels=False, include_lowest=True)
        sma_states = pd.cut(sma_array, bins=self.sbins, labels=False, include_lowest=True)
        bbp_states = pd.cut(bbp_array, bins=self.bbins, labels=False, include_lowest=True)

        total_states = nsteps ** 3

        df_prices = prices.copy()
        df_prices['Cash'] = pd.Series(1.0, index=prices.index)
        df_trades = df_prices.copy()
        df_trades[:] = 0.0
        indices = df_prices.index
        holdings = 0

        # Cycle through dates
        for j in range(pmomentum_value.shape[0]):
            daily_ret = 0.0
            result = 0
            indicators = [momen_states[j], sma_states[j], bbp_states[j]]

            for k in range(len(indicators)):
                if indicators[k] not in range(total_states):  # == np.nan:
                    indicates = 0
                else:
                    indicates = indicators[k]
                result = result + indicates * (nsteps ** k)
            # result = result + indicators[k]*(nsteps**k)
            new_combined_state = result

            # Query learner with current state and reward to get action

            action = self.learner.querysetstate(new_combined_state)
            # Implement action returned by learner and update portfolio
            rDates = indices[j]
            if action == 0:
                if holdings == -200:
                    # Buy if holdings permit
                    df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 400.0 * df_prices[syms[0]].ix[
                        rDates] * -1.0
                    df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] + 400.0
                    holdings = holdings + 400.0
                if holdings == 0:
                    df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 200.0 * df_prices[syms[0]].ix[
                        rDates] * -1.0
                    df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] + 200.0
                    holdings = holdings + 200.0
            if action == 1:
                if holdings == 200:
                    df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 400.0 * df_prices[syms[0]].ix[rDates]
                    df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] - 400.0
                    holdings = holdings - 400.0
                if holdings == 0:
                    df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + 200.0 * df_prices[syms[0]].ix[rDates]
                    df_trades[syms[0]].ix[rDates] = df_trades[syms[0]].ix[rDates] - 200.0
                    holdings = holdings - 200.0

        df_holdings = df_trades.copy()
        df_holdings[:] = 0.0
        df_values = df_holdings.copy()

        # Populate holdings dataframe, calculate value of holdings and portfolio value
        df_holdings['Cash'].ix[indices[0]] = start_val
        df_holdings.ix[indices[0]] = df_holdings.ix[indices[0]] + df_trades.ix[indices[0]]

        for i in range(1, len(indices)):
            hDates = indices[i - 1]
            rDates = indices[i]
            df_holdings.ix[rDates] = df_holdings.ix[hDates] + df_trades.ix[rDates]

        df_values = df_holdings * df_prices
        df_port_val = df_values.sum(axis=1)
        period_end = df_port_val.ix[-1]
        commul = df_port_val.pct_change()
        cum_ret = (period_end - start_val) / start_val
        print cum_ret * 100
        return df_trades

if __name__ == "__main__":
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