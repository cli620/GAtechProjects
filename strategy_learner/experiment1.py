import StrategyLearner as slearner
import ManualStrategy as mstrat
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import util as ut
from QLearner import QLearner
from indicators import indicator
from marketsimcode import compute_portvals, evaluate_port_val

def author(self):
    return 'cli620'


if __name__=="__main__":
    commish = 0.0
    sl = slearner.StrategyLearner(verbose=False, impact=0.0)
    ml = mstrat.ManualStrategy(commissions=commish, impact=0.0)
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
    traindf_trades_sl = sl.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
    traindf_trades_ml = ml.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val, window=window)

    train_pv_sl = compute_portvals(orders_df=sl.df_trades, start_val=start_val, commission=commish, impact=0)
    train_pv_ml= compute_portvals(orders_df=ml.df_trades, start_val=start_val, commission=commish, impact=0)

    train_sl_cr, train_sl_dravg,train_sl_drstd,train_sl_sr = evaluate_port_val(port_val=train_pv_sl)
    train_ml_cr, train_ml_dravg,train_ml_drstd,train_ml_sr = evaluate_port_val(port_val=train_pv_ml)
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Cumulative Return SL of {}: {}".format(symbol, train_sl_cr)
    print "Cumulative Return MS of {}: {}".format(symbol, train_ml_cr)
    print
    print "Standard Deviation SL of {}: {}".format(symbol, train_sl_drstd)
    print "Standard Deviation MS of {}: {}".format(symbol, train_ml_drstd)
    print
    print "Mean SL of {}: {}".format(symbol, train_sl_dravg)
    print "Mean MS of {}: {}".format(symbol, train_ml_dravg)
    print
    print "Sharpe Ratio SL of {}: {}".format(symbol, train_sl_sr)
    print "Sharpe Ratio MS of {}: {}".format(symbol, train_ml_sr)
    print
    print "Final SL Portfolio Value: {}".format(train_pv_sl.values[-1])
    print "Final MS Benchmark Value: {}".format(train_pv_ml.values[-1])

    # sl.evaluate_train()

    ### PART 4: Comparative Analysis
    ## TESTING PART
    start_date = test_start_date
    end_date = test_end_date
    testdf_trades_sl = sl.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
    testdf_trades2 = ml.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val, window=window)

    test_pv_sl = compute_portvals(orders_df=sl.df_trades, start_val=start_val, commission=commish, impact=0)
    test_pv_ml= compute_portvals(orders_df=ml.df_trades, start_val=start_val, commission=commish, impact=0)

    test_sl_cr, test_sl_dravg,test_sl_drstd,test_sl_sr = evaluate_port_val(port_val=train_pv_sl)
    test_ml_cr, test_ml_dravg,test_ml_drstd,test_ml_sr = evaluate_port_val(port_val=train_pv_ml)

    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Cumulative Return SL of {}: {}".format(symbol, test_sl_cr)
    print "Cumulative Return MS of {}: {}".format(symbol, test_ml_cr)
    print
    print "Standard Deviation SL of {}: {}".format(symbol, test_sl_drstd)
    print "Standard Deviation MS of {}: {}".format(symbol, test_ml_drstd)
    print
    print "Mean SL of {}: {}".format(symbol, test_sl_dravg)
    print "Mean MS of {}: {}".format(symbol, test_ml_dravg)
    print
    print "Sharpe Ratio SL of {}: {}".format(symbol, test_sl_sr)
    print "Sharpe Ratio MS of {}: {}".format(symbol, test_ml_sr)
    print
    print "Final SL Portfolio Value: {}".format(test_pv_sl.values[-1])
    print "Final MS Portfolio Value: {}".format(test_pv_ml.values[-1])


    # all_port_vals = pd.concat([train_pv_sl, train_pv_ml, test_pv_sl, test_pv_ml])
    # ax = all_port_vals.plot(title='Manual Strat versus Strategy learner', fontsize=12, color=['r', 'b', 'k', 'g'])
    ax1 = train_pv_sl.plot(title='Train Strat Learner port vals', fontsize=12, color='r')
    ax1.set_xlabel('dates')
    ax1.set_ylabel('Port Values')
    plt.tight_layout()
    plt.savefig('SL_pv_Train.png')
    plt.close()

    ax2 = train_pv_ml.plot(title='Train Manual Strat port vals', fontsize=12, color='b')
    ax2.set_xlabel('dates')
    ax2.set_ylabel('Port Values')
    plt.tight_layout()
    plt.savefig('MS_pv_Train.png')
    plt.close()

    ax3 = test_pv_sl.plot(title='Test Strat Learner port vals', fontsize=12, color='r')
    ax3.set_xlabel('dates')
    ax3.set_ylabel('Port Values')
    plt.tight_layout()
    plt.savefig('SL_pv_Test.png')
    plt.close()

    ax4 = test_pv_ml.plot(title='Test Manual Strat port vals', fontsize=12, color='b')
    ax4.set_xlabel('dates')
    ax4.set_ylabel('Port Values')
    plt.tight_layout()
    plt.savefig('MS_pv_Test.png')
    plt.close()
