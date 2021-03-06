"""MC2-P1: Market simulator. 			  		 			 	 	 		 		 	  		   	  			  	

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

import pandas as pd
import numpy as np 			  		 			 	 	 		 		 	  		   	  			  	
import datetime as dt 			  		 			 	 	 		 		 	  		   	  			  	
import os 			  		 			 	 	 		 		 	  		   	  			  	
from util import get_data, plot_data 			  		 			 	 	 		 		 	  		   	  			  	


def author():
    return 'cli620'

def compute_portvals(orders_df, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your 			  		 			 	 	 		 		 	  		   	  			  	
    # code should work correctly with either input 			  		 			 	 	 		 		 	  		   	  			  	

    # pv = []
    # date_list = []
    # portvals = pd.DataFrame

    # get the data and sort it
    # orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df = orders_df.sort_index()

    # initializing parameters we'll need
    cash = float(start_val)

    date_list = pd.date_range(orders_df.axes[0][0], orders_df.axes[0][-1])
    portvals = pd.DataFrame(index = date_list, data = [None]*len(date_list))

    # get all the adjusted prices
    unique_symbols = np.unique(orders_df['Symbol'])
    all_prices = get_data(list(unique_symbols), date_list)

    stock_shares = {}

    # loop through all the orders
    for order_ct in range(orders_df.shape[0]):

        this_order = orders_df.values[order_ct]
        # print('Currently on order: ', this_order)
        # grab the data from the historical list --> we want to find the prices for this stock on this day.
        thisdate = orders_df.axes[0][order_ct]
        stock_name = this_order[0]

        # print('- Updating stocks list prices here ')
        ## UPDATE STOCKS LIST PRICES
        # check to see if we have the stock in our dictionary.
        # if not then add it, if yes then update it.
        if stock_name not in stock_shares:
            newstock = Portfolio_for_one_stock(name=stock_name)
            stock_shares.update({stock_name: newstock})

        # Filter on the current stocks in portfolio and the date
        # allstocks = all_prices[list(stock_shares.keys())].ix[thisdate]
        for key in stock_shares:
            # stock_shares[key].get_current_price(date=thisdate)
            stock_shares[key].price = all_prices[key].ix[thisdate]

        # print('- Updating Cash and share # based on BUY or SELL ')
        # we want to calculate the new cash from this exchange --> -(commission + stock_price*impact)
        cash -= (commission + stock_shares[stock_name].price * impact)# -(commission + stock_price * impact)

        # if we are buying --> update the shares and subtract the money
        if this_order[1] == 'BUY':
            stock_shares[stock_name].update_shares(delta= this_order[2])
            # cash -= stock_shares[stock_name].price*stock_shares[stock_name].shares
            cash -= stock_shares[stock_name].price * this_order[2]
        # if we are selling --> update the shares and add the money
        elif this_order[1] == 'SELL':
            stock_shares[stock_name].update_shares(delta= -this_order[2])
            # cash += stock_shares[stock_name].price * stock_shares[stock_name].shares
            cash += stock_shares[stock_name].price * this_order[2]
        else:
            print(['ERROR!!!! WHY IS THE ORDER NEITHER A BUY OR SELL']*50)

        # This checks to see if it is the last transaction of the day
        if order_ct+1 < orders_df.shape[0]:
            if thisdate == orders_df.axes[0][order_ct+1]:
                newday = False
            else:
                newday = True
                nextday = orders_df.axes[0][order_ct+1]
        else: # This means we are on the last day.
            newday = True
            nextday = thisdate

        # if we see the next day is a new day then we record it. else set it as nan
        if newday:
            # print('-- Next day is a new day! Looping through all future days to fastfoward update based on current portfolio')
            # this day's portfolio value = cash + the current values of equities

            for future_dates in pd.date_range(thisdate, nextday):
                if future_dates in all_prices.axes[0]:
                    # we are fast forwarding to the next day here.
                    equities = 0
                    for key in stock_shares:
                        # stock_shares[key].get_current_price(date=future_dates)
                        # thisprice = allstocks[key].ix[future_dates]
                        thisprice = all_prices[key].ix[future_dates]
                        # equities += stock_shares[key].price * stock_shares[key].shares
                        equities += thisprice * stock_shares[key].shares

                    thisportvals = cash + equities
                    portvals.ix[future_dates] = thisportvals
                else:
                    # This date is actually a weeknd --> drop it.
                    portvals = portvals.drop(future_dates)
                    # thisportvals = None
                    # portvals.ix[future_dates] = thisportvals

    # print(' We are out ! ')
    # print(' FILLING IN THE NaNs!!! ')
    portvals.fillna(method='ffill', inplace=True)
    portvals.fillna(method='bfill', inplace=True)
    # print(' PLotting the data!!!')
    # plot_data(portvals, title='portvals fill forward', xlabel='dates', ylabel=' port vals ')
    return portvals

class Portfolio_for_one_stock:
    def __init__(self, name, price=0, shares=0.0):
        self.name = name
        self.price = price
        self.shares = shares

    def get_current_price(self, date): # not using this anymore --> takes too long to pull individually.
        thisstock = get_data([self.name], pd.date_range(date, date))
        if not thisstock['SPY'].empty: # check to see if it is the weekend.
            self.price = thisstock[self.name][0]

    def update_shares(self, delta):
        self.shares += delta

def str2dt(strng):
    year, month, day = map(int, strng.split('-'))
    return dt.datetime(year, month, day)

def evaluate_port_val(port_val, rf = 0.0,gen_plot=False):
    # filtered_port_val = port_val.ix[sd:ed]
    filtered_port_val = port_val
    daily_ret = (filtered_port_val / filtered_port_val .shift(1)) - 1
    # Calculate the risk free parameter (default to 0)
    daily_rf = np.sqrt(1.0 + rf) - 1
    # Calculate Cumulative Return
    cr = (filtered_port_val.ix[-1] / filtered_port_val.ix[0]) - 1  # [port_val[-1] / port_val[0]]-1
    # Calculate the Average Daily Return
    adr = daily_ret.mean()  # daily_ret.mean()
    # Calculate the standard deviation of daily return
    sddr = daily_ret.std()  # daily_ret.std()
    # Calculate the sharpe ratio
    sr = np.sqrt(252.0) * np.mean(
    daily_ret - daily_rf) / sddr  # sqrt(252) * mean(daily_ret - daily_rf) / std(daily_ret)
    if gen_plot:
        plot_data(filtered_port_val , title='portvals fill forward', xlabel='dates', ylabel=' port vals ')
    return cr, adr, sddr, sr

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called. 			  		 			 	 	 		 		 	  		   	  			  	
    # Define input parameters 			  		 			 	 	 		 		 	  		   	  			  	

    of = 'C:\Users\cli09\Documents\cs7646_ml4t\WorkingFolder\marketsim\orders\orders-12.csv'
    sv = 1000000 			  		 			 	 	 		 		 	  		   	  			  	

    # Process orders

    orders_df = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])
    portvals = compute_portvals(orders_df=orders_df, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else: 			  		 			 	 	 		 		 	  		   	  			  	
        "warning, code did not return a DataFrame" 			  		 			 	 	 		 		 	  		   	  			  	

    # Get portfolio stats
    SPY_port_val = get_data(['$SPX'], portvals.axes[0])#TODO
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = evaluate_port_val(port_val=portvals, gen_plot=False)

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = evaluate_port_val(port_val=SPY_port_val['$SPX'], gen_plot=False)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date) 			  		 			 	 	 		 		 	  		   	  			  	
    print 			  		 			 	 	 		 		 	  		   	  			  	
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio) 			  		 			 	 	 		 		 	  		   	  			  	
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY) 			  		 			 	 	 		 		 	  		   	  			  	
    print 			  		 			 	 	 		 		 	  		   	  			  	
    print "Cumulative Return of Fund: {}".format(cum_ret) 			  		 			 	 	 		 		 	  		   	  			  	
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY) 			  		 			 	 	 		 		 	  		   	  			  	
    print 			  		 			 	 	 		 		 	  		   	  			  	
    print "Standard Deviation of Fund: {}".format(std_daily_ret) 			  		 			 	 	 		 		 	  		   	  			  	
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY) 			  		 			 	 	 		 		 	  		   	  			  	
    print 			  		 			 	 	 		 		 	  		   	  			  	
    print "Average Daily Return of Fund: {}".format(avg_daily_ret) 			  		 			 	 	 		 		 	  		   	  			  	
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY) 			  		 			 	 	 		 		 	  		   	  			  	
    print 			  		 			 	 	 		 		 	  		   	  			  	
    print "Final Portfolio Value: {}".format(portvals[-1]) 			  		 			 	 	 		 		 	  		   	  			  	

if __name__ == "__main__":
    test_code()
