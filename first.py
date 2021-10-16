import matplotlib.pyplot as plt

from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas_datareader.data as web

def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

def bs_put(S,K,T,r,sigma):
    return K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)

value_start = 9.04

stock = 'SPY'
expiry = '11-05-2021'
strike_price = 437

today = datetime.now()
print(today)
one_year_ago = today.replace(year=today.year-1)

df = web.DataReader(stock, 'yahoo', one_year_ago, today)

df = df.sort_values(by="Date")
df = df.dropna()
df = df.assign(close_day_before=df.Close.shift(1))
df['returns'] = ((df.Close - df.close_day_before)/df.close_day_before)

sigma = np.sqrt(252) * df['returns'].std()
uty = (web.DataReader(
    "^TNX", 'yahoo', today.replace(day=today.day-1), today)['Close'].iloc[-1])/100
lcp = df['Close'].iloc[-1]
t = ((datetime.strptime(expiry, "%m-%d-%Y") - datetime.utcnow()).days + 1) / 365
t2 = (t*365 - 15) / 365
# print(datetime.utcnow())
# print(t*365)

print("lcp: " + str(lcp))
print("strike_price: " + str(strike_price))
print("t: " + str(t))
print("uty: " + str(uty))
print("sigma: " + str(sigma))

print('The Option Price is: ', bs_call(lcp, strike_price, t, uty, sigma))

lcp = 445.5

lcp_var = int(lcp)
print(lcp_var)
lcp_range = 50

lcp_array = [i for i in range(lcp_var - lcp_range, lcp_var + lcp_range)]
current_value_array = [bs_call(i,strike_price,t,uty,sigma) for i in lcp_array]
value_array = [current_value_array[i]-value_start for i in range(len(lcp_array))]

lcp_array2 = [i for i in range(lcp_var - lcp_range, lcp_var + lcp_range)]
current_value_array2 = [bs_call(i,strike_price,t2,uty,sigma) for i in lcp_array2]
value_array2 = [current_value_array2[i]-value_start for i in range(len(lcp_array2))]




# print(lcp_array)
# print(current_value_array)
# print(value_array)


'''
# x axis values
x = [1,2,3]
# corresponding y axis values
y = [2,4,1]
'''

# plotting the points
plt.plot(lcp_array, value_array, color='blue')
plt.plot([425,475], [0,0], ':')
plt.plot(lcp_array2, value_array2, color='red')

# naming the x axis
plt.xlabel('stock price')
# naming the y axis
plt.ylabel('P/L')

# giving a title to my graph
plt.title('P/L graph')

# function to show the plot
plt.show()
