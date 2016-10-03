import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import statsmodels.api as sm
import scipy.stats as scs


def read_dow_data():
    dow = pdr.DataReader('^DJI', data_source='yahoo',
                         start='30-08-2008', end='30-08-2016')
    dow.rename(columns={'Adj Close': 'index'}, inplace=True)
    dow['returns'] = np.log(dow['index'] / dow['index'].shift(1))
    dow['rea_var'] = 252 * np.cumsum(dow['returns'] ** 2) / np.arange(len(dow))
    dow['rea_vol'] = np.sqrt(dow['rea_var'])
    dow = dow.dropna()
    return dow


def quotes_return(data):
    plt.figure(figsize=(9, 6))
    plt.subplot(211)
    data['index'].plot()
    plt.ylabel('daily quotes')
    plt.grid(True)
    plt.axis('tight')

    plt.subplot(212)
    data['returns'].plot()
    plt.ylabel('daily log returns')
    plt.grid(True)
    plt.axis('tight')
    plt.show()


def pdf_normal(x, mu, sigma):
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
    return pdf


def return_histogram(data):
    plt.figure(figsize=(9, 6))
    x = np.linspace(min(data['returns']), max(data['returns']), 100)
    plt.hist(np.array(data['returns']), bins=50, normed=True)
    y = pdf_normal(x, np.mean(data['returns']), np.std(data['returns']))
    plt.plot(x, y, linewidth=2)
    plt.xlabel('log returns')
    plt.ylabel('frequency/probability')
    plt.grid(True)
    plt.show()


def return_qqplot(data):
    plt.figure(figsize=(9, 5))
    sm.qqplot(data['returns'], line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.show()


def print_stats(data):
    print('Mean Daily Log Return', np.mean(data['returns']))
    print('Std of Daily Log Returns', np.std(data['returns']))
    print('Mean of Annual Log Returns', np.mean(data['returns']) * 252)
    print('Std of Annual Log Returns', np.std(data['returns']) * math.sqrt(252))
    print('-----------------------------------------------')
    print('Skew of Sample Log Returns', scs.skew(data['returns']))
    print('Skew Normal Test p-value', scs.skewtest(data['returns'])[1])
    print('Kurtosis of Sample Log Returns', scs.kurtosis(data['returns']))
    print('Kurtosis Normal Test p-value', scs.kurtosistest(data['returns'])[1])
    print('-----------------------------------------------')
    print('Normal Test p-value', scs.normaltest(data['returns'])[1])
    print('-----------------------------------------------')
    print('Realized Volatility', data['rea_vol'].iloc[-1])
    print('Realized Variance', data['rea_var'].iloc[-1])


def realized_volatility(data):
    plt.figure(figsize=(9, 6))
    data['rea_vol'].plot()
    plt.ylabel('realized volatility')
    plt.grid(True)
    plt.show()


def rolling_statistics(data):
    plt.figure(figsize=(11, 8))

    plt.subplot(311)
    mean_return = pd.rolling_mean(data['returns'], 252) * 252
    mean_return.plot()
    plt.grid(True)
    plt.ylabel('returns (252d)')
    plt.axhline(mean_return.mean(), color='r', ls='dashed', lw=1.5)

    plt.subplot(312)
    vol = pd.rolling_std(data['returns'], 252) * math.sqrt(252)
    vol.plot()
    plt.grid(True)
    plt.ylabel('voltaility (252d)')
    plt.axhline(vol.mean(), color='r', ls='dashed', lw=1.5)
    vx = plt.axis()

    plt.subplot(313)
    corr = pd.rolling_corr(mean_return, vol, 252)
    corr.plot()
    plt.grid(True)
    plt.ylabel('correlation (252d)')
    cx = plt.axis()
    plt.axis([vx[0], vx[1], cx[2], cx[3]])
    plt.axhline(corr.mean(), color='r', ls='dashed', lw=1.5)

    plt.show()


if __name__ == '__main__':

    dow = read_dow_data()
    quotes_return(dow)
    return_histogram(dow)
    return_qqplot(dow)
    print_stats(dow)
    realized_volatility(dow)
    rolling_statistics(dow)
