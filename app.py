from numpy._core.defchararray import index

import yfinance as yf # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from math import *

tickers = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMD', 'TSM', 'PLTR']

prices = yf.download(tickers, start='2022-08-25', end='2026-08-24')['Close']

returns = np.log(prices/prices.shift(1)).dropna()
portfolio_weights_list = []
portfolio_variances_list = []

returns_mean = np.mean(returns, axis=0)
centered_returns = returns - returns_mean

covariance_matrix = (centered_returns.T @ centered_returns)/(len(returns)-1)
ones_vector = np.ones(len(tickers))

mean_ones_matrix = np.c_[returns_mean, ones_vector]

solved_covariance_matrix = np.linalg.solve(covariance_matrix, mean_ones_matrix)

mu_T_sigma_inv_mu = returns_mean.T @ solved_covariance_matrix[:, 0] # A
mu_T_sigma_inv_ones = returns_mean.T @ solved_covariance_matrix[:, 1] # B
ones_T_sigma_inv_ones = ones_vector.T @ solved_covariance_matrix[:, 1] # C

print('A: ', mu_T_sigma_inv_mu) # type: ignore
print('B: ', mu_T_sigma_inv_ones)
print('C: ', ones_T_sigma_inv_ones)

print('--------------------------------')
print('--------------------------------')

def weight_r(expected_return):

    lagrange_mult_matrix = np.array([[mu_T_sigma_inv_mu, mu_T_sigma_inv_ones], [mu_T_sigma_inv_ones, ones_T_sigma_inv_ones]])   

    '''
    print('Lagrange Multipliers Matrix: ')
    print(lagrange_mult_matrix)

    print('--------------------------------')
    print('--------------------------------')    
    '''

    return_constraint = np.linalg.solve(lagrange_mult_matrix, np.array([expected_return, 1]))[0] # Lambda
    budget_constraint = np.linalg.solve(lagrange_mult_matrix, np.array([expected_return, 1]))[1] # Gamma
    
    '''
    print('Return Constraint: ', return_constraint)
    print('Budget Constraint: ', budget_constraint)

    print('--------------------------------')
    print('--------------------------------')

    print('Experimental value of expected return: ')
    print(return_constraint * mu_T_sigma_inv_mu + budget_constraint * mu_T_sigma_inv_ones)
    print('Experimental value of total budget: ')
    print(return_constraint * mu_T_sigma_inv_ones + budget_constraint * ones_T_sigma_inv_ones)

    print('--------------------------------')
    print('--------------------------------')
    '''
    

    weight = return_constraint * solved_covariance_matrix[:, 0] + budget_constraint * solved_covariance_matrix[:, 1]
    # print(f'Weights of the portfolio for r = {expected_return} : ')

    '''
    for i in range(len(tickers)):
            print(tickers[i], ': ', weight[i])
    
        print(f'Total weight of the portfolio: {np.sum(weight)}')
    
        print(weight @ returns_mean) # Expected return of the portfolio
    '''

    return weight

r_min = mu_T_sigma_inv_ones / ones_T_sigma_inv_ones
r_max = max(returns_mean)*3

r_values = np.linspace(r_min, r_max, 100)

def portfolio_variance(weights):
    return weights.T @ covariance_matrix @ weights

'''
print('Values of return to sweep over: ')
print(r_values)

print(len(r_values))
'''

for i in range(len(r_values)):
    portfolio_weights_list.append(weight_r(r_values[i]))
    portfolio_variances_list.append(portfolio_variance(weight_r(r_values[i])))

# print('Portfolio Variance: ')
# print(portfolio_variances_list)

portfolio_standard_deviations_list = np.sqrt(portfolio_variances_list)

plt.plot(portfolio_standard_deviations_list, r_values)
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.show()

irx = yf.download('^IRX', start='2022-08-25', end='2026-08-24')['Close']
r_f = np.mean((1+ irx/100)**(1/252) - 1)

Sharpe_ratios = (r_values - r_f) / portfolio_standard_deviations_list
print('Sharpe Ratios: ')
print(Sharpe_ratios)

print(portfolio_weights_list[np.argmax(Sharpe_ratios)])