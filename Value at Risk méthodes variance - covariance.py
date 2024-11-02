# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:28:48 2024

@author: Yassine Rahali
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fetch historical data for a stock
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
returns = data['Adj Close'].pct_change().dropna()

# Calculate the mean and standard deviation of returns
mean_return = float(np.mean(returns))
std_dev = float(np.std(returns)) 

# Calculate the VaR at 95% confidence level using the Z-score
confidence_level = 0.95
z_score = norm.ppf(1 - confidence_level)
VaR_variance_covariance = mean_return + z_score * std_dev

print(f"Variance-Covariance VaR (95% confidence level): {VaR_variance_covariance:0.2%}")

# Plot the normal distribution and VaR threshold
plt.figure(figsize=(10, 6))
x = np.linspace(mean_return - 3*std_dev, mean_return + 3*std_dev, 1000)
y = norm.pdf(x, mean_return, std_dev)
plt.plot(x, y, label='Normal Distribution')
plt.axvline(VaR_variance_covariance, color='red', linestyle='--', label=f'VaR (95%): {VaR_variance_covariance:.2%}')
plt.fill_between(x, 0, y, where=(x <= VaR_variance_covariance), color='red', alpha=0.5)
plt.title('Normal Distribution of Returns with VaR Threshold')
plt.xlabel('Returns')
plt.ylabel('Probability Density')
plt.legend()
plt.show()