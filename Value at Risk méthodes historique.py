# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:30:49 2024

@author: Yassine Rahali
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical data for a stock
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
returns = data['Adj Close'].pct_change().dropna()

# Calculate the historical VaR at 95% confidence level
confidence_level = 0.95
VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)

print(f"Historical VaR (95% confidence level): {VaR_historical:.2%}")

# Plot the historical returns and VaR threshold
plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.axvline(VaR_historical, color='red', linestyle='--', label=f'VaR (95%): {VaR_historical:.2%}')
plt.title('Historical Returns of AAPL')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()