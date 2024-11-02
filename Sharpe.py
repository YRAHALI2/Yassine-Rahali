# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:29:07 2024

@author: Yassine Rahali
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour obtenir les données historiques
def get_donnees_historiques(codes, date_depart):
    # On récupère les données de Yahoo Finance pour plusieurs actions
    data = yf.download(codes, start=date_depart)['Adj Close']
    # Calcul des variations quotidiennes (rendements)
    data = data.pct_change().dropna()
    return data

# Calcul de l'écart type
def std_dev(data):
    n = len(data)
    moyenne = sum(data) / n
    ecart = sum([(x - moyenne) ** 2 for x in data])
    variance = ecart / (n - 1)
    return variance ** (1/2)

# Calcul du ratio de Sharpe
def ratio_sharpe(data, taux_sans_risque=0.0):
    # Calcul de la variation moyenne par jour
    variation_moyenne_par_jour = data.mean()
    # Calcul de l'écart type des rendements
    s = data.std()
    # Ratio de Sharpe quotidien
    ratio_sharpe_quotidien = (variation_moyenne_par_jour - taux_sans_risque) / s
    # On annualise le ratio de Sharpe (on prend 252 jours de cotation par an)
    ratio_sharpe_annualise = (252 ** (1/2)) * ratio_sharpe_quotidien
    return ratio_sharpe_annualise

# Fonction pour calculer le ratio de Sharpe du portefeuille
def ratio_sharpe_portefeuille(data, poids, taux_sans_risque=0.0):
    # Calcul du rendement moyen du portefeuille
    rendement_portefeuille = np.dot(data.mean(), poids)
    # Calcul de la covariance des rendements
    covariance = np.dot(poids.T, np.dot(data.cov() * 252, poids))
    volatilite_portefeuille = np.sqrt(covariance)
    # Calcul du ratio de Sharpe
    ratio_sharpe_portefeuille = (rendement_portefeuille - taux_sans_risque) / volatilite_portefeuille
    return ratio_sharpe_portefeuille

# Liste des tickers des actions
tickers = ['AMZN', 'NFLX', 'AAPL', 'GOOGL']

# Récupération des données historiques depuis le 1er janvier 2023
data = get_donnees_historiques(tickers, '2023-01-01')

# Affichage des premiers rendements
print(data.head())

# On suppose des poids égaux pour chaque action dans le portefeuille
poids_egaux = np.array([0.25, 0.25, 0.25, 0.25])

# Calcul et affichage du ratio de Sharpe pour chaque action
for ticker in tickers:
    sharpe = ratio_sharpe(data[ticker])
    print(f"Ratio de Sharpe pour {ticker}: {sharpe:.2f}")

# Calcul du ratio de Sharpe pour le portefeuille combiné
sharpe_portefeuille = ratio_sharpe_portefeuille(data, poids_egaux)
print(f"\nRatio de Sharpe pour le portefeuille (poids égaux): {sharpe_portefeuille:.2f}")

# Affichage des rendements cumulés du portefeuille et des actions
rendements_cumules = (1 + data).cumprod()

plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.plot(rendements_cumules.index, rendements_cumules[ticker], label=ticker)

plt.title('Rendements cumulés des actions')
plt.xlabel('Date')
plt.ylabel('Rendement cumulé')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
