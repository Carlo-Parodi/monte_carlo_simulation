#!/usr/bin/env python
# coding: utf-8

# In[2]:

import arch
from arch import arch_model
import pandas as pd
import numpy as np
from numpy import array
import matplotlib as mt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels
import scipy
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller


# In[1]:


class MonteCarloSimulation:
    def __init__(self,  volatility_model, time_horizon = 100, number_simulations = 50):
        self.time_horizon = time_horizon
        self.number_simulations = number_simulations
        self.volatility_model = volatility_model
        
        self.average_model = self.volatility_model.average_model
        self.const = self.average_model.MeanParamsDict.get('const')
        self.ar1 = self.average_model.MeanParamsDict.get('ar1')
        self.ar2 = self.average_model.MeanParamsDict.get('ar2')
        self.ma1 = self.average_model.MeanParamsDict.get('ma1')
        self.ma2 = self.average_model.MeanParamsDict.get('ma2') 
        self.return_ = self.average_model.return_
        self.y = self.average_model.y
        self.new_serie = self.average_model.new_serie
        self.price_list = np.zeros(1)
        
        self.omega = self.volatility_model.VolatilityParamsDict.get('omega')
        self.alpha1 = self.volatility_model.VolatilityParamsDict.get('alpha[1]')
        self.alpha2 = self.volatility_model.VolatilityParamsDict.get('alpha[2]')
        self.beta1 = self.volatility_model.VolatilityParamsDict.get('beta[1]')
        self.beta2 = self.volatility_model.VolatilityParamsDict.get('beta[2]')
    
    def get_simulation(self):
        price_list = np.zeros((self.time_horizon, self.number_simulations), dtype=float)
        new_ret= np.zeros(self.time_horizon)

        new_ret[0] = self.return_[-2]
        new_ret[1] = self.return_[-1]
        price_list[0] = self.y[-1]
        eps_lag1 = 0
        eps_lag2 = 0
        cond_vol_lag1 = 0
        cond_vol_lag2 = 0
        somma = 0
        for colonne in range(int(np.shape(price_list)[1])):
            for righe in range(1, int(np.shape(price_list)[0])):

                cond_volatility = self.omega + self.alpha1*(eps_lag1**2) + self.alpha2*(eps_lag2**2) + self.beta1*cond_vol_lag1 + self.beta2*cond_vol_lag2
                expect = self.const + self.ar1*new_ret[righe-1] + self.ar2*new_ret[righe-2] + self.ma1*eps_lag1 + self.ma2*eps_lag2
                if cond_volatility:
                    new_eps = np.random.normal(scale = 1)*cond_volatility
                else:
                    new_eps = np.random.normal(scale = 0.01) 
                new_ret[righe] = expect + new_eps

                price_list[righe][colonne] = price_list[righe-1][colonne]*(1+float(new_ret[righe]))

                cond_vol_lag1 = cond_volatility
                cond_vol_lag2 = cond_vol_lag1
                eps_lag2 = eps_lag1
                eps_lag1 = new_eps
                
        self.price_list = pd.DataFrame(price_list)
        self.__plotter(price_list)
    
    def get_average_scenario(self):
        try: 
            average_scenario = np.zeros((len(self.price_list)), dtype = float)

            for i in range(len(self.price_list)):
                obs = np.average(self.price_list.iloc[i,:])
                average_scenario[i] = obs

            self.__plotter(average_scenario)
            print('average scenario final observation: {0}'.format(average_scenario[-1]))
        except AttributeError as e:
            print('You should first use ''get_simulation()'' method in order to use this method')

    def get_extreme_scenarios(self):
        try:
            ranking_dict = {}
            ext_scenario = np.zeros((len(self.price_list), 3), dtype = float)

            for i in range(len(self.price_list)):
                obs = np.average(self.price_list.iloc[i,:])
                ext_scenario[i,0] = obs

            for k in range (self.price_list.shape[1]):
                ranking_dict[k] = np.average(self.price_list[k])

            ranking_dict.items()
            worst_scenario = min(ranking_dict.items(), key=lambda x: x[1])
            best_scenario = max(ranking_dict.items(), key=lambda x: x[1])

            ext_scenario[:,1] = self.price_list[best_scenario[0]]
            ext_scenario[:,2] = self.price_list[worst_scenario[0]]

            self.__plotter(ext_scenario)
            print('worst scenario final observation is: {0}'.format(ext_scenario[-1,2]))
            print('best scenario final observation is: {0}'.format(ext_scenario[-1,1]))
        except AttributeError as e:
            print('You should first use ''get_simulation()'' method in order to use this method')

    def get_confidence_scenarios(self, conf_interval):
        try: 
            ranking_dict = {}
            int_scenario = np.zeros((len(self.price_list), 3), dtype = float)
            scenario_num = int(conf_interval*self.price_list.shape[1])

            for i in range(len(self.price_list)):
                obs = np.average(self.price_list.iloc[i,:])
                int_scenario[i,0] = obs

            for k in range (self.price_list.shape[1]):
                ranking_dict[k] = np.average(self.price_list[k])

            for i in range (self.price_list.shape[1]-scenario_num):
                worst = min(ranking_dict.items(), key=lambda x: x[1])
                best = max(ranking_dict.items(), key=lambda x: x[1])
                del ranking_dict[worst[0]]
                del ranking_dict[best[0]]

            best_conf_int_scenario = max(ranking_dict.items(), key=lambda x: x[1])
            worst_conf_int_scenario = min(ranking_dict.items(), key=lambda x: x[1])

            int_scenario[:,1] = self.price_list[best_conf_int_scenario[0]]
            int_scenario[:,2] = self.price_list[worst_conf_int_scenario[0]]
            self.__plotter(int_scenario)
            print('worst scenario final observation is: {0}. confidence interval: {1}'.format(int_scenario[-1,2], conf_interval))
            print('best scenario final observation is: {0}. confidence interval: {1}'.format(int_scenario[-1,1], conf_interval))
        except IndexError as e:
            print('You should first use ''get_simulation()'' method in order to use this method')
            
    def __plotter(self, scenarios):

        scenarios = pd.DataFrame(scenarios)
        frames = [pd.DataFrame(self.y), scenarios]
        monte_carlo_forecast = pd.concat(frames, sort = True)

        monte_carlo = monte_carlo_forecast.iloc[:,:].values
        monte_carlo = monte_carlo.astype(float)
        plt.style.use('dark_background')
        plt.figure(figsize=(28,14))
        plt.plot(monte_carlo)
        plt.show()

# In[ ]:




