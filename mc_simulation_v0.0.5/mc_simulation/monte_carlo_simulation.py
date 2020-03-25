#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
from contextlib import contextmanager
import sys, os

#this function is used to suppress a special Warning when computing the volatility models
@contextmanager
def suppress_stdout():
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:  
        yield
    finally:
        sys.stdout = old_stdout
# In[2]:
#this is a class employed for internal use only, in order to clean the data in input
class DataCleaner():
    
    def __init__(self, link, file_type, column_name):

        self.link = link
        self.file_type = file_type
        self.column_name = column_name
        
        if file_type == 'csv':
             self.data_preprocessed = pd.read_csv(self.link)
        elif file_type == 'xls':
            self.data_preprocessed = pd.read_excel(self.link)
        
        self.new_columns = list(self.data_preprocessed.iloc[0])
        
        self.data_preprocessed.columns = self.new_columns
        self.data_preprocessed = self.data_preprocessed.drop([0])
        self.y = np.zeros(len(self.data_preprocessed))
        
        for i in range (0, len(self.data_preprocessed)):
                self.y[i] = self.data_preprocessed.iloc[i][self.column_name]
                
    def get_data(self):
        return pd.DataFrame(self.data_preprocessed)
    
    def get_serie(self):
        return self.y


# In[3]:
#this is a class employed in the computation of the best-fitting average model
class AverageModel:
    def __init__(self, link, file_type = 'csv', column_name = 'Close', p_value = .95):
        
        self.p_value = p_value
        self.link = link
        self.file_type = file_type
        self.column_name = column_name
        self.return_ = np.zeros(1)
        self.max_mean_aic = np.inf
        self.alert = True
        self.__deprecated_models_dictionary = []
        self.firstModel = None
        self.y = np.zeros(1)
        self.new_serie = np.zeros(1)
        
        self.t_test_values = {'const':0.0, 'ar1':0.0, 'ar2':0.0, 'ma1':0.0, 'ma2':0.0}
        self.MeanParamsDict = self.t_test_values.copy()
        self.MeanModelsDict = {}
        self.bestModel = None
        self.adf_result = None
    
    def fit(self):
        self.new_serie = DataCleaner(self.link, self.file_type, self.column_name)
        self.y = self.new_serie.get_serie()
        self.return_ = np.zeros(len(self.y))

        for i in range(0,len(self.y)):
            self.return_[i] = (self.y[i] - self.y[i-1]) / self.y[i-1]
        ret = pd.DataFrame(self.return_)   
        data1 = ret.iloc[:,0].values
        adf_result = self.__adf_test(data1)
        if adf_result[0]>-3.44:
            raise UnitRootError('ADF test statistic value: ' + str(adftest_result[0]) + 
                         ', critical value: -3.44 circa\n cannot proceed with the evaluation of any model')
        else:
            self.__models_estim()
            self.__models_eval()
            test_results = self.__t_test()
            
            while test_results:
                self.__models_eval()
                test_results = self.__t_test()
            
            self.__model_constructor()
            
    def __adf_test(self, data1, maxlag=None, regression='c', autolag='AIC'):
        adftest_result = adfuller(data1, maxlag=None, regression='c', autolag='AIC')
        return adftest_result
    
    def __models_estim(self):
        bestModel = None

        for i in range(3):
            for k in range(3):
                try:
                    model = ARIMA(self.return_, order=(i,0,k))
                    model_fit = model.fit(trend='c')
                    self.MeanModelsDict['ARMA({0},{1})'.format(i,k)] = (model_fit.aic, model_fit.params,
                                                                        model_fit.tvalues, 
                                                                        model_fit.cov_params())
                except:
                    self.MeanModelsDict['ARMA({0},{1})'.format(i,k)] = (self.max_mean_aic, np.zeros(1),
                                                                   np.zeros(1), np.zeros(1))
        return
    
    def __models_eval(self):
        
        for j in self.MeanModelsDict:
            if float(self.MeanModelsDict[j][0]) < self.max_mean_aic:
                if j not in self.__deprecated_models_dictionary:
                    self.max_mean_aic = self.MeanModelsDict[j][0]
                    self.bestModel = j
                    
        if self.alert:
            self.firstModel = self.bestModel
            self.alert = False
        return
    
    def __t_test(self):
        marker = True

        if self.bestModel in self.__deprecated_models_dictionary:
            self.bestModel = self.firstModel
            marker = False
            return marker
        else:
            cov_matrix = self.MeanModelsDict.get(self.bestModel)[-1]
            deg_of_freedom = (len(self.return_)-len(self.MeanModelsDict.get(self.bestModel)[1])-1)
            t_student = scipy.stats.t(deg_of_freedom)
            cov_matrix = self.MeanModelsDict.get(self.bestModel)[-1]
            k = 0
            
            for beta in self.MeanModelsDict.get(self.bestModel)[1]:
                point = cov_matrix[k][k]
                test_statistic = beta/point
                if test_statistic < t_student.ppf(self.p_value) or test_statistic > t_student.ppf(1-
                                                  self.p_value):
                    self.__deprecated_models_dictionary.append(self.bestModel)
                    self.max_mean_aic = np.inf
                    marker = True
                    return marker
                k = k+1
            marker = False
        return marker
        
    def __model_constructor(self):
        i = 0
        for j in self.MeanParamsDict:
            try:
                self.MeanParamsDict[j] = self.MeanModelsDict.get(self.bestModel)[1][i]
                self.t_test_values[j] = self.MeanModelsDict.get(self.bestModel)[2][i]
            except:
                self.MeanParamsDict[j] = 0
                self.t_test_values[j] = 0
            i = i+1


# In[4]:
#this is a class employed for finding the best fitting volatility model
class VolatilityModel:
    
    def __init__(self, average_model, errors_distribution = 'Normal', p_value = .95):
        
        self.p_value = p_value
        self.test_results = False
        self.errors_distribution = errors_distribution #alternative values: 'ged', 't', 'skewt'
        self.average_model = average_model
        self.__deprecated_models_dictionary = []
        self.alert = True
        self.firstModel = None
        self.const = self.average_model.MeanParamsDict.get('const')
        self.ar1 = self.average_model.MeanParamsDict.get('ar1')
        self.ar2 = self.average_model.MeanParamsDict.get('ar2')
        self.ma1 = self.average_model.MeanParamsDict.get('ma1')
        self.ma2 = self.average_model.MeanParamsDict.get('ma2') 
        self.return_ = self.average_model.return_
        
        self.max_variance_aic = np.inf
        self.VolatilityParamsDict = {'omega': 0, 'alpha[1]': 0, 'alpha[2]':
                                     0, 'beta[1]': 0, 'beta[2]': 0}
        self.t_test_values = self.VolatilityParamsDict.copy()
        self.bestVolatilityModel = None
        self.VarianceModelsDict = {}
        return

    def fit(self):
        residuals = self.__residuals_creation()
        self.__models_estim(residuals)
        self.__models_eval()
        self.test_results = self.__t_test()
        while self.test_results:
            self.__models_eval()
            self.test_results = self.__t_test()

        self.__model_constructor()
        return
    
    def __residuals_creation(self):
        eps_barrel = np.zeros(len(self.return_))
        forecast = np.zeros(len(self.return_))
        for i in range (len(self.return_)):
            forecast[i] = self.const + self.ar1*self.return_[i-1] +self.ar2*self.return_[i-2] + self.ma1*eps_barrel[i-1] + self.ma2*eps_barrel[i-2]
            eps_barrel[i] = self.return_[i] - forecast[i]
        return eps_barrel
    
    def __models_estim(self, eps_barrel):
        
        for i in range(3):
            for k in range(3):
                try:
                    with suppress_stdout():
                        am = arch_model(eps_barrel*10, mean = 'zero', p=i, o=0, q=k, rescale = False, dist = 
                                        self.errors_distribution)
                        res = am.fit(update_freq=5)
                    self.VarianceModelsDict['GARCH({0},{1})'.format(i,k)] = (res.aic, res.params, res.tvalues)
                except:
                    self.VarianceModelsDict['GARCH({0},{1})'.format(i,k)] = (self.max_variance_aic, 
                                                                             np.zeros(1), np.zeros(1))
        return
    
    def __models_eval(self):
        
        for model in self.VarianceModelsDict:
            if float(self.VarianceModelsDict[model][0]) < self.max_variance_aic:
                if model not in self.__deprecated_models_dictionary:
                    self.max_variance_aic = self.VarianceModelsDict[model][0]
                    self.bestVolatilityModel = model
        if self.alert:
            self.firstModel = self.bestVolatilityModel
            self.alert = False
        return
    
    def __t_test(self):
        marker = True
        
        if self.bestVolatilityModel in self.__deprecated_models_dictionary:
            self.bestVolatilityModel = self.firstModel
            marker = False
            return marker
        else:
            cov_matrix = self.VarianceModelsDict.get(self.bestVolatilityModel)[-1]
            params = self.VarianceModelsDict.get(self.bestVolatilityModel)[1]
            deg_of_freedom = (len(self.return_)-len(self.VarianceModelsDict.get(self.bestVolatilityModel)
                                                    [1])-1)
            t_student = scipy.stats.t(deg_of_freedom)
            k = 0
            
            for beta in params:
                test_statistic = beta/cov_matrix[k]
                if test_statistic < t_student.ppf(self.p_value) or test_statistic > t_student.ppf(1-
                                                  self.p_value):
                    self.__deprecated_models_dictionary.append(self.bestVolatilityModel)
                    self.max_variance_aic = np.inf
                    marker = True
                    return marker
                k = k+1
            marker = False
        return marker
    
    def __model_constructor(self):
        for j in self.VolatilityParamsDict:
            try:
                self.VolatilityParamsDict[j] = self.VarianceModelsDict.get(self.bestVolatilityModel)[1].get(j)
                self.t_test_values[j] = self.VarianceModelsDict.get(self.bestVolatilityModel)[-1].get(j)
            except:
                self.VolatilityParamsDict[j] = 0
                self.t_test_values[j] = 0
            if self.VolatilityParamsDict[j] is None:
                self.VolatilityParamsDict[j] = 0
                self.t_test_values[j] = 0
        return

# In[5]:
#class for Monte Carlo Simulations
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
        
        self.semaphore1 = True
        self.semaphore2 = True
        self.ranking_dict1 = {}
        self.ranking_dict2 = {}
        self.lim_scenario1 = np.zeros(1)
        self.lim_scenario2 = np.zeros(1)
        
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
            self.__scenarios_ranker('returns_average')
            average_scenario = self.lim_scenario1[:,0]
            self.__plotter(average_scenario)
            print('average scenario final observation: {0}'.format(average_scenario[-1]))
        except AttributeError as e:
            print('You should first use ''get_simulation()'' method in order to use this method')

    def get_extreme_scenarios(self, criterion = 'returns_average'):
        try:
            lim_scenario = np.zeros((len(self.price_list), 3), dtype = float)
            self.__scenarios_ranker(criterion)
            if criterion == 'returns_average':
                ranking_copy = self.ranking_dict1.copy()
                lim_scenario[:,0] = self.lim_scenario1[:,0]
            else:
                ranking_copy = self.ranking_dict2.copy()
                lim_scenario[:,0] = self.lim_scenario2[:,0]
                
            worst_scenario = min(ranking_copy.items(), key=lambda x: x[1])
            best_scenario = max(ranking_copy.items(), key=lambda x: x[1])

            lim_scenario[:,1] = self.price_list[best_scenario[0]]
            lim_scenario[:,2] = self.price_list[worst_scenario[0]]

            self.__plotter(lim_scenario)
            print('worst scenario final observation is: {0}'.format(lim_scenario[-1,2]))
            print('best scenario final observation is: {0}'.format(lim_scenario[-1,1]))
        except AttributeError as e:
            print('You should first use ''get_simulation()'' method in order to use this method')

    def get_confidence_scenarios(self, conf_interval, criterion = 'returns_average'):
        try:
            lim_scenario = np.zeros((len(self.price_list), 3), dtype = float)
            self.__scenarios_ranker(criterion)
            
            if criterion == 'returns_average':
                ranking_copy = self.ranking_dict1.copy()
                lim_scenario[:,0] = self.lim_scenario1[:,0]
            else:
                ranking_copy = self.ranking_dict2.copy()
                lim_scenario[:,0] = self.lim_scenario2[:,0]
                
            scenario_num = int(conf_interval*self.price_list.shape[1])

            for i in range (self.price_list.shape[1]-scenario_num):
                worst = min(ranking_copy.items(), key=lambda x: x[1])
                best = max(ranking_copy.items(), key=lambda x: x[1])
                del ranking_copy[worst[0]]
                del ranking_copy[best[0]]

            best_conf_int_scenario = max(ranking_copy.items(), key=lambda x: x[1])
            worst_conf_int_scenario = min(ranking_copy.items(), key=lambda x: x[1])

            lim_scenario[:,1] = self.price_list[best_conf_int_scenario[0]]
            lim_scenario[:,2] = self.price_list[worst_conf_int_scenario[0]]
            self.__plotter(lim_scenario)
            print('worst scenario final observation is: {0}. confidence interval: {1}'.format(lim_scenario[-1,2], conf_interval))
            print('best scenario final observation is: {0}. confidence interval: {1}'.format(lim_scenario[-1,1], conf_interval))
        except IndexError as e:
            print('You should first use ''get_simulation()'' method in order to use this method')
    
    def __scenarios_ranker(self, criterion):
        if criterion == 'returns_average':
            if self.semaphore1:
                self.lim_scenario1 = np.zeros((len(self.price_list), 3), dtype = float)
                for i in range(len(self.price_list)):
                        obs = np.average(self.price_list.iloc[i,:])
                        self.lim_scenario1[i,0] = obs

                for k in range (self.price_list.shape[1]):
                        self.ranking_dict1[k] = np.average(self.price_list[k])
                self.semaphore1 = False
            return
        elif criterion == 'last_obs':
            if self.semaphore2:
                self.lim_scenario2 = np.zeros((len(self.price_list), 3), dtype = float)
                for i in range(len(self.price_list)):
                        obs = np.average(self.price_list.iloc[i,:])
                        self.lim_scenario2[i,0] = obs

                for k in range (self.price_list.shape[1]):
                        self.ranking_dict2[k] = self.price_list.iloc[-1, k]
                self.semaphore2 = False
            return
        else:
            raise Error('incorrect input: please edit ''returns_average'' or ''last_obs'' as ranking criterion')

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






