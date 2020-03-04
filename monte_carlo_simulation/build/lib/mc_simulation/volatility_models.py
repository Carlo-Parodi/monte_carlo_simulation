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
#import sklearn
import scipy
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller


# In[2]:


class VolatilityModel:
    
    def __init__(self, average_model, errors_distribution = 'Normal'):
        
        self.test_results = False
        self.errors_distribution = errors_distribution #alternative values: 'ged', 't', 'skewt'
        self.average_model = average_model
        self.deprecated_models_dictionary = []
        self.semaphore = True
        self.first_model = None
        self.const = self.average_model.MeanParamsDict.get('const')
        self.ar1 = self.average_model.MeanParamsDict.get('ar1')
        self.ar2 = self.average_model.MeanParamsDict.get('ar2')
        self.ma1 = self.average_model.MeanParamsDict.get('ma1')
        self.ma2 = self.average_model.MeanParamsDict.get('ma2') 
        self.return_ = self.average_model.return_
        
       #parametri di interesse per l'utente
        self.max_variance_aic = np.inf
        self.VolatilityParamsDict = {'omega': 0, 'alpha[1]': 0, 'alpha[2]':
                                     0, 'beta[1]': 0, 'beta[2]': 0}
        self.t_test_values = self.VolatilityParamsDict.copy()
        self.bestVolatilityModel = None
        return
    
    #questo metodo è l'unico della classe che rendo pubblico
    def fit(self):
        residuals = self.__residuals_creation()
        volatility_models = self.__models_estim(residuals)
        result = self.__models_eval(volatility_models)
        self.test_results = self.__t_test(volatility_models, result)
        while self.test_results:
            result = self.__models_eval(volatility_models, self.bestVolatilityModel)
            self.test_results = self.__t_test(volatility_models, result)

        self.__model_constructor(volatility_models)
        return
    
    def __residuals_creation(self):
        eps_barrel = np.zeros(len(self.return_))
        forecast = np.zeros(len(self.return_))
        for i in range (len(self.return_)):
            forecast[i] = self.const + self.ar1*self.return_[i-1] +self.ar2*self.return_[i-2] + self.ma1*eps_barrel[i-1] + self.ma2*eps_barrel[i-2]
            eps_barrel[i] = self.return_[i] - forecast[i]
        return eps_barrel
    
    def __models_estim(self, eps_barrel):
        VarianceModelsDict = {}
        for i in range(1, 3):
            for k in range(3):
                try:
                    am = arch_model(eps_barrel*10, mean = 'zero', p=i, o=0, q=k, rescale = False, dist = self.errors_distribution)
                    res = am.fit(update_freq=5)
                    VarianceModelsDict['GARCH({0},{1})'.format(i,k)] = (res.aic, res.params, res.tvalues)
                except:
                    VarianceModelsDict['GARCH({0},{1})'.format(i,k)] = (self.max_variance_aic, np.zeros(1), np.zeros(1))
        return VarianceModelsDict
    
    def __models_eval(self, VarianceModelsDict, deprecated = None):
        
        self.deprecated_models_dictionary.append(deprecated)
        alert = 0
        
        for model in VarianceModelsDict:
            if float(VarianceModelsDict[model][0]) < self.max_variance_aic:
                if model not in self.deprecated_models_dictionary:
                    self.max_variance_aic = VarianceModelsDict[model][0]
                    self.bestVolatilityModel = model
                    alert = alert + 1
        if alert == 0:
            self.bestVolatilityModel = self.first_model
        return alert
    
    def __t_test(self, VarianceModelsDict, result):
        
        if self.semaphore:
            self.first_model = self.bestVolatilityModel
            self.semaphore = False
        
        if result is not 0:
            cov_matrix = VarianceModelsDict.get(self.bestVolatilityModel)[-1]
            params = VarianceModelsDict.get(self.bestVolatilityModel)[1]

            deg_of_freedom = (len(self.return_)-len(VarianceModelsDict.get(self.bestVolatilityModel)[1])-1)
            t_student = scipy.stats.t(deg_of_freedom)
            k = 0
            marker = False

            for beta in params:
                test_statistic = beta/cov_matrix[k]
                if test_statistic >= t_student.ppf(.95) or test_statistic <= t_student.ppf(.05):
                    print('{0} accepted, test statistic value is: {1}'.format(beta, test_statistic))
                else:
                    print('{0} rejected, test statistic value is: {1}'.format(beta, test_statistic))
                    marker = True
                k = k+1
        else:
            marker = False
        return marker
    
    def __model_constructor(self, VarianceModelsDict):
        for j in self.VolatilityParamsDict:
            try:
                self.VolatilityParamsDict[j] = VarianceModelsDict.get(self.bestVolatilityModel)[1].get(j)
                self.t_test_values[j] = VarianceModelsDict.get(self.bestVolatilityModel)[-1].get(j)
            except:
                self.VolatilityParamsDict[j] = 0
                self.t_test_values[j] = 0
            if self.VolatilityParamsDict[j] is None:
                self.VolatilityParamsDict[j] = 0
                self.t_test_values[j] = 0
        return


# In[ ]:




