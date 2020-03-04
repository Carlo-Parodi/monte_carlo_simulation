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


# In[2]:


#questa classe mi fornisce dei dati puliti (in csv), devo rifarla per ottenere i dati sia da file csv,
#sia formato xls, sia xlsx, e una stringa che mi indichi quale colonna considerare per i price

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


#classe per la creazione del modello di media condizionata

class AverageModel:
    def __init__(self, link, file_type = 'csv', column_name = 'Close'):
        
        self.link = link
        self.file_type = file_type
        self.column_name = column_name
        self.return_ = np.zeros(1)
        self.max_mean_aic = np.inf
        self.deprecated_models_dictionary = []
        self.y = np.zeros(1)
        self.new_serie = np.zeros(1)
        
        #parametri di interesse per l'utente
        self.t_test_values = {'const':0.0, 'ar1':0.0, 'ar2':0.0, 'ma1':0.0, 'ma2':0.0}
        self.MeanParamsDict = self.t_test_values.copy()
        self.bestModel = None
        self.adf_result = None

        
    #questo metodo è l'unico della classe che rendo pubblico
    
    def fit(self):
        self.new_serie = DataCleaner(self.link, self.file_type, self.column_name)
        self.y = self.new_serie.get_serie()
        self.return_ = np.zeros(len(self.y))

        for i in range(0,len(self.y)):
            self.return_[i] = (self.y[i] - self.y[i-1]) / self.y[i-1]
        ret = pd.DataFrame(self.return_)   
        data1 = ret.iloc[:,0].values
        
        adf_result = self.__adf_test(data1, maxlag=None, regression='c', autolag='AIC')
        if adf_result[0]>-3.44:
            raise UnitRootError('ADF test statistic value: ' + str(adftest_result[0]) + 
                         ', critical value: -3.44 circa\n cannot proceed with evaluating any model')
        else:
            model_dictionary = self.__models_estim()
            self.__models_eval(model_dictionary)
            test_results = self.__t_test(model_dictionary)
            
            while test_results:
                self.__models_eval(test_results, self.bestModel)
                test_results = self.__t_test(model_dictionary)
            
            self.__model_constructor(model_dictionary)
            
    def __adf_test(self, data1, maxlag=None, regression='c', autolag='AIC'):
        adftest_result = adfuller(data1, maxlag=None, regression='c', autolag='AIC')
        return adftest_result
    
    def __models_estim(self):
        MeanModelsDict = {}
        bestModel = None

        for i in range(3):
            for k in range(3):
                try:
                    model = ARIMA(self.return_, order=(i,0,k))
                    model_fit = model.fit(trend='c')
                    print('ARMA({0},{1}): '.format(i,k) + str(model_fit.params) + ', t-statistic values: ' + str(model_fit.tvalues) + str(model_fit.pvalues))
                    MeanModelsDict['ARMA({0},{1})'.format(i,k)] = (model_fit.aic, model_fit.params, model_fit.tvalues, model_fit.cov_params())
                except:
                    MeanModelsDict['ARMA({0},{1})'.format(i,k)] = (self.max_mean_aic, np.zeros(1),
                                                                   np.zeros(1), np.zeros(1))
        return MeanModelsDict
    
    def __models_eval(self, MeanModelsDict, deprecated = None):
        
        self.deprecated_models_dictionary.append(deprecated)
        
        for j in MeanModelsDict:
            if float(MeanModelsDict[j][0]) < self.max_mean_aic:
                if j not in self.deprecated_models_dictionary:
                    self.max_mean_aic = MeanModelsDict[j][0]
                    self.bestModel = j
            return
    
    def __t_test(self, MeanModelsDict):
        cov_matrix = MeanModelsDict.get(self.bestModel)[-1]
        deg_of_freedom = (len(self.return_)-len(MeanModelsDict.get(self.bestModel)[1])-1)
        t_student = scipy.stats.t(deg_of_freedom)
        cov_matrix = MeanModelsDict.get(self.bestModel)[-1]
        k = 0
        marker = False

        for beta in MeanModelsDict.get(self.bestModel)[1]:
            point = cov_matrix[k][k]
            test_statistic = beta/point
            if test_statistic >= t_student.ppf(.95) or test_statistic <= t_student.ppf(.05):
                print('{0} accepted, test statistic value is: {1}'.format(beta, test_statistic))
            else:
                print('{0} rejected, test statistic value is: {1}'.format(beta, test_statistic))
                marker = True
                return marker
            k = k+1
        return marker
    
    def __model_constructor(self, MeanModelsDict):
        i = 0
        for j in self.MeanParamsDict:
            try:
                self.MeanParamsDict[j] = MeanModelsDict.get(self.bestModel)[1][i]
                self.t_test_values[j] = MeanModelsDict.get(self.bestModel)[2][i]
            except:
                self.MeanParamsDict[j] = 0
                self.t_test_values[j] = 0
            i = i+1


# In[ ]:




