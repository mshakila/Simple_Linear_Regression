# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:51:08 2019

@author: Admin
"""
###### SIMPLE LINEAR REGRESSION
# Q4) predcit delivery time using sorting time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading a csv file using pandas library
delivery= pd.read_csv("file:///E:/EXCELR/Assignments/Simple_Linear_Regression_Assignment/delivery_time.csv")
type(delivery)

delivery.columns
delivery.columns= ['deliverytime','sortingtime']
# 'Delivery Time', 'Sorting Time'
delivery.head()

# large dataset or not
from scipy.stats import kurtosis, skew
10 * kurtosis(delivery.deliverytime)

# I business moments
np.mean(delivery)
# deliverytime    16.790952
# sortingtime      6.190476
np.median(delivery.deliverytime) # 17.83
np.median(delivery.sortingtime) #6.0
import statistics
statistics.mode(delivery.deliverytime) # no mode
statistics.mode(delivery.sortingtime) # 7

np.var(delivery.deliverytime) # 24.528208616780045
np.var(delivery.sortingtime) # 6.154195011337867
np.std(delivery.deliverytime) # 4.95259614917066
np.std(delivery.sortingtime) # 2.4807650052630676

np.max((delivery.deliverytime)) - np.min(delivery.deliverytime) # 29 -8 =21
np.max(delivery.sortingtime) - np.min(delivery.sortingtime) # 10-2=8

from scipy.stats import skew, kurtosis
kurtosis(delivery) #array([-0.02558577, -1.16539014])
skew(delivery) # array([-0.02558577, -1.16539014])
# visualizations
plt.hist(delivery.deliverytime);plt.title('Histogram of delivery time')

plt.hist(delivery.sortingtime); plt.title("Histogram of sorting time")
plt.boxplot(delivery.deliverytime,0,'rs',0);plt.title('Boxplot of delivery time')
plt.boxplot(delivery.sortingtime,0,'rs',0);plt.title('Boxplot of sorting time')

# check normality
from statsmodels.graphics.gofplots import qqplot
qqplot()
qqplot(delivery.deliverytime,line='s');plt.title('QQ plot of delivery time')
qqplot(delivery.sortingtime,line='s');plt.title('QQ plot of sorting time')
from scipy.stats import shapiro ,anderson, normaltest
shapiro(delivery.deliverytime) # (0.9781284928321838, 0.8963273763656616)
shapiro(delivery.sortingtime) #  (0.9367821216583252, 0.1881045252084732)
anderson(delivery.deliverytime)
anderson(delivery.sortingtime)
normaltest(delivery.deliverytime)
# statistic=0.8639105397907696, pvalue=0.6492384165379566
normaltest(delivery.sortingtime)
# statistic=2.7066189560535268, pvalue=0.2583837288180371)

# scatter plot
plt.plot(delivery.sortingtime,delivery.deliverytime,'bo');plt.title('Scatter plot')
plt.xlabel('sorting time');plt.ylabel('delivery time')
np.corrcoef(delivery.sortingtime,delivery.deliverytime) 
delivery.deliverytime.corr(delivery.sortingtime) # 0.82599726

############   model building using Y ~ X  ##########################
import statsmodels.formula.api as smf
reg_simple=smf.ols("deliverytime~ sortingtime", data=delivery).fit()

reg_simple.params
# Intercept    6.582734
# st           1.649020
reg_simple.summary()
# R2 = 0.682, adj R2 = 0.666
# F stats, pvalue=0
# b0, b1 both sig

reg_simple.conf_int(0.05)

# Predicting values of delivery time using the model
pred_simple = reg_simple.predict(delivery.iloc[:,1]) 
# residuals
residuals=pred_simple - delivery.deliverytime

# RMSE is 2.792
np.sqrt(np.mean((pred_simple-delivery.deliverytime)**2))

# plotting regression line
plt.scatter(x=delivery.sortingtime,y=delivery.deliverytime,color='red')
plt.plot(delivery.sortingtime,pred_simple,color='blue')
plt.xlabel('Sorting time');plt.ylabel('Delivery time')
plt.title('Regression line of predicted Y')

# pred vs actuals
plt.plot(pred_simple,delivery.deliverytime,'bo');plt.xlabel('predicted');plt.ylabel('actuals')
# finding corr of predicted vs actual values
delivery.deliverytime.corr(pred_simple) # corre is 0.8259

# residual plots - residual vs fitted values
# The plot is used to detect non-linearity, unequal error variances, and outliers.
plt.plot(pred_simple,residuals,'bo');plt.xlabel('Fitted'),plt.ylabel('Residuals')
plt.title('Residual plot')

# Residual plot - Residuals vs independent variable
plt.scatter(delivery.sortingtime, residuals)
plt.xlabel('sorting time');plt.ylabel("Residuals")
plt.title('Residual plot: Residuals vs independent variable')

 
# normality of residuals
from statsmodels.graphics.gofplots import qqplot
qqplot(residuals); plt.title('Normal QQ plot of Residuals')
from scipy.stats import shapiro, anderson, normaltest
shapiro(residuals) # (0.9376433491706848, 0.19567471742630005)
anderson(residuals)
anderson(delivery.deliverytime)
normaltest(residuals) # (statistic=3.6487821448446733, pvalue=0.16131584389453138)

import seaborn as sns
sns.set(style='whitegrid')
# Plot the residuals after fitting a linear model
# sns.residplot(delivery.deliverytime,pred_simple,lowess=True,color='g')
# plt.ylabel('Residuals');plt.title('Residual plot')

#########################################################################

# LOG TRANSFORMATION
# normality
sns.set(style='white')
# normality
shapiro(np.log(delivery.sortingtime)) # (0.9227311611175537, 0.09841115027666092)
qqplot(delivery.sortingtime); plt.title('Normal qqplot of log(X)')
# correlation
delivery.deliverytime.corr(np.log(delivery.sortingtime)) # corr is 0.8339325279256244

# model
import statsmodels.formula.api as smf
reg_log=smf.ols('deliverytime ~ np.log(sortingtime)', data=delivery).fit()

reg_log.params
reg_log.summary()
# R2 is 0.695, intercept not significant
# B0 is 1.1597 (pvalue 0.642) and B1 is 9.0434
reg_log.conf_int(0.05) # conf intv is 95%
pred_log = reg_log.predict(delivery.iloc[:,1])
print(pred_log)
residuals_log=delivery.deliverytime - pred_log
#RMSE
np.sqrt(np.mean(residuals_log*residuals_log)) # 2.7331
shapiro(residuals_log) # (0.9284946322441101, 0.12840934097766876)

# plotting regression line
plt.scatter(x=np.log(delivery.sortingtime),y=delivery.deliverytime,color='red')
plt.plot(np.log(delivery.sortingtime),pred_log,color='blue')
plt.xlabel('log(Sorting time)');plt.ylabel('Delivery time')
plt.title('Regression line of predicted Y using log(X)')

# residual plot
sns.set(style='whitegrid')
plt.plot(pred_log,residuals_log,'go');plt.xlabel('Fitted'),plt.ylabel('Residuals')
plt.title('Residual plot using log(X)')


###########################################################
# sqrt(X) - very slight difference in values as comapred to log(X)

delivery.deliverytime.corr(np.sqrt(delivery.sortingtime)) # corr 0.83415
shapiro(np.sqrt(delivery.sortingtime)) # normal
# (0.9384300708770752, 0.20284484326839447)
reg_sqrt = smf.ols('deliverytime ~ np.sqrt(sortingtime)',data=delivery).fit()
reg_sqrt.summary()

pred_sqrt = reg_sqrt.predict(delivery.iloc[:,1])
residuals_sqrt=delivery.deliverytime - pred_sqrt

#RMSE
np.sqrt(np.mean(residuals_sqrt*residuals_sqrt)) #  RMSE is 2.7315

#corr of actual and predicted
delivery.deliverytime.corr(pred_sqrt) # 0.83415
#normality
shapiro(residuals_sqrt)
# (0.9264540076255798, 0.1168602854013443), residuals normal
# residual plot
plt.plot(pred_sqrt,residuals_sqrt,'go');plt.xlabel('Fitted'),plt.ylabel('Residuals')
plt.title('Residual plot')


##########################
# sqaure(X)
delivery.deliverytime.corr(np.square(delivery.sortingtime)) #corr 0.79390
# corr value is less hence no further analysis with this

######################################
# inverse of squareroot
delivery.deliverytime.corr(1/np.sqrt((delivery.sortingtime))) # corr -0.8235
# corr value is less hence no further analysis with this

################################################
# 1/X
delivery.deliverytime.corr(1/(delivery.sortingtime)) # corr -0.80208
# corr value is less hence no further analysis with this

##################################################
# cuberoot(X) - very slight difference in values as comapred to log(X)

8**(1/3)
sorting_cuberoot = delivery.sortingtime**(1/3)
delivery.deliverytime.corr(sorting_cuberoot) # corr 0.8351156

reg_cuberoot = smf.ols('deliverytime ~ sorting_cuberoot',data=delivery).fit()
reg_cuberoot.summary()
# R2 0.697



