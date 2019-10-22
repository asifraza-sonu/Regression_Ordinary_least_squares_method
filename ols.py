import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
pd.read_csv ('father_son_heights.csv')
dataset = pd.read_csv ('father_son_heights.csv')
dataset.columns
dataset.describe
X = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size = 0.3,random_state = 101)
plt.scatter(X,y)
plt.xlabel('FatherTestData')
plt.ylabel('SonTestData')
fig1 = plt.gcf()
fig1.savefig('scatterplot.png')
plt.show()
plt.close()
reg = LinearRegression()
reg.fit(X_train,y_train)
reg.intercept_
reg.coef_
print('Predicted intercept value is: ', reg.intercept_)
print('Predicted  coeffient value is: ',reg.coef_ )
np.amin(X_test)
np.savetxt('Coefficient_Intercept.txt',(reg.coef_,reg.intercept_))
np.amax(X_test)
x1 = np.linspace(np.amin(X_test),np.amax(X_test),50)
predictions = reg.predict(X_test)
z = np.linspace(np.amin(predictions),np.amax(predictions),50)
plt.scatter(x1,z)
plt.xlabel('FatherGivenData')
plt.ylabel('PredictedSonData')
fig2 = plt.gcf()
fig2.savefig('line_and_scatter.png')
plt.show()
plt.close()
#For distribution of residuals - additional to the question
import seaborn as sns
distribution_residuals=sns.distplot(y_test-predictions)
Error_distribution = distribution_residuals.figure
Error_distribution.savefig('Error_distribution.png')
#For Metrics like Mean absolute error (MAE) ,Mean squared error (MSE), Root Mean squared error(RMSE) - Additional to the question
MAE = metrics.mean_absolute_error(y_test,predictions)
MSE = metrics.mean_squared_error(y_test,predictions)
RMSE = np.sqrt(MSE)
print('Mean absolute error (MAE) is :',MAE)
print('Mean squared error (MSE)) is :',MSE)
print('Root Mean squared error(RMSE) is :',RMSE)