# Data Gathering
from pydataset import data
cars=data('mtcars')

# Data Processing
cars=cars[['wt','mpg']]

# Handling Missing Values
cars.isna().sum()

x=pedictors=cars['wt'].values.reshape(-1, 1)
y=target=cars['mpg'].values.reshape(-1, 1)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.xlabel('WT')
plt.ylabel('MPG')

# Model creation
# Simple linear regression model
# ols- ordinary least squared
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

b0=model.intercept_ 
b1=model.coef_
print("b0",b0)
print("b1",b1)
#mpg=b0+b1*wt#   mpg=37.28+-5.34*(wt)
pred=model.predict(x)


plt.scatter(x,y)
plt.plot(x,pred,c='r')

# predicting mpg when wt is 5
model.predict([[5]])
# from numpy import sqrt
# sqrt(9)
# import numpy as np
# np.sqrt(9)


# Model Evaluation
# RMSE
from numpy import sqrt,mean
error=pred-y
mse=mean(error*error)
rmse=sqrt(mean(error*error))
mae=mean(abs(error))

print("MSE",round(mse,2))
print("RMSE",round(rmse,2))
print("MAE",round(mae,2))

from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error(pred,y)
r2_score(pred,y)# 67%


# Train and Test Split
from sklearn.model_selection import train_test_split
train,test=train_test_split(cars,test_size=.2)
trainx=train['wt'].values.reshape(-1,1)
trainy=train['mpg']
testx=test['wt'].values.reshape(-1,1)
testy=test['mpg']
# Model Creation
model_train=LinearRegression()
model_train.fit(trainx,trainy)

# Model Evaluation
# Training Accuracy
train_pred=model_train.predict(trainx)
sqrt(mean_squared_error(train_pred,trainy)) #2.74
# Testing Accuracy
test_pred=model_train.predict(testx)
sqrt(mean_squared_error(test_pred,testy))  #3.75



