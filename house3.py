# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:51:32 2020

@author: Predator
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

veriler_train = pd.read_csv("train.csv")

yapim_yil = veriler_train[["YearBuilt"]]
satis_fiyati = veriler_train[["SalePrice"]]
lot_area = veriler_train[["LotArea"]]
GrLivArea = veriler_train[["GrLivArea"]]
BedroomAbvGr = veriler_train[["BedroomAbvGr"]]
GarageArea = veriler_train[["GarageArea"]]
SsnPorch = veriler_train[["3SsnPorch"]]

Neighborhood = veriler_train[["Neighborhood"]]

BldgType = veriler_train[["BldgType"]]


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = 'all')
Neighborhood2 = ohe.fit_transform(Neighborhood).toarray()
Neighborhood3 = pd.DataFrame(data=Neighborhood2, index =range(1460), columns=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"])

ohe = OneHotEncoder(categorical_features = 'all')
BldgType2 = ohe.fit_transform(BldgType).toarray()

BldgType3 = pd.DataFrame(data=BldgType2, index =range(1460), columns=["TwinSE","TwinS","Dublex","2FMCON","1FAM"])





tum_veriler = pd.concat([yapim_yil, lot_area, GrLivArea, BedroomAbvGr, GarageArea, SsnPorch, BldgType3], axis = 1 )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(tum_veriler, satis_fiyati, test_size = 0.33, random_state=0 )


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

print("Linear Regression R2")
print(r2_score(y_test, lr_pred))

plt.plot(x_test, y_test)
plt.plot(x_test, lr_pred)

import statsmodels.api as sm
X = np.append(arr = np.ones((1460, 1)).astype(int), values=tum_veriler, axis=1)
r_ols = sm.OLS(endog = satis_fiyati, exog= X).fit()
print(r_ols.summary())

from sklearn.tree import DecisionTreeRegressor

R_dt = DecisionTreeRegressor()
R_dt.fit(x_train, y_train)
lr_pred2 = R_dt.predict(x_test)

print("Decision Tree R2")
print(r2_score(y_test, lr_pred2))

from sklearn.ensemble import RandomForestRegressor

R_F = RandomForestRegressor()
R_F.fit(x_train, y_train)
lr_pred3 = R_F.predict(x_test)

print("Random Forest R2")
print(r2_score(y_test, lr_pred3))



from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()

Tum_veriler = sc1.fit_transform(tum_veriler)
Satis_fiyati = sc2.fit_transform(satis_fiyati)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Tum_veriler, Satis_fiyati, test_size = 0.33, random_state=0 )


from sklearn.svm import SVR

svr = SVR(kernel= 'rbf')
svr.fit(x_train, y_train)
y_pred5 = svr.predict(x_test)

print("r2 score SVR")
print(r2_score(y_test, y_pred5))

print("rmse değeri SVR")

from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, y_pred5)
print(rmse)




from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
x_poly = poly_reg.fit_transform(x_train)
x_poly_test = poly_reg.fit_transform(x_test)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)


y_pred4 = lin_reg2.predict(x_poly_test)

print("poly_r değeri")
print(r2_score(y_test, y_pred4))