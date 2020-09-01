import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os


#1. kutuphaneler
# linear algebra
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML

# collection of machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import pylab as pl
from sklearn.metrics import roc_curve

#from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score
pd.set_option("display.max_rows",2200)  # KISALTMA ENGELLEME
from scipy import stats
from scipy.stats import norm,skew 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# import plotly.graph_objects as go
# Ignore warnings
import warnings
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
tum= pd.concat([test,train],axis=0)



sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.show()

sns.set_style("whitegrid")
missing = test.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()




sıfır = ["LotFrontage" , "Alley" , "MasVnrType" , "BsmtQual" ,  
"BsmtCond", "BsmtExposure" , "FireplaceQu", "GarageFinish", "GarageQual" , "GarageCond" ,
"BsmtFinType1" ,"BsmtFinType2" , "GarageType" ,"GarageFinish", "PoolQC" ,"Fence",
"MiscFeature", "Functional","KitchenQual", "TotalBsmtSF","Exterior1st", "Exterior2nd"]


for i in sıfır:
    tum[i]=tum[i].fillna("0")  
 
    
tum.drop(["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF","Id" ],axis=1,inplace=True)


tum['BsmtFullBath']=tum['BsmtFullBath'].fillna(tum['BsmtFullBath'].mode()[0]) 
tum['BsmtHalfBath']=tum['BsmtHalfBath'].fillna(tum['BsmtHalfBath'].mode()[0]) 
tum['Electrical']=tum['Electrical'].fillna(tum['Electrical'].mode()[0]) 
tum['GarageArea']=tum['GarageArea'].fillna(tum['GarageArea'].mean())
tum['GarageCars']=tum['GarageCars'].fillna(tum['GarageCars'].mode()[0]) 
tum['GarageYrBlt']=tum['GarageYrBlt'].fillna(tum['GarageYrBlt'].mean())
tum['MSZoning']=tum['MSZoning'].fillna(tum['MSZoning'].mode()[0]) 
tum['MasVnrArea']=tum['MasVnrArea'].fillna(tum['MasVnrArea'].mean())
tum['SaleType']=tum['SaleType'].fillna(tum['SaleType'].mode()[0]) 
tum['Utilities']=tum['Utilities'].fillna(tum['Utilities'].mode()[0]) 

#plt.subplots(figsize=(16,16))
#sns.heatmap(train.corr(), annot=True, fmt=".2f")
#plt.show()   
#
#
#sns.distplot(train['SalePrice']);
#plt.show() 

var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

tum = tum.drop(tum[(tum['GrLivArea']>4000) & (tum['SalePrice']<300000)].index)



sns.distplot(train['SalePrice']);
plt.show()
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#print(tum.info()) 


numeric_feats = tum.dtypes[tum.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = tum[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

sns.distplot(tum['MiscVal'] , fit=norm);









label = ["LotShape" , "LandContour" ,  "LandSlope" , "OverallQual", "OverallCond", "ExterQual" ,
"ExterCond" , "BsmtQual", "BsmtCond" ,"BsmtExposure" , "BsmtFinType1" , "BsmtFinType2" ,  
"HeatingQC" , "CentralAir" ,  "KitchenQual" , "Fireplaces" , "FireplaceQu",
  "TotRmsAbvGrd" , "GarageFinish", "GarageCars" , "GarageQual", "GarageCond",
"PoolQC","Fence" ]


for c in label:
    lbl = LabelEncoder() 
    lbl.fit(list(tum[c].values)) 
    tum[c] = lbl.transform(list(tum[c].values))


tum["TotalBsmtSF"] = tum["TotalBsmtSF"].astype('int64')
tum["LotFrontage"] = tum["LotFrontage"].astype('int64')
tum["MSSubClass"] = tum["MSSubClass"].astype("object")
tum = pd.get_dummies(tum)

standardize = ["LotFrontage" ,"LotArea", "YearBuilt", 
               "MasVnrArea", "TotalBsmtSF",
"1stFlrSF" ,  "2ndFlrSF" , "LowQualFinSF", 
"GarageYrBlt", "GarageArea" , "WoodDeckSF", "OpenPorchSF" , "EnclosedPorch", "3SsnPorch" , "ScreenPorch",
"PoolArea" , "MiscVal", "YrSold", "YearRemodAdd"]


sc = StandardScaler()
for i in standardize:
    tum[[i]] = sc.fit_transform(tum[[i]])


tum = tum.reindex(sorted(tum.columns), axis=1)


tum['GrLivArea'] = np.log1p(tum['GrLivArea'])






test=tum.iloc[:1457,:]
train=tum.iloc[1457:,:]




sale_price = train[["SalePrice"]]    

train.drop(["SalePrice" ],axis=1,inplace=True)
test.drop(["SalePrice" ],axis=1,inplace=True)


sale_price = np.log1p(sale_price)



sns.distplot(sale_price['SalePrice'] , fit=norm);
sns.distplot(train['GrLivArea'] , fit=norm);



kf = KFold(n_splits=5, random_state=42, shuffle=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=train):
    rmse = np.sqrt(-cross_val_score(model, X, sale_price, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)






# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)


# XGBoost Regressor
xgboost = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# Ridge Regressor
ridge_alphas = [0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)



scores = {}

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())


score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())


score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())

score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())



print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(train), np.array(sale_price))


print('lightgbm')
lgb_model_full_data = lightgbm.fit(train, sale_price)


print('Svr')
svr_model_full_data = svr.fit(train, sale_price)

print('Ridge')
ridge_model_full_data = ridge.fit(train, sale_price)


print('RandomForest')
rf_model_full_data = rf.fit(train, sale_price)


print('GradientBoosting')
gbr_model_full_data = gbr.fit(train, sale_price)

print('xgboost')
xgb_model_full_data = xgboost.fit(train, sale_price)




# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(x):
    return (( (  0.1 * svr_model_full_data.predict(x)) + \
                (0.1 * gbr_model_full_data.predict(x)) + \
                (0.2 * xgb_model_full_data.predict(x)) + \
                (0.2 * lgb_model_full_data.predict(x)) + \
                (0.05 * rf_model_full_data.predict(x)) + \
                (0.35 * stack_gen_model.predict(np.array(x)))))



blended_score = rmsle(sale_price, blended_predictions(train))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


submission = pd.read_csv("sample_submission.csv")

stacked_pred = np.expm1(blended_predictions(test))
#
y_pred_17= pd.DataFrame(data = stacked_pred, index = range(1459) , columns = ["SalePrice"])
idler = pd.read_csv("sample_submission.csv")
idler = idler[["Id"]]

nihai=pd.concat([idler,y_pred_17], axis=1)

#nihai.to_csv('18042020.csv',index=False)











