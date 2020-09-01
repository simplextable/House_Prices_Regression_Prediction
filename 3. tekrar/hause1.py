# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:38:13 2020

@author: Predator
"""

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


















#from sklearn.model_selection import train_test_split
#x_train, x_test,y_train,y_test = train_test_split(train,sale_price,test_size=0.33, random_state=0)







from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


y_train = sale_price


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))



#//////////// NORMAL STACK MODEL /////////////

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   



averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



#/////////////// SÜPER STACK MODELİ //////////////

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)



stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


















