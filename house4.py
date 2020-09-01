import numpy as np
import pandas as pd
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import skew, norm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from IPython.core.display import display, HTML
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import model_selection
import pylab as pl
from sklearn.metrics import roc_curve
from scipy import stats


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Missing data
sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.show()

sns.set_style("whitegrid")
missing2 = test.isnull().sum()
missing2 = missing2[missing2 > 0]
missing2.sort_values(inplace=True)
missing2.plot.bar()
plt.show()



plt.subplots(figsize=(16,16))
sns.heatmap(train.corr(), annot=True, fmt=".2f")
plt.show()  

 
#Discover Outliers
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


#Deleting outliers especially Correlation Rate is High One
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


tum= pd.concat([train,test],axis=0)


# Filling "0" instead of Missing Values
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


tum.isnull().sum()




#Distribution of Saleprice 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()



#Log-transformation of the target SalePrice
train["SalePrice"] = np.log1p(train["SalePrice"])

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

sns.distplot(tum['GrLivArea'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(tum['GrLivArea'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('GrLivArea distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(tum['GrLivArea'], plot=plt)
plt.show()


# Log-transformation of the target GrLivArea
tum['GrLivArea'] = np.log1p(tum['GrLivArea'])

sns.distplot(tum['GrLivArea'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(tum['GrLivArea'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('GrLivArea distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(tum['GrLivArea'], plot=plt)
plt.show()


# Label Encoding of Some Columuns
label = ["LotShape" , "LandContour" ,  "LandSlope" , "OverallQual", "OverallCond", "ExterQual" ,
"ExterCond" , "BsmtQual", "BsmtCond" ,"BsmtExposure" , "BsmtFinType1" , "BsmtFinType2" ,  
"HeatingQC" , "CentralAir" ,  "KitchenQual" , "Fireplaces" , "FireplaceQu",
  "TotRmsAbvGrd" , "GarageFinish", "GarageCars" , "GarageQual", "GarageCond",
"PoolQC","Fence" ]

for c in label:
    lbl = LabelEncoder() 
    lbl.fit(list(tum[c].values)) 
    tum[c] = lbl.transform(list(tum[c].values))

tum.info()

tum["TotalBsmtSF"] = tum["TotalBsmtSF"].astype('int64')
tum["LotFrontage"] = tum["LotFrontage"].astype('int64')
tum["MSSubClass"] = tum["MSSubClass"].astype("object")
tum = pd.get_dummies(tum)


print(tum.head())
print(tum.info())

# Some Columuns have been Standardized
standardize = ["LotFrontage" ,"LotArea", "YearBuilt", 
               "MasVnrArea", "TotalBsmtSF",
"1stFlrSF" ,  "2ndFlrSF" , "LowQualFinSF", 
"GarageYrBlt", "GarageArea" , "WoodDeckSF", "OpenPorchSF" , "EnclosedPorch", "3SsnPorch" , "ScreenPorch",
"PoolArea" , "MiscVal", "YrSold", "YearRemodAdd"]


sc = StandardScaler()
for i in standardize:
    tum[[i]] = sc.fit_transform(tum[[i]])
    
tum = tum.reindex(sorted(tum.columns), axis=1)


train=tum.iloc[:train.shape[0], :]
test=tum.iloc[train.shape[0]:,:]
test.head()

sale_price = train[["SalePrice"]]    
train.drop(["SalePrice" ],axis=1,inplace=True)
test.drop(["SalePrice" ],axis=1,inplace=True)


# Stack Model

kf = KFold(n_splits=5, random_state=42, shuffle=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=train):
    rmse = cross_val_score(model, X, sale_price, scoring="r2", cv=kf)
    return (rmse)



# LightGbm Regressor
lightgbm = LGBMRegressor()


# XGBoost Regressor
xgboost = XGBRegressor()


# Ridge Regressor
ridge_alphas = [0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor()

# Random Forest Regressor
rf = RandomForestRegressor()

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# Skorlar

scores = {}

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())


score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())

score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())


# Fit Models

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(train), np.array(sale_price))


print('lightgbm')
lgb_model_full_data = lightgbm.fit(train, sale_price)


print('Ridge')
ridge_model_full_data = ridge.fit(train, sale_price)


print('RandomForest')
rf_model_full_data = rf.fit(train, sale_price)


print('GradientBoosting')
gbr_model_full_data = gbr.fit(train, sale_price)

print('GradientBoosting')
xgb_model_full_data = xgboost.fit(train, sale_price)

def blended_predictions(x):
    return ((   
                (0.2 * gbr_model_full_data.predict(x)) + \
                (0.2 * xgb_model_full_data.predict(x)) + \
                (0.2 * lgb_model_full_data.predict(x)) + \
                (0.05 * rf_model_full_data.predict(x)) + \
                (0.35 * stack_gen_model.predict(np.array(x)))))



blended_score = rmsle(sale_price, blended_predictions(train))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)

submission = pd.read_csv("sample_submission.csv")

stacked_pred = blended_predictions(test)

y_pred_17= pd.DataFrame(data = stacked_pred, index = range(1459) , columns = ["SalePrice"])
idler = pd.read_csv("sample_submission.csv")
idler = idler[["Id"]]

nihai=pd.concat([idler,y_pred_17], axis=1)
