# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:38:13 2020

@author: Predator
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


test_df=pd.read_csv('test.csv')
df = pd.read_csv("train.csv")
pd.set_option("display.max_rows",2200)

print(df.head())
print(df.isnull().sum()) #null değerleri gördük
print(sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm'))



def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']



#/////////////***********    Train Data Düzenleme     ***************//////////////////

print(df.shape)
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])



df.drop(['Alley'],axis=1,inplace=True)
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df.drop(['PoolQC','Fence','MiscFeature',"Id"],axis=1,inplace=True)
print("1.",df.shape)
df.dropna(inplace=True) # NaN olan satırları temizledi...
print("2.",df.shape)
print(df.isnull().sum())
#print(sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm'))


main_df=df.copy()
test_df=pd.read_csv('formulatedtest.csv')

print("train data : ", df.shape)
print("test data : ",test_df.shape)

final_df=pd.concat([df,test_df],axis=0)




from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
#
unique_values = pd.unique(final_df.values.ravel())
#
ohe = OneHotEncoder(categories=[str]*len(final_df), sparse=False)


encoded = pd.DataFrame(ohe.fit_transform(final_df), columns=ohe.get_feature_names(final_df.columns))

"""
final_df=category_onehot_multcols(columns)
final_df =final_df.loc[:,~final_df.columns.duplicated()]

df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]

df_Test.drop(['SalePrice'],axis=1,inplace=True)

X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
"""



"""
from sklearn.ensemble import RandomForestRegressor


rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X_train,y_train)
y_pred = rf_reg.predict(df_Test)

#print(r2_score(Y,rf_reg.predict(X)))
plt.plot(X_train,rf_reg.predict(X_train), color = 'blue')
"""