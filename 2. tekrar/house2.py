###Öğrendiklerim:
"""
OTURUM1
1-heatmap size ayarlama
2-sadece seçili satırı çalıştırıp tek bir kere heatmap yazma
3-drop ve heatmap'i veri işlemeye ekledim
4-float - integer farkı öğrenildi
5-int numarası yapan categorik verileri tespit et methodu öğrenildi
6-int numarası yapan scale edilmesi lazım verileri tespit et methodu öğrenildi
7-exerqual gibi categorik olup aslında scale edilmiş int mantığına uyan verileri int'e 
dönüştürmek gerekir mi?
8-veri üzerinde edit yapmayı öğrendim
9-cat.codes yönetemi işe yaramaz çünkü bu label encoder 
10-def function anlamına geliyor

OTURUM2
1-concat'ı veri ön işleme dosyasına ekledim
2-train ve test verilerini birleştirip işlem yapma methodunu denedim.
3-get_dummies fonksiyonu araştırıldı
4-if-else sentax'ında parantez olayını öğrendik
5-multple onehot fonksiyonu
6-print info neden kategorikleştirmeden sonra özet şeklinde gelmeye başladı?
7-veri_isleme dosyasının eksik veri kısmı

OTURUM3
1-iloc sanırım ana değişkenden satır yok etmiyormuş
2-ana veri işleme dosyasına r2 score import eklendi
3-değişkenlerin p değerleri incelendi
4-veri işlemeye rmse eklendi
5-f9 direkt o satırı çalıştırıyor
6-rmse formülü incelendi. neden r2 score'u tümleyemeyeceği anlaşıldı
7-aynı isimde kolon adlarını nasıl değiştirebiliriz problemi üzerinde düşünüyorum. 
önce otomatik bir döngü yazayım dedim aklıma farklı fikir geldi.
panelden editleyeyim dedim, headerlar editlenmiyor.
kodla manuel olarak değiştireyim dedim bu sefer de indis vermezsek her iki duplicati de 
değiştirecek.
csv'ye import edip orda manuel değiştireyim dedim ama kod yazıp otomatik yapmak 
ilk anda zor olsa da aslında daha mantıklı

OTURUM4
1-sıralı bir index'e sahip bir değişken ile farklı indislere sahip bir değişken nasıl
grafiği çizilir?

OTURUM5
1-gridsearch nasıl yapılır pratiği

OTURUM6
1-list fonksiyonu ile tüm sütunları görüntülemeyi öğrendik
2-yılları eşit aralıklara bölmek mantıklı mı? son grup 1984'den 2010'a uzanıyor. acaba ilk marjinal kısmı ayrı mı gruplasak?
3-yılları gruplarken kaç gruba ayrıcağımızı bilmek için yeni değişken yaratıp o değişkeni açıp bakıp grup sınırları neymiş diye bakmaya gerek yok.
printle yapınca direk görüyoruz hem de yarattığımız drop etmekten kar ediyoruz. ayrıca direkt kopy-paste yapabiliyoruz.     
4-ilginç bir olay oldu. garage year built değişkeninde 2007 yazacaklarına 2207 diye yanlış bir değer vardı. onu değiştirip o değişkeni gruplayıp modeli 
çalıştırdım. model çöktü. r2 değeri -0.02'ye falan indi. rmse skoru 700 milyondan 7 milyara çıktı. 

OTURUM7
1-RF'e bakarken rmse ve r2 score değişiyor her bir defa
2-onehot yerine label yapılması gereken verilere label yapınca sonuç kötüleşiyor.
"""
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
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
pd.set_option("display.max_rows",10000)  # KISALTMA ENGELLEME
pd.set_option("display.max_columns",10000)
# import plotly.graph_objects as go
# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

veriler_train = pd.read_csv('train.csv')
veriler_test = pd.read_csv('test.csv')

sale_price = veriler_train[["SalePrice"]]

veriler_train.drop(["SalePrice"],axis=1,inplace=True)

idler = pd.read_csv("sample_submission.csv")
idler = idler[["Id"]]




#print("****train")
#print(veriler_train.isnull().sum()) # eksik verileri görme...
#print(veriler_tum.info())         #Veri Tiplerinin ne olduğunu bulma

#fig, ax = plt.subplots(figsize=(20,5))
#sns.heatmap(veriler_train.isnull(),yticklabels=False,cbar=False)     ###heatmap haritası
#
#fig, ax = plt.subplots(figsize=(20,5))
#sns.heatmap(veriler_test.isnull(),yticklabels=False,cbar=False)     ###heatmap haritası



#print("****test")
#print(veriler_train.isnull().sum()) #eksik verileri görme...
#print(veriler_tum.info())         #Veri Tiplerinin ne olduğunu bulma



veriler_tum=pd.concat([veriler_train,veriler_test],axis=0)

veriler_tum.drop(["Id","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1,inplace=True)


print(veriler_train.dtypes)


cat_cols = [col for col, dt in veriler_tum.dtypes.items() if dt == object]
cat_cols.append("MSSubClass")


cont_cols = [col for col in veriler_tum.columns if col not in cat_cols]

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='mean')
#most_frequent de diyebiliriz

imputer = imputer.fit(veriler_tum[cont_cols])
veriler_tum[cont_cols] = imputer.transform(veriler_tum[cont_cols])

#sonra categorik'lar
for i in cat_cols:
    if veriler_tum[i].isnull().sum() > 0:
        veriler_tum[i] = veriler_tum[i].fillna(veriler_tum[i].mode()[0])





#label encoder yapılması gereken veriler

#Street
#LotShape
#LandSlope
#ExterQual
#ExterCond 
#roofstyle'da kaldım 



#/////////burada streeti label edip catcolsdan çıkardık


#label_cols = ["Street","LotShape","LandSlope","ExterQual","ExterCond"]
#
#
#
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#for k in label_cols:
#    print(k)
#    veriler_tum[k] = le.fit_transform(veriler_tum[k])
#    
#    cat_cols.remove(k)
#
#print(veriler_tum["ExterCond"])





def category_onehot_multcols(multcolumns):
    kopya = veriler_tum
    i=0
    for sutun_ismi in multcolumns:
        
        print(sutun_ismi)
        v1=pd.get_dummies(veriler_tum[sutun_ismi],drop_first=True)
        
        veriler_tum.drop([sutun_ismi],axis=1,inplace=True)
        if i==0:
            kopya=v1.copy()
        else:
            
            kopya=pd.concat([kopya,v1],axis=1)
                
        i=i+1
       
        
    kopya=pd.concat([kopya,veriler_tum],axis=1)
        
    return kopya

veriler_tum=category_onehot_multcols(cat_cols)



veriler_train.columns[veriler_train.columns.duplicated()]





      
dupp_cols = veriler_tum.columns[veriler_tum.columns.duplicated()]
print(dupp_cols)    



counter = 0

for i in range(len(veriler_tum.columns)):
    for j in range(len(dupp_cols)):
        if veriler_tum.columns[i] == dupp_cols[j]:            
            veriler_tum.columns.values[i] = (f'{veriler_tum.columns[i]}{counter}')     
            counter = counter +1    



#veriler_tum['CategoricalYearBuilt'] = pd.cut(veriler_tum['YearBuilt'], 5)
print(veriler_tum.dtypes)

#veriler_tum['CategoricalYearBuilt'] = veriler_tum['CategoricalYearBuilt'].astype(int)


veriler_tum.loc[ veriler_tum['YearBuilt'] <= 1899.6, 'YearBuilt'] = 0
veriler_tum.loc[(veriler_tum['YearBuilt'] > 1899.6) & (veriler_tum['YearBuilt'] <= 1927.2), 'YearBuilt'] = 1
veriler_tum.loc[(veriler_tum['YearBuilt'] > 1927.2) & (veriler_tum['YearBuilt'] <= 1954.8), 'YearBuilt'] = 2
veriler_tum.loc[(veriler_tum['YearBuilt'] > 1954.8) & (veriler_tum['YearBuilt'] <= 1982.4), 'YearBuilt'] = 3
veriler_tum.loc[ veriler_tum['YearBuilt'] > 1982.4, 'YearBuilt'] = 4


#print(pd.cut(veriler_tum['GarageYrBlt'], 5))

#veriler_tum.loc[ veriler_tum['GarageYrBlt'] <= 1918.0, 'GarageYrBlt'] = 0
#veriler_tum.loc[(veriler_tum['GarageYrBlt'] > 1918.0) & (veriler_tum['GarageYrBlt'] <= 1941.0), 'GarageYrBlt'] = 1
#veriler_tum.loc[(veriler_tum['GarageYrBlt'] > 1941.0) & (veriler_tum['GarageYrBlt'] <= 1964.0), 'GarageYrBlt'] = 2
#veriler_tum.loc[(veriler_tum['GarageYrBlt'] > 1964.0) & (veriler_tum['GarageYrBlt'] <= 1987.0), 'GarageYrBlt'] = 3
#veriler_tum.loc[ veriler_tum['GarageYrBlt'] > 1987.0, 'GarageYrBlt'] = 4




#YearRemodAdd'i gruplayınca sonuç kötüleştiği için bu kısmı sildik
#print(pd.cut(veriler_tum['YearRemodAdd'], 5))

#veriler_tum.loc[ veriler_tum['YearRemodAdd'] <= 1962.0, 'YearRemodAdd'] = 0
#veriler_tum.loc[(veriler_tum['YearRemodAdd'] > 1962.0) & (veriler_tum['YearRemodAdd'] <= 1974.0), 'YearRemodAdd'] = 1
#veriler_tum.loc[(veriler_tum['YearRemodAdd'] > 1974.0) & (veriler_tum['YearRemodAdd'] <= 1986.0), 'YearRemodAdd'] = 2
#veriler_tum.loc[(veriler_tum['YearRemodAdd'] > 1986.0) & (veriler_tum['YearRemodAdd'] <= 1998.0), 'YearRemodAdd'] = 3
#veriler_tum.loc[ veriler_tum['YearRemodAdd'] > 1998.0, 'YearRemodAdd'] = 4



#YearSold'i gruplayınca sonuç değişmediği için bu kısmı sildik
#print(pd.cut(veriler_tum['YrSold'], 5))
#veriler_tum.loc[ veriler_tum['YrSold'] <= 2006.8, 'YrSold'] = 0
#veriler_tum.loc[(veriler_tum['YrSold'] > 2006.8) & (veriler_tum['YrSold'] <= 2007.6), 'YrSold'] = 1
#veriler_tum.loc[(veriler_tum['YrSold'] > 2007.6) & (veriler_tum['YrSold'] <= 2008.4), 'YrSold'] = 2
#veriler_tum.loc[(veriler_tum['YrSold'] > 2008.4) & (veriler_tum['YrSold'] <= 2009.2), 'YrSold'] = 3
#veriler_tum.loc[ veriler_tum['YrSold'] > 2009.2, 'YrSold'] = 4


veriler_train = veriler_tum.iloc[:1460,:]

veriler_test = veriler_tum.iloc[1460:,:]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(veriler_train,sale_price,test_size=0.33, random_state=0)





#/////////XGBOOST DIŞINDAKİ MODELLERİ DEVRE DIŞI BIRAKTIK ////////////

#from sklearn.linear_model import LinearRegression
#lr= LinearRegression()
#lr.fit(x_train, y_train)
#
#lr_pred = lr.predict(x_test)
#
#print("Linear Regression R2")
#print(r2_score(y_test, lr_pred))
#print("rmse değeri Linear")
#print(mean_squared_error(y_test, lr_pred))
#
#
##import statsmodels.api as sm
##X = np.append(arr = np.ones((1460, 1)).astype(int), values=tum_veriler, axis=1)
##r_ols = sm.OLS(endog = satis_fiyati, exog= X).fit()
##print(r_ols.summary())
#
#from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures(degree =2)
#x_poly = poly_reg.fit_transform(x_train)
#x_poly_test = poly_reg.fit_transform(x_test)
#
#lin_reg2 = LinearRegression()
#lin_reg2.fit(x_poly,y_train)
#
#
#y_pred4 = lin_reg2.predict(x_poly_test)
#
#print("poly_r değeri")
#print(r2_score(y_test, y_pred4))
#print("rmse değeri PolyLinear")
#print(mean_squared_error(y_test, y_pred4))
#
#
#from sklearn.svm import SVR
#
#svr = SVR(kernel= 'rbf')
#svr.fit(x_train, y_train)
#y_pred5 = svr.predict(x_test)
#
#print("r2 score SVR")
#print(r2_score(y_test, y_pred5))
#print("rmse değeri SVR")
#print(mean_squared_error(y_test, y_pred5))
#
#
#from sklearn.ensemble import RandomForestRegressor
#
#R_F = RandomForestRegressor()
#R_F.fit(x_train, y_train)
#y_pred3 = R_F.predict(x_test)
#
#print("Random Forest R2")
#print(r2_score(y_test, y_pred3))
#print("rmse değeri RFR")
#print(mean_squared_error(y_test, y_pred3))
#
#
#
#from sklearn.tree import DecisionTreeRegressor
#
#R_dt = DecisionTreeRegressor()
#R_dt.fit(x_train, y_train)
#y_pred2 = R_dt.predict(x_test)
#
#print("Decision Tree R2")
#print(r2_score(y_test, y_pred2))
#print("rmse değeri DT")
#print(mean_squared_error(y_test, y_pred2))




#//////////////////ESKİ MODELİMİZ /////////////
#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#
#import xgboost
#classifier=xgboost.XGBRegressor()
#
#
#regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
#       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=1)
#
#regressor.fit(x_train,y_train)
#y_pred6 = regressor.predict(x_test)
#print(y_pred6)
#print("xgboost R2")
#
#print(r2_score(y_test, y_pred6))
#print("rmse değeri xgboost")
#print(mean_squared_error(y_test, y_pred6))
#
#y_pred6_nihai = regressor.predict(veriler_test)
#y_pred6_nihai= pd.DataFrame(data = y_pred6_nihai, index = range(1459) , columns = ["SalePrice"])
#
#nihai=pd.concat([idler,y_pred6_nihai], axis=1)

#nihai.to_csv('04042020.csv',index=False)




#/////parametresiz xgboost
#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#
#import xgboost
#classifier=xgboost.XGBRegressor()
#
#
#regressor=xgboost.XGBRegressor()
#
#regressor.fit(x_train,y_train)
#y_pred6 = regressor.predict(x_test)
#print(y_pred6)
#print("xgboost R2")
#
#print(r2_score(y_test, y_pred6))
#print("rmse değeri xgboost")
#print(mean_squared_error(y_test, y_pred6))





#/////////////XGBOOST
from xgboost import XGBClassifier
classifier = XGBClassifier()

import xgboost
classifier=xgboost.XGBRegressor()


regressor=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=5, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=1,
       reg_alpha=0.6, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.9)

regressor.fit(x_train,y_train)
y_pred6 = regressor.predict(x_test)
print(y_pred6)
print("xgboost R2")

print(r2_score(y_test, y_pred6))
print("rmse değeri xgboost")
print(mean_squared_error(y_test, y_pred6))

y_pred6_nihai = regressor.predict(veriler_test)
y_pred6_nihai= pd.DataFrame(data = y_pred6_nihai, index = range(1459) , columns = ["SalePrice"])

nihai=pd.concat([idler,y_pred6_nihai], axis=1)

#nihai.to_csv('07042020.csv',index=False)




#SCAlE EDİLECEKLER
#YearBuilt
#YearRemodAdd
#GarageYrBlt
#YrSold


##train'de 1460 tane var 80 sütun
##test'te 1459 tane var 80 sütun



#///////GRIDSEARCH //////////
#from sklearn.model_selection import GridSearchCV
#p = [{'base_score':[0.5], 'booster':['gbtree'], 'colsample_bylevel':[0.1], 'colsample_bytree':[1], 
#      'gamma':[0], 'learning_rate':[0.1], 'max_delta_step':[0], 'max_depth':[5], 'min_child_weight':[1], 
#      'n_estimators':[1000], 'n_jobs':[1],'random_state':[1], 'reg_alpha':[0.6], 'reg_lambda':[1], 'scale_pos_weight':[1],
#      'subsample':[0.9]}]
#
#
#
#gs=GridSearchCV(estimator=classifier, 
#                param_grid=p,
#                scoring= 'neg_mean_squared_error',
#                cv=10,
#                n_jobs = -1)
#
#grid_search = gs.fit(x_train, y_train)
#eniyisonuc = grid_search.best_score_
#eniyiparametreler = grid_search.best_params_
#
#print("en iyi sonuç:",eniyisonuc)
#print("en iyi parametre:", eniyiparametreler)




###eldeki optimum varsayılan parametrelerin
#önceki çözümün rmse skoru: 911551304.1281614
#base_score=0.25, booster=gbtree

#911551304.1281614
#875980559

###colsample_bylevel'i 1'den 0.1'e düşür dedi. colsample_bytree değiştirme 1 kalsın dedi. gamma 0 kalsın dedi. max_delta 0 kalsın.
#learning_rate aynı kalsın 0.1 olsun dedi.
#911551304.1281614
#875980559
#--819119326
#max_depth 5 olsun dedi sonuç bir tık iyileşti
#911551304
#875980559
#--819119326
#-793353865

#n_estimator 1000 olsun dedi
#-793353865
#-789740030

#random_state 1 olsun dedi. 
#-789740030
#-763598883

#reg_alpha 0.6 olsun dedi.
#-763598883
#-761685769

#subsample 0.9 olsun dedi
#761685769
#758952040


##yılları scale etmeden gelen sonuç:
#758952040
#721503160

#yearbuilt sütununu grupladığımızda gelen sonuç:
#721503160
#713228363

#YearRemodAdd sütununu gruplayınca sonuç kötüleşti
#713228363
#720494754

#YearSld sütununu gruplayınca sonuç tamamen aynı kaldı

#GarageYrBlt sütununu gruplayınca sonuç kötüleşti
#720494754
#720903486

#Street'i label edince sonuç kötüleşti
#720903486
#734629216

#LotShape'i label yapınca sonuç kötüleşti
#720903486
#772393201

#Sıradan RF
#1044270554 - #992443427

#Sıradan RF'de ExterQualı label yapınca sonuç kötüleşti
#1044270554 - #992443427 - #1119105879
#995403954 - #1011675693 - #949048182

#sıradan RF'de 4 tane label yapınca sonuç kötüleşti
#1044270554 - #992443427 - #1119105879
#1116417826

#xgboost parametresiz, 5 labelli sonuç kötüleşti
#782755821
#806571845


#////görselleştirme için yeni yapay değişkenler ////
#gercekfiyat_yapimyil=pd.concat([y_test_1.SalePrice,x_test.YearBuilt],axis=1)
#tahminfiyat_yapimyil=pd.concat([y_pred6_1,x_test.YearBuilt],axis=1)
#
#
#y_test_1 = y_test.reset_index()
#y_pred6_1= pd.DataFrame(data = y_pred6, index = range(482) , columns = ["SalePrice"])
#
##y_pred6_nihai_1 = y_pred6_nihai.reset_index()
#
#
#plt.figure(figsize=(60,10))
#
#
#plt.plot(gercekfiyat_yapimyil.SalePrice, color="blue")
#plt.plot(tahminfiyat_yapimyil, color="red")
#
#
#plt.xlabel("yapım yılı")
#plt.ylabel("ev fiyatı")
#plt.show()










