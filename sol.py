# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:57:02 2018

@author: PAVEETHRAN
"""

#IMPORTING LIBS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing files
train1=pd.read_csv('train.csv')
train2=pd.read_csv('historical_user_logs.csv')
test=pd.read_csv('test.csv')
ss=pd.read_csv('ss.csv')

#preprocess
train=train1

train=train.drop(['session_id','DateTime'],axis=1)
test_id=test['session_id']
test=test.drop(['session_id','DateTime'],axis=1)

train.hist()

#plt.bar(train['user_depth'],train['is_click'])
#plt.xlabel('user interaction')
#plt.ylabel('click y/n')
#plt.show()


import seaborn as sns
sns.countplot(train['is_click'],label='count')
plt.show()

sns.distplot(train['is_click'])

#corr matrix
corr=train.corr()
sns.heatmap(corr,robust=True,square=True)

#MISSING DATA
mv=train.isnull().sum()
mv=mv*100/len(train)
mv=mv.drop(mv[mv==0].index)
mv=mv.sort_values()

sns.barplot(x=mv.index,y=mv)
plt.xlabel('features')
plt.ylabel('frequency of missing values')

#train[((train['product_category_2']==82527)&(train['is_click']==1)).index]

y=train['is_click']
train=train.drop('is_click',1)


#HERE IM TRYING TO FIND WHAT INDEX IS PRODUCT CATEGORY =.........
#is_c=train1[train1['is_click']==1].index
#train.loc['product_category_2']


train.dtypes
cat_fea=train.dtypes.loc[train.dtypes=='object'].index
int_fea=train.dtypes.loc[train.dtypes=='integer'].index
float_fea=train.dtypes.loc[train.dtypes=='float'].index

#from sklearn.preprocessing import Imputer
#values = train[int_fea]
#imputer = Imputer(missing_values='nan', strategy='most_frequent')
#transformed_values = imputer.fit_transform(values)

for var in cat_fea:
    r=train[var].mode()
    train[var].fillna(r[0],inplace=True)

for var in int_fea:
    r=train[var].mode()
    train[var].fillna(r[0],inplace=True)


for var in float_fea:
    r=train[var].mode()
    train[var].fillna(r[0],inplace=True)


train.isnull().sum()

#HANDLING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
oh=OneHotEncoder()

for var in cat_fea:
    print(var)
    train[var]=le.fit_transform(train[var])
    
    
    
#-----------

test.dtypes
cat_fea=test.dtypes.loc[test.dtypes=='object'].index
int_fea=test.dtypes.loc[test.dtypes=='integer'].index
float_fea=test.dtypes.loc[test.dtypes=='float'].index


for var in cat_fea:
    r=test[var].mode()
    test[var].fillna(r[0],inplace=True)

for var in int_fea:
    r=test[var].mode()
    test[var].fillna(r[0],inplace=True)


for var in float_fea:
    r=test[var].mode()
    test[var].fillna(r[0],inplace=True)


test.isnull().sum()

#HANDLING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
oh=OneHotEncoder()

for var in cat_fea:
    print(var)
    test[var]=le.fit_transform(test[var])
 
    #DROP ONE UNWANTED COLUMN
train=train.drop('user_id',1)
test=test.drop('user_id',1)
    
##onehotencode
#rt=train1
#cat_fea=rt.dtypes.loc[rt.dtypes=='object'].index
#cat_fea=cat_fea.drop('DateTime')
#rt=train

#train
indx=[]
for var in cat_fea:
    indx.append(train.columns.get_loc(var))
    print(indx)
    
    
oh=OneHotEncoder(categorical_features=[0])
xt=oh.fit_transform(train).toarray()
           
#test

indx=[]
for var in cat_fea:
    indx.append(test.columns.get_loc(var))
    print(indx)

 oh=OneHotEncoder(categorical_features=[0])
 xtest=oh.fit_transform(test).toarray()

    
test.isnull().sum()

#SCALE THE VALUES
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xt=sc.fit_transform(xt)
xtest=sc.transform(xtest)

#skews
from scipy.stats import skew

sktr=skew(xt)
skte=skew(xtest)

xt=np.log(abs(xt))
xtest=np.log(abs(xtest))

#
#from scipy.special import boxcox1p
#xrt=boxcox1p(xrt,0.5)
#from scipy.stats import boxcox
#boxcox(abs(xt))

#RFC
from sklearn.ensemble import RandomForestClassifier
%timeit
cl=RandomForestClassifier(n_estimators=70)
cl.fit(xt,y)
y_pred=cl.predict(xtest)#.......0.5004774878287908.(100 estimators)

#....: 0.500491232777522.(N_EST-50)
#..... 0.5004767707534293(NEST=20)
#..... 0.5004767707534293.(N_EST=70)
#......0.5005324676237161(est=120)

#
#cl.feature_importances_

para=[{'n_estimators':[130,135]}]
from sklearn.model_selection import GridSearchCV
gs1=GridSearchCV(estimator=cl,param_grid=para,cv=5,verbose=True)
#Fitting 5 folds for each of 3 candidates, totalling 15 fits
#[Parallel(n_jobs=1)]: Done  15 out of  15 | elapsed: 14.9mi

gs1.fit(xt,y)
gs1.best_score_
gs1.best_params_#.......best estiators=120

y_pred=gs.predict(xtest)

#XGB
import xgboost
from xgboost import XGBClassifier

xg=XGBClassifier()
xg.fit(xt,y)

y_pred2=xg.predict(xtest)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',verbose=1)
lr.fit(xt,y)

y_pred3=lr.predict(xtest)



from sklearn.model_selection import cross_val_score as cvs
acc=[]

acc=cvs(estimator=xg,X=xt,y=y,cv=10)
#3 WAYS TO FIND MEAN
np.mean(acc)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(xt, y)


from sklearn.metrics import roc_auc_score as roc


#SUBMISSIONNNNNNNNNNNNNNNNNNNNNNNN
su=pd.DataFrame({'session_id':test_id,'is_click':y_pred})
su.to_csv('rand_for_500_aa.csv',index=False)#0.5004



from sklearn.ensemble import GradientBoostingClassifier
gboost=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=5)
gboost.fit(xt,y)
pre=gboost.predict(xtest)

#kernel approxIMATION
from sklearn.kernel_approximation import RBFSampler
rbf_fea=RBFSampler(gamma=1,random_state=1)
x_fea=rbf_fea.fit_transform(xt)
xtest_fea=rbf_fea.fit_transform(xtest)

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(verbose=1,random_state=1,n_iter=5)
sgd.fit(x_fea,y)
sgd.score(x_fea,y)#0.9323729578170091

y_pred=sgd.predict(xtest_fea)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(xt, y, batch_size = 10, nb_epoch = 10)

# Part 3 - Making the predictions and evaluating the model
y_pred_nn=classifier.predict(xtest)

for i  in range(len(y_pred)):
    if y_pred_nn[i]>0.5:
        y_pred_nn[i]=1
    else:
        y_pred_nn[i]=0    


#A NEW IDEA....THIS IS A PROBLEM OF ONE SIDED DATA(IMBALANCED DATA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing files
train1=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
ss=pd.read_csv('ss.csv')


from sklearn.preprocessing import StandardScaler

train=train1

train=train.drop(['session_id','DateTime'],axis=1)
test_id=test['session_id']
test=test.drop(['session_id','DateTime'],axis=1)


#MISSING DATA
mv=train.isnull().sum()
mv=mv*100/len(train)
mv=mv.drop(mv[mv==0].index)
mv=mv.sort_values()

sns.barplot(x=mv.index,y=mv)
plt.xlabel('features')
plt.ylabel('frequency of missing values')

import seaborn as sns
y=train['is_click']
train=train.drop('is_click',1)


#HERE IM TRYING TO FIND WHAT INDEX IS PRODUCT CATEGORY =.........
#is_c=train1[train1['is_click']==1].index
#train.loc['product_category_2']


train.dtypes
cat_fea=train.dtypes.loc[train.dtypes=='object'].index
int_fea=train.dtypes.loc[train.dtypes=='integer'].index
float_fea=train.dtypes.loc[train.dtypes=='float'].index

#from sklearn.preprocessing import Imputer
#values = train[int_fea]
#imputer = Imputer(missing_values='nan', strategy='most_frequent')
#transformed_values = imputer.fit_transform(values)

for var in cat_fea:
    r=train[var].mode()
    train[var].fillna(r[0],inplace=True)

for var in int_fea:
    r=train[var].mode()
    train[var].fillna(r[0],inplace=True)


for var in float_fea:
    r=train[var].mode()
    train[var].fillna(r[0],inplace=True)


train.isnull().sum()

#HANDLING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
oh=OneHotEncoder()

for var in cat_fea:
    print(var)
    train[var]=le.fit_transform(train[var])
    
    
    
#-----------

test.dtypes
cat_fea=test.dtypes.loc[test.dtypes=='object'].index
int_fea=test.dtypes.loc[test.dtypes=='integer'].index
float_fea=test.dtypes.loc[test.dtypes=='float'].index


for var in cat_fea:
    r=test[var].mode()
    test[var].fillna(r[0],inplace=True)

for var in int_fea:
    r=test[var].mode()
    test[var].fillna(r[0],inplace=True)


for var in float_fea:
    r=test[var].mode()
    test[var].fillna(r[0],inplace=True)


test.isnull().sum()

#HANDLING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
oh=OneHotEncoder()

for var in cat_fea:
    print(var)
    test[var]=le.fit_transform(test[var])
 
    #DROP ONE UNWANTED COLUMN
train=train.drop('user_id',1)
test=test.drop('user_id',1)
    

#train
indx=[]
for var in cat_fea:
    indx.append(train.columns.get_loc(var))
    print(indx)
    
    
oh=OneHotEncoder(categorical_features=[0])
xt=oh.fit_transform(train).toarray()
           
#test

indx=[]
for var in cat_fea:
    indx.append(test.columns.get_loc(var))
    print(indx)

oh=OneHotEncoder(categorical_features=[0])
xtest=oh.fit_transform(test).toarray()

    
test.isnull().sum()

#SCALE THE VALUES
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xt=sc.fit_transform(xt)
xtest=sc.transform(xtest)


#0---not clicked
#1---clicked

#from keras.utils import to_categorical
#cat=train['product']
#
#enc1=to_categorical(train.iloc[:,0])

xt=pd.DataFrame(xt)
xtest=pd.DataFrame(xtest)

#fraud=====click

clicks=len(train1[train1['is_click']==1])
click_indices=np.array(train1[train1['is_click']==1].index)

noclick_indices=np.array(train1[train1['is_click']==0].index)

import random
rand_normal_indices=np.array(np.random.choice(noclick_indices,size=clicks,replace=False))
xt
#appending them
undersample_indices=np.concatenate([click_indices,rand_normal_indices])


# under sampled dataset
undersample_data=xt.iloc[undersample_indices,:]

x_undersampled=undersample_data
y_undersampled=y[undersample_indices]

print('percentage of clicks:', len(y_undersampled[y_undersampled==1])*100/len(y_undersampled))
print('percentage of no clicks:', len(y_undersampled[y_undersampled==0])*100/len(y_undersampled))
print('total size of under sampled data', len(y_undersampled))


from sklearn.cross_validation import train_test_split

x_train_un,x_test_un,y_train_un,y_test_un=train_test_split(x_undersampled,y_undersampled,test_size=0.25,random_state=1)

print("Number transactions train dataset: ", len(x_train_un))
print("Number transactions test dataset: ", len(x_test_un))
print("Total number of transactions: ", len(x_train_un)+len(x_test_un))

from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import confusion_matrix,roc_curve,recall_score


best_c = kfold_printing(x_train_un,y_train_un)


j=0
def kfold_printing(x_t,y_t):
    fold=KFold(n=len(y_t),n_folds=5,shuffle=False)
    
    #varying for different c parameters
    c_para=[{'C':[0.01,0.1,1,10,100]}]
    
    
    results_table=pd.DataFrame(index=range(len(c_para)),columns=['c_parameter','recall_score'])
    results_table['c_parameter']=c_para
    
    #perform the k fold
    for c in c_para:
        print('---------')
        print('c_parameter',c_para)
        print('---------')
        recall_accs=[]
        # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
        for iteration,indices in enumerate(fold,start=1):
            print('hiii')
            lr=LogisticRegression(penalty='l1',C=c)
            print('hiii')
            lr.fit(x_t.iloc[[indices[0]],:],y_t.iloc[idx[indices[0]],:])
            
            #predict using test indices
            y_pred_un=lr.predict(x_t.iloc[indices[1],:])
            recall_acc=recall_score(y_t.iloc[indices[1],:],y_pred_un)
            recall_acc.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)
        print('mean recall score ',np.mean(recall_acc))
        results_table[j,'recall_score']=np.mean(recall_acc)
        print('')
        
    #best_c=results_table.loc[results_table['recall_score'].idxmax()]['c_parameter']
   # return best_c


best_c=kfold_printing(x_train_un,y_train_un)
            

    c_para=[{'C':[10]}]
            

from sklearn.model_selection import GridSearchCV
lr=LogisticRegression()

gs=GridSearchCV(estimator=lrf,param_grid=c_para,verbose=5,n_jobs=-1,cv=5)
gs.fit(x_train_un,y_train_un)

gs.best_estimator_
gs.best_params_
gs.best_score_


lrf=LogisticRegression(C=100,penalty='l1',verbose=5)
lrf.fit(x_train_un,y_train_un.ravel())



y_pred_un=lrf.predict(x_test.loc[:,rank[:5]])
    

from sklearn.cross_validation import cross_val_score as cvs
acc=[]
acc=cvs(estimator=lrf,X=x_train_un,y=y_train_un,cv=10)
np.mean(acc)


y_pred_un=lrf.predict(x_test_un)
cnfm=confusion_matrix(y_test,y_pred_un)
    
cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])

(cnfm[0][0]+ cnfm[1][1])/(cnfm[0][0]+cnfm[0][1]+cnfm[1][0]+cnfm[1][1])


# Whole dataset
x_train, x_test, y_train, y_test = train_test_split(xt,y,test_size = 0.3, random_state = 0)
lrf.fit(x_train_un,y_train_un.ravel())

y_pre=lrf.predict(x_test)

cnfm=confusion_matrix(y_test,y_pre)

c_para=[{'C':[0.01,0.001,0.05]}]            

from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(estimator=lrf,param_grid=c_para,verbose=5,n_jobs=-1,cv=5)
gs.fit(x_train,y_train)

gs.best_estimator_
gs.best_params_
gs.best_score_

y_pred=gs.predict(x_test)#this xtest is split of xt
confusion_matrix(y_test,y_pred)

y_pred=lrf.predict(x_test)#this xtest is split of xt

xtesss=xtest

#ACTUAL DATASET
y_pred1=gs.predict(x_test)#..try this fitting

c_para=[{'C':[0.01,0.1,1,10]}]            

#BEFORE NORMALIZATION
gs1=GridSearchCV(estimator=lrf,param_grid=c_para,verbose=5,n_jobs=-1,cv=5)
gs1.fit(xt,y)

xtest=oh.fit_transform(test).toarray()


from sklearn.ensemble import RandomForestClassifier
%timeit
rf=RandomForestClassifier()
para=[{'n_estimators':[500,1000,2000]}]

gs_rf=GridSearchCV(estimator=rf,param_grid=para,verbose=5,n_jobs=-1,cv=5)
gs_rf.fit(x_train_un,y_train_un)
gs_rf.best_params_
gs_rf.best_score_


rff=RandomForestClassifier(n_estimators=1000,n_jobs=-1)
rff.fit(x_train_un,y_train_un)
confusion_matrix(y_test,y_peet)

rank=np.argsort(-rff.feature_importances_)
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train_un.columns, np.argsort(-rff.feature_importances_)):
    feats[feature] = importance #add the name/value pair 



y_pee=rff.predict(x_test_un)
    

y_rf1=rff.predict(xtest)



most_imp_fea=[]
j=0

for i in range (0,11):
    j=rank[i]
    val=x_train_un.columns.values[j]
    most_imp_fea.append(val)
    
xtest=pd.DataFrame(xtest)

x_train_un=x_train_un.iloc[:,rank[:11]]
x_test_un=x_test_un.iloc[:,rank[:11]]
xtest=xtest.iloc[:,rank[:11]]
    
#gs_rf1.fit(x_train,y_train)
y_rf2=gs_rf.predict(xtest)

rff.fit(x_train,y_train)
y_pr=rff.predict(x_test)

cnfm=confusion_matrix(y_test_un,y_pr)

cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])

(cnfm[0][0]+ cnfm[1][1])/(cnfm[0][0]+cnfm[0][1]+cnfm[1][0]+cnfm[1][1])


#xg

import xgboost
from xgboost import XGBClassifier

xg=XGBClassifier()
xg.fit(x_test_un,y_test_un.ravel())

from sklearn.cross_validation import cross_val_score as cvs
acc=[]
acc=cvs(estimator=xg,X=x_test_un,y=y_test_un,cv=10)
acc.mean()


para=[{'learning_rate':[0.5,0.01,0.1,1,10]}]
gs_xg=GridSearchCV(estimator=xg,param_grid=para,verbose=5,n_jobs=-1,cv=5)
gs_xg.fit(x_train_un,y_train_un)
gs_xg.best_score_
gs_xg.best_params_

yt=xg.predict(x_test_un)
ytt=gs_xg.predict(x_test_un)

cnfm=confusion_matrix(y_test_un,y_s)

cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])

(cnfm[0][0]+ cnfm[1][1])/(cnfm[0][0]+cnfm[0][1]+cnfm[1][0]+cnfm[1][1])

y_xg1=xg.predict(xtest)



lrf=LogisticRegression(C=100,penalty='l1',verbose=5)
lrf.fit(x_train_un,y_train_un.ravel())


y_rf3=rff.predict(xtest)

y_f10=lrf.predict(xtest)
y_f2=xg.predict(xtest)
y_f3=gs1.predict(xtest)
y_f4=rff.predict(xtest)



su=pd.DataFrame({'session_id':test_id,'is_click':y_f10})
su.to_csv('f10.csv',index=False)

su=pd.DataFrame({'session_id':test_id,'is_click':y_f2})
su.to_csv('f2.csv',index=False)#0.5004

su=pd.DataFrame({'session_id':test_id,'is_click':y_rf3})
su.to_csv('rf3.csv',index=False)#0.5004


su=pd.DataFrame({'session_id':test_id,'is_click':y_f4})
su.to_csv('rffff.csv',index=False)#0.5004

#STK
y_st_f1=lrf.predict(x_train_un)
y_st_rff=rff.predict(x_train_un)
y_st_xg=xg.predict(x_train_un)

from sklearn.kernel_approximation import RBFSampler
rbf_fea=RBFSampler(gamma=1,random_state=1)
x_fea=rbf_fea.fit_transform(x_train_un)

xtest_fea=rbf_fea.transform(x_test_un)

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(verbose=5,random_state=1,n_iter=5)
sgd.fit(x_train_un,y_train_un)

acc=cvs(estimator=sgd,X=x_fea,y=y_train_un,cv=20)
acc.mean()

sgd.score(x_train_un,y_train_un)#0.9323729578170091

y_s=sgd.predict(x_test_un)

y_st_sgd=sgd.predict(x_train_un)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
x_tra = lda.fit_transform(x_train_un, y_train_un)
x_tes = lda.transform(x_test_un)
lrf.fit(x_tra,y_train_un)
y_lda=lrf.predict(x_tes)
xte=lda.transform(xtest)
y_lda1=lrf.predict(xte)

su=pd.DataFrame({'session_id':test_id,'is_click':y_lda1})
su.to_csv('lda1.csv',index=False)#0.55312

#svc
from sklearn.svm import SVC
svc=SVC()
c_para=[{'C':[0.01,0.1,1,10]}]

gs_svc=GridSearchCV(estimator=svc,param_grid=c_para,verbose=5,n_jobs=-1,cv=5)
gs_svc.fit(x_train_un,y_train_un)            
gs_svc.best_score_
gs_svc.best_params_


svcc=SVC(C=10)
svcc.fit(x_train_un,y_train_un.ravel())
y_sv=svcc.predict(xtest)


cnfm=confusion_matrix(y_test_un,y_sv)

cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])
recall_score(y_test_un,y_sv)

y_svv=svcc.predict(x_test_un)
recall_score(y_test_un,y_svv)

y_sv1=svcc.predict(xtest)#check acc

su=pd.DataFrame({'session_id':test_id,'is_click':y_sv1})
su.to_csv('sv1.csv',index=False)#0.55312

y_st_sv=svcc.predict(x_train)

cnfm=confusion_matrix(y_test_un,y_lda)

cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])

(cnfm[0][0]+ cnfm[1][1])/(cnfm[0][0]+cnfm[0][1]+cnfm[1][0]+cnfm[1][1])




#ADABOOST
from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(n_estimators=100,learning_rate=1,random_state=1)
abc.fit(x_train_un,y_train_un)

c_para=[{'n_estimators':[10,100,500,1000],'learning_rate':[0.1,1,10,100]}]

gs_abc=GridSearchCV(estimator=abc,param_grid=c_para,verbose=5,n_jobs=-1,cv=10)
gs_abc.fit(x_train_un,y_train_un)            
gs_abc.best_score_
gs_abc.best_params_

abc1=AdaBoostClassifier(n_estimators=500,learning_rate=1,random_state=1)
abc1.fit(x_train_un,y_train_un)


acc=[]
acc=cvs(estimator=abc1,X=x_test_un,y=y_test_un,cv=10)
acc.mean()

pr=abc1.predict(x_test_un)
recall_score(y_test_un,pr)

y_adb=abc1.predict(xtest)

su=pd.DataFrame({'session_id':test_id,'is_click':y_adb})
su.to_csv('ada1.csv',index=False)#0.51


xtrain2=pd.DataFrame({'lr':y_st_f1,'xg':y_st_xg,'rf':y_st_rff,'sgd':y_st_sgd,'svc':y_st_sv,'dl':y_st_dl})


y_st_f1_tes=lrf.predict(xtest)
y_st_rff_tes=rff.predict(xtest)
y_st_xg_tes=xg.predict(xtest)
y_st_sgd_tes=sgd.predict(xtest)
y_st_sv_tes=y_sv

xtest2=pd.DataFrame({'lr':y_st_f1_tes,'xg':y_st_xg_tes,'rf':y_st_rff_tes,'sgd':y_st_sgd_tes,'svc':y_st_sv_tes,'dl':y_st_dl_tes})

ytrain2=y_train_un

#without lda

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

def recall_a(y_true,y_pred):
    recall=recall_score(y_true,y_pred)
    return recall
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Fitting the ANN to the Training set
classifier.fit(x_train_un, y_train_un, batch_size = 10, nb_epoch = 100)

dl_pre=classifier.predict(x_test_un)

for i  in range(len(dl_pre)):
    if dl_pre[i]>0.5:
        dl_pre[i]=1
    else:
        dl_pre[i]=0    



recall_score(y_test_un,dl_pre)

y_dl=classifier.predict(xtest)

for i  in range(len(y_dl)):
    if y_dl[i]>0.5:
        y_dl[i]=1
    else:
        y_dl[i]=0    

y_dl=y_dl.reshape(1,-1)
len(y_dl[0])
dl_pred=y_dl[0]
su=pd.DataFrame({'session_id':test_id,'is_click':dl_pred})
su.to_csv('dl1.csv',index=False)

y_st_dl=classifier.predict(x_train_un)

for i  in range(len(y_st_dl)):
    if y_st_dl[i]>0.5:
       y_st_dl[i]=1
    else:
        y_st_dl[i]=0    

y_st_dl=y_st_dl.reshape(1,-1)
y_st_dl=y_st_dl[0]

y_st_dl_tes=classifier.predict(x_test_un)


for i  in range(len(y_st_dl_tes)):
    if y_st_dl_tes[i]>0.5:
       y_st_dl_tes[i]=1
    else:
        y_st_dl_tes[i]=0    

y_st_dl_tes=y_st_dl_tes.reshape(1,-1)
y_st_dl_tes=dl_pred



#model for stststst
xtrain2
xtest2
ytrain2

# linear svc ....then knn ....then lr
svc1=SVC(kernel='linear')
par=[{'C':[0.1,1,10,0.01,20,50]}]

gs_lsv=GridSearchCV(estimator=svc1,param_grid=par,verbose=5,n_jobs=-1,cv=10)
gs_lsv.fit(xtrain2,ytrain2)
            
gs_lsv.best_score_
gs_lsv.best_params_

svc_op=SVC(kernel='linear',C=0.1)
#
#YTRAINN=ytrain2.values.ravel()
#YTRAINN=np.array(YTRAINN)
#YTRAINN[2][1]
#
#=YTRAINN.reshape(-1,1)
#
#
#XTRAINN=xtrain2.values
#XTRAINN=np.array(XTRAINN)
#

#xtr,xts,ytr,yts=train_test_split(XTRAINN,Y=YTRAINN,test_size=0.2,random_state=1)

#svc_op.fit(xtrain2,ytrain2)
#acc=[]
#acc=cvs(estimator=svc_op,X=xts,y=yts,cv=3)
#acc.mean()
#

y_stpred_svc=svc_op.predict(xtest2)

#KNN
from sklearn.neighbors import KNeighborsClassifier as knc
knn=knc()
param=[{'n_neighbors':[240,250,230],'p':[2],'metric':['minkowski']}]
gs_knn=GridSearchCV(estimator=knn,param_grid=param,verbose=5,n_jobs=-1,cv=10)
gs_knn.fit(xtrain2,ytrain2)#try later.....this skips fitting on kfold data

gs_knn.best_params_
gs_knn.best_score_

knnf=knc(n_neighbors=230 ,p=2 ,metric='minkowski')
knnf.fit(xtrain2,ytrain2)

200=59.82
300-59.881
250-59.94

#knnf.fit(xtrain2,ytrain2)
##acc=[]
#acc=cvs(estimator=knnf,X=xts,y=yts,cv=3)
#acc.mean()


y_stpred_knn1=knnf.predict(xtest2)#dont try
y_stpred_knn=gs_knn.predict(xtest2)

#LOGREG
logreg=LogisticRegression()

pars=[{'C':[0.1],'penalty':['l1']}]
gs_log=GridSearchCV(estimator=logreg,param_grid=pars,verbose=5,n_jobs=-1,cv=100)

gs_log.fit(xtrain2,ytrain2)#try later.....this skips fitting on kfold data

gs_log.best_params_
gs_log.best_score_


logreg1=LogisticRegression(C=0.1,penalty='l1')
logreg1.fit(xtrain2,ytrain2)

y_stpred_lr1=logreg1.predict(xtest2)
y_stpred_lr=gs_log.predict(xtest2)


X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.25,random_state=100)

cat_fea=X_train.dtypes.loc[X_train.dtypes=='object'].index
cat_fea_in=[]

for i in cat_fea:
    r=df.columns.get_loc(i)
    cat_fea_in.append(r)
cat_fea_in=np.array(cat_fea_in)

categorical_features_indices = np.where(X_train.dtypes =='object')[0]
#catboost model
from catboost import CatBoostClassifier
cat_model=CatBoostClassifier(n_estimators=1000,eval_metric='AUC',learning_rate=0.05,depth=8)

cat_model.fit(X_train.values,y_train.values,cat_features=cat_fea_in,eval_set=(X_test,y_test)
,plot=False,early_stopping_rounds=100,use_best_model=True)

 # early stopping set to 100 to prevent overfitting

#FEATURE IMPORTANCE
zip(3,2)
#zip returns #a a zip object ..view using sorted mthod
sorted(zip(cat_model.feature_importances_,X_train),reverse=True)
cat_model.best_score_
cat_model.get_params()

#EITHER REMOVE SOME LOW IMPO FEATURES OR PROCEED AS IT IS
pred=cat_model.predict_proba(X_test)
pred=pred[:,1]
from sklearn.metrics import roc_auc_score
pr=cat_model.predict(X_test)

len(pr==0)
len(y_test==0)

from sklearn.metrics import confusion_matrix as cm
com=cm(y_test,pr)
roc=com[0][0]/com[0][0]+com[0][1]
roc_auc_score(y_test,pred)

#FULL DATA
X=df
y=X['is_click']
X.drop(['is_click'],1,inplace=True)
Xtest=dftest

cat_model.fit(X,y,cat_features=cat_fea_in,eval_set=(X,y),early_stopping_rounds=100,use_best_model=True)

#BOOSTING STACKING(STACK MODEL ON SAME CLASSIFEIR MODEL AND TAKE AVG
err=[]
pred_tot=[]

from sklearn.model_selection import KFold,StratifiedKFold
fold=StratifiedKFold(n_splits=4,shuffle=True,random_state=25)
for train_index,test_index in fold.split(X,y):
    x_train,x_test=X.iloc[train_index],X.iloc[test_index]
    ytrain,ytest=y.iloc[train_index],y.iloc[test_index]
    cat_model2=CatBoostClassifier(n_estimators=1000,eval_metric='AUC',learning_rate=0.05,depth=10)
    cat_model2.fit(x_train,ytrain,cat_features=cat_fea_in,eval_set=(x_test,ytest),early_stopping_rounds=50,use_best_model=True)
    predss=cat_model2.predict_proba(x_test)[:,1]
    er=roc_auc_score(ytest,predss)
    print('roc_auc_score is (error score):',er)
    err.append(er)
    p=cat_model2.predict_proba(Xtest)
    p=p[:,1]
    pred_tot.append(p)
    print('appended!!!!!!!')
    

err=np.mean(err)

cat_model2.fit(x_train,y_train)
y_pr=cat_model2.predict(x_test)

cnfm=confusion_matrix(y_test,y_pr)

cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])

(cnfm[0][0]+ cnfm[1][1])/(cnfm[0][0]+cnfm[0][1]+cnfm[1][0]+cnfm[1][1])


y_pred=np.mean(pred_tot,0)
    
len(y_pred[y_pred>0.1])

#SUBMISSION
y_predict=cat_model2.predict_proba(xtest)
y_predict[y_predict>0.1]
su=pd.DataFrame({'session_id':test_id,'is_click':y_predict})
su.to_csv('sol_catboost.csv',index=False)

