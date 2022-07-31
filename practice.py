#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import pandas as pd
from pandas import plotting
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
#get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
#init_notebook_mode(connected=True)  
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix

import xgboost as xgb #決策樹
import lightgbm as  lgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer
# auxiliary function
from sklearn.preprocessing import LabelEncoder
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color



import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = sns.load_dataset('iris')
df.head()


# # EDA
# 

# In[3]:


df.info()
df.shape, df.columns


# In[4]:


df.describe()


# In[5]:


Species = df['species'].unique()
Species


# In[6]:


corr = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'species']].corr()


plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True)


# In[7]:


sns.pairplot(df,hue = 'species',size=1.5)


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='sepal_length',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='sepal_width',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='petal_length',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='petal_width',data=df)


# In[ ]:


Profile = pp.ProfileReport(df, explorative=True)
Profile


# In[8]:


#import sweetviz as sv
#report = sv.analyze(df)
#report.show_html(filepath='iris.html')


# # Machine learning

# In[9]:


X_iris = df.drop('species',axis = 1) #特徵矩陣
X_iris.shape


# In[10]:


y_iris = df['species'].values#目標矩陣
y_iris.shape


# In[11]:


encoder = LabelEncoder()
y_iris = encoder.fit_transform(y_iris)
y_iris


# In[12]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest, = train_test_split(X_iris,y_iris,test_size = 0.3, random_state = 101)
#Xtrain,Xtest,Ytrain,Ytest

#X_train 與 y_train 是實際參與行訓練的資料。
#而 X_test 與 y_test是未參與訓練的資料，它是被拿來測試評估最終訓練好的模型。


# # logistic regression

# In[13]:


lr_model = LogisticRegression()
lr_model.fit(Xtrain,Ytrain)
lr_predict = lr_model.predict(Xtest)

print('Logistic Regression - ',accuracy_score(lr_predict,Ytest))
cm = confusion_matrix(Ytest, lr_predict)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # naive bayes

# In[14]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain,Ytrain)
y_model = model.predict(Xtest)
print('naive bayes - ',accuracy_score(lr_predict,Ytest))

cm = confusion_matrix(Ytest, y_model)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # PCA

# In[15]:


from sklearn.decomposition import PCA
model = PCA (n_components = 2) #主成分兩個
model.fit(X_iris)
X_2D = model.transform(X_iris)#將資料轉化為二維
#X_2D


# In[16]:


df["PCA1"] = X_2D[:,0]
df["PCA2"] = X_2D[:,1]
sns.lmplot("PCA1","PCA2", hue="species", data = df, fit_reg=False)#,fit_reg=False


# In[17]:


df.head()


# # Gaussian mixture model

# In[19]:


from sklearn.mixture import GaussianMixture as GMM
model = GMM(n_components = 3,covariance_type = 'full') #主成分三個
model.fit(X_iris)#訓練
y_gmm = model.predict(X_iris)#Y_GMM = 訓練後的數據集
y_gmm


# In[20]:


df['cluster'] = y_gmm
sns.lmplot("PCA1","PCA2", hue="species", col = 'cluster', data = df, fit_reg=False)


# # SVM 

# In[21]:


svm_model = SVC(kernel='linear')
svm_model.fit(Xtrain,Ytrain)
svc_predict = svm_model.predict(Xtest)

print('SVM - ',accuracy_score(svc_predict,Ytest))

cm = confusion_matrix(Ytest, svc_predict)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # Decision tree

# In[22]:


dt_model = DecisionTreeClassifier(max_leaf_nodes=3)
dt_model.fit(Xtrain,Ytrain)
dt_predict = dt_model.predict(Xtest)

print('Decision Tree - ',accuracy_score(dt_predict,Ytest))

cm = confusion_matrix(Ytest,dt_predict)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # Random forest 

# In[23]:


rfc_model = RandomForestClassifier(max_depth=3)
rfc_model.fit(Xtrain,Ytrain)
rfc_predict = rfc_model.predict(Xtest)


print('random forest ',accuracy_score(rfc_predict,Ytest))
cm = confusion_matrix(Ytest,rfc_predict)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # Extra Tree Classifier

# In[24]:


etc_model = ExtraTreesClassifier()
etc_model.fit(Xtrain,Ytrain)
etc_predict = etc_model.predict(Xtest)

print('Extra Tree Classifier - ',accuracy_score(etc_predict,Ytest))
cm = confusion_matrix(Ytest,etc_predict)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # KNN

# In[25]:


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(Xtrain,Ytrain)
knn_predict = knn_model.predict(Xtest)

print('knn - ',accuracy_score(knn_predict,Ytest))

cm = confusion_matrix(Ytest, knn_predict)
sns.heatmap(cm, annot=True,cmap = 'Blues')


# # XGBoost

# In[27]:


xg_model = xgb.XGBClassifier()
xg_model = xg_model.fit(Xtrain,Ytrain)
print('XGBoost -',xg_model.score(Xtest, Ytest))


# In[2]:


#get_ipython().run_line_magic('history', '')


# In[ ]:




