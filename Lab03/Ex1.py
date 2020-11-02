#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
# Import libraries for cross validation
from sklearn.model_selection import KFold,cross_val_score
plt.style.use('ggplot')



# In[2]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[3]:


df = pd.read_csv('creditcard.csv')
df


# In[4]:


df.tail()


# In[5]:


pd.set_option('precision',3)
df.describe()


# In[6]:


plot = sns.countplot('Class', data=df)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()
# Dựa vào hình dưới ta nhận thấy rằng data không hề cân bằng nên vì vậy ta cần balance chúng lại


# In[7]:


fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(len(fraud) * 5)
non_fraud.reset_index(drop=True, inplace=True)
fraud.reset_index(drop=True, inplace=True)
df= pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)
df.describe()

# Làm cân bằng data  


# In[8]:


print('Distribution of the Classes in the subsample dataset')
print(df['Class'].value_counts()/len(df))
sns.countplot('Class', data=df)
plt.title('Equally Distributed Classes', fontsize=14)


# Đồ thị sau khi cân bằng 


# In[9]:


df.shape


# In[10]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])




plt.show()
# 2 đồ thị dưới cho ta thấy được số tiền giao dịch và thời điểm giao dịch của data
# Đối với hình 1 , ta thấy rằng đồ thị này lệch phải (tức là có độ lệch dương), cho thấy lượng tiền giao dịch càng lớn thì sẽ không được thực hiện nhiều
# Đối với hình 2 , thì ta thấy đây là 1 đồ thị đa đỉnh , cho thấy thời điểm xảy ra giao dịch. Trong khoảng Time [0,20000] và [90000,110000] sẽ ít giao dịch hơn thông thường 


# In[11]:


#heatmap 
corr = df.corr()
plt.figure(figsize = (12,10))
heat = sns.heatmap(data =corr)
plt.title('Heatmap of Correlation')
# Đồ thị để nhận biết độ tương quan giữa các features


# In[12]:


df.Class.value_counts()


# In[13]:


X = df.drop(['Class'],axis = 1)
y= df['Class']


# In[14]:


X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 1)


# ##  Model, predict and estimate the result:

# In[15]:


# clf = DecisionTreeClassifier(criterion='entropy')
clf = DecisionTreeClassifier()
# Fit Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict testset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))


# In[16]:


0.5
# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)
# Predict testset
y_pred=rdf.predict(X_test)
# Evaluate performance of the model
print("RDF, accuracy:  ", metrics.accuracy_score(y_test, y_pred))


# ## Drop high correlation features

# In[17]:


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]


# In[18]:





# In[19]:


df.drop(df[to_drop], axis=1, inplace = True)


# In[20]:


df


# ## Remove duplicate Rows 

# In[21]:


rows = df.shape[0]
df.drop_duplicates(subset = df.columns.values[:-1], keep= 'first',inplace = True)
print(rows-df.shape[0],'duplicated Rows has been removed')


# In[22]:


df.isnull().any().any()


# ##  Remove outliers

# In[23]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# 
# 

# In[24]:


df


# In[25]:


df.shape


# In[26]:


df.describe()


# In[27]:


X = df.drop(['Class'],axis = 1)
y= df['Class']


# In[28]:


X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 1)


# ## Data Normalization

# In[29]:


from sklearn.preprocessing import Normalizer
# normalizer
norm = Normalizer()
norm = norm.fit(X_train)
X_train_norm = norm.transform(X_train)
norm = norm.fit(X_test)
X_test_norm = norm.transform(X_test)


# In[30]:


df.describe()


# ## Model, predict and estimate the result:

# In[31]:




# clf = DecisionTreeClassifier(criterion='entropy')
clf = DecisionTreeClassifier()
# Fit Decision Tree Classifier
clf = clf.fit(X_train_norm, y_train)
# Predict testset
y_pred = clf.predict(X_test_norm)
# Evaluate performance of the model
print('**********************Normalization*********************')
print("CART (Tree Prediction) Accuracy :  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))


# 


# In[32]:


# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_norm, y_train)
# Predict testset
y_pred=rdf.predict(X_test_norm)
# Evaluate performance of the model
print("RDF, accuracy:  ", metrics.accuracy_score(y_test, y_pred))


# # Data Standardization

# In[33]:


from sklearn.preprocessing import StandardScaler  

st_x= StandardScaler()  
X_train_Stan= st_x.fit_transform(X_train) 
X_test_Stan= st_x.transform(X_test) 


# In[34]:


# clf = DecisionTreeClassifier(criterion='entropy')
clf = DecisionTreeClassifier()
# Fit Decision Tree Classifier
clf = clf.fit(X_train_Stan, y_train)
# Predict testset
y_pred = clf.predict(X_test_Stan)
# Evaluate performance of the model
print('********************Standardization***********************')
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))


# In[35]:



# Fit Random Forest Classifier
rdf = RandomForestClassifier(random_state=1)
rdf.fit(X_train_Stan, y_train)
# Predict testset
y_pred=rdf.predict(X_test_Stan)
# Evaluate performance of the model
print("RDF, accuracy:  ", metrics.accuracy_score(y_test, y_pred))


# ## Data MinMax Scaler

# In[36]:


from sklearn.preprocessing import MinMaxScaler


# In[37]:


MM = MinMaxScaler()
MM = MM.fit(X_train)
X_train_MM = MM.transform(X_train)
MM = MM.fit(X_test)
X_test_MM = MM.transform(X_test)


# In[38]:


# clf = DecisionTreeClassifier(criterion='entropy')
clf = DecisionTreeClassifier()
# Fit Decision Tree Classifier
clf = clf.fit(X_train_MM, y_train)
# Predict testset
y_pred = clf.predict(X_test_MM)
# Evaluate performance of the model
print('***************MinMaxScaler*******************')
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))


# In[39]:


# Fit Random Forest Classifier
rdf = RandomForestClassifier(random_state=1)
rdf.fit(X_train_MM, y_train)
# Predict testset
y_pred=rdf.predict(X_test_MM)
# Evaluate performance of the model
print("RDF, accuracy:  ", metrics.accuracy_score(y_test, y_pred))
# In[40]:
from sklearn.preprocessing import RobustScaler
X_scale = RobustScaler()
X_scale.fit(X_train)
X_train_scaled = X_scale.transform(X_train)
X_scale.fit(X_test)
X_test_scaled = X_scale.transform(X_test)


clf = DecisionTreeClassifier()
# Fit Decision Tree Classifier
clf = clf.fit(X_train_scaled, y_train)
# Predict testset
y_pred = clf.predict(X_test_scaled)
# Evaluate performance of the model
print('***************RobustScaler*******************')
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))


# In[39]:


# Fit Random Forest Classifier
rdf = RandomForestClassifier(random_state=1)
rdf.fit(X_train_scaled, y_train)
# Predict testset
y_pred=rdf.predict(X_test_scaled)
# Evaluate performance of the model
print("RDF, accuracy:  ", metrics.accuracy_score(y_test, y_pred))
# In[40]:








