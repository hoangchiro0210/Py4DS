#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
from sklearn import preprocessing
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


df = pd.read_csv('AB_NYC_2019.csv')
df.head()


# In[4]:


pd.set_option('precision',3)
df.describe()


# In[5]:


#heatmap 
corr = df.corr()
plt.figure(figsize = (12,10))
heat = sns.heatmap(data =corr)
plt.title('Heatmap of Correlation')
# Dựa vào hình dưới , ta nhận thấy rằng 'host_id' , 'id' ,number_of_reviews có độ tương quan cao


# In[6]:


df.drop(['id','name','host_name','last_review'], axis=1, inplace=True)
df.head()
# Loại bỏ các cột dư thừa bên trong feature


# In[ ]:





# In[7]:


df.isnull().any()


# In[8]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(13,7))
plt.title("Neighbourhood Group")
g = plt.pie(df.neighbourhood_group.value_counts(), labels=df.neighbourhood_group.value_counts().index,autopct='%1.1f%%', startangle=180)
plt.show()

# Biểu đồ tròn dưới cho ta thấy 2 khu vực Brooklyn và Mahhatan chiếm đa số


# In[9]:


plt.figure(figsize=(13,7))
plt.title("Type of Room")
sns.countplot(df.room_type, palette="muted")
fig = plt.gcf()
plt.show()

# Các kiểu phòng ở NYC 
# như đã thấy thì private room và entire home chiếm da số 


# In[10]:


plt.style.use('classic')
plt.figure(figsize=(13,7))
plt.title("Neighbourhood Group vs. Availability Room")

sns.boxplot(x='neighbourhood_group',y='availability_365',data=df)

plt.show()

# mối quan hệ giữa cột neighbourhood_group và  availability_365 


# In[11]:


plt.style.use('classic')
plt.figure(figsize=(13,7))
plt.title("Neighbourhood Group Price Distribution < 500")
sns.boxplot(y="price",x ='neighbourhood_group' ,data = df[df.price<500])
plt.show()

# ở đồ thị dưới ta nhận thấy có rất nhiều outlier ở các feature này 


# In[12]:


# plt.figure(figsize=(13,7))
# plt.title("Map of Price Distribution")
ax=df[df.price<500].plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4)
ax.set_title('Map of Price Distribution')
ax.legend()
# plt.ioff()
plt.show()
# Dưới đây là bản đồ phân bố giá của từng khu vực 


# In[13]:


corr = df.corr(method='kendall')
plt.figure(figsize=(13,10))
plt.title("Correlation Between Different Variables\n")
sns.heatmap(corr, annot=True)
plt.show()


# In[14]:


encode = preprocessing.LabelEncoder()


# In[15]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# In[16]:


df.dtypes


# In[17]:


y=df['price']
X=df.drop('price',axis=1)
y.head()


# In[18]:


X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 1)


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)


# In[20]:


predicts = l_reg.predict(X_test)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


# In[21]:


df.reviews_per_month.fillna(0, inplace=True)


# In[22]:


df = df.dropna() 

#Then we drop all prices that are equal to 0
df = df[df.price != 0]


# In[23]:


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]


# In[24]:


to_drop


# ## Remove duplicate Rows

# In[25]:


rows = df.shape[0]
df.drop_duplicates(subset = df.columns.values[:-1], keep= 'first',inplace = True)
print(rows-df.shape[0],'duplicated Rows has been removed')


# In[26]:


df.isnull().any().any()


# ## Remove outliers

# In[27]:


# z = np.abs(stats.zscore(df['price'])) 
# df = df[z<3]
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[28]:


df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape


# In[29]:


df


# ## Data MinMaxScaler

# In[30]:


# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_MM = norm.transform(X_train)

# transform testing datas
X_test_MM = norm.transform(X_test)


# In[31]:
print('***********************MinMaxScaler*****************************')

predicts = l_reg.predict(X_test_MM)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


# ## Data Standardization

# In[32]:


from sklearn.preprocessing import StandardScaler  

st_x= StandardScaler()  
X_train_Stan= st_x.fit_transform(X_train) 
X_test_Stan= st_x.transform(X_test) 


# In[33]:

print('***********************StandardScaler*****************************')
predicts = l_reg.predict(X_test_Stan)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


# ## Data Normalization

# In[34]:


from sklearn.preprocessing import Normalizer
# normalizer
norm = Normalizer()
norm = norm.fit(X_train)
X_train_norm = norm.transform(X_train)
norm = norm.fit(X_test)
X_test_norm = norm.transform(X_test)


# In[35]:

print('***********************Normalizer*****************************')
predicts = l_reg.predict(X_test_norm)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


# In[ ]:
from sklearn.preprocessing import RobustScaler
X_scale = RobustScaler()
X_scale.fit(X_train)
X_train_scaled = X_scale.transform(X_train)
X_scale.fit(X_test)
X_test_scaled = X_scale.transform(X_test)


print('***********************RobustScaler*****************************')
predicts = l_reg.predict(X_test_scaled)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


