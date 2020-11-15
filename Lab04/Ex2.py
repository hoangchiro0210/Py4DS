#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn.model_selection import KFold,cross_val_score
plt.style.use('ggplot')


# In[2]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# In[3]:


df = pd.read_csv('diabetes.csv')
df.head()


# ## 1.EDA

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


for i in range(1,len(df.columns)):
    print(df.iloc[:,i].value_counts())


# In[9]:


#histograms for each variable in df
hist = df.hist(bins=10,figsize =(10,10))


# In[10]:


sns.pairplot(df, hue = 'Outcome')


# In[11]:


plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths= .1, cmap= 'YlGnBu', annot = True)
plt.yticks(rotation = 0)
plt.show()


# In[12]:


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[13]:


Result = sns.countplot(x='Outcome', data = df, linewidth  = 2 ,edgecolor = sns.color_palette('dark'))


# In[14]:


plt.subplots(figsize = (20,8))
df['Age'].value_counts().sort_index().plot.bar()
plt.title('No. of times vs No. of students raised their hands on particular time')
plt.xlabel('No. of times,students raised their hands', fontsize =14)
plt.ylabel('No. of students, on particular times',fontsize = 14)
plt.show()


# In[15]:


Raised_hand = sns.boxplot(x= 'Outcome', y = 'Age',data =df)
plt.show()


# In[16]:


df.groupby(['Age'])['Outcome'].value_counts()


# In[17]:


plt.subplots(figsize = (20,8))
sns.countplot(x = 'Age',data = df, hue ='Outcome', palette= 'bright')
plt.show()


# ## 2. DATA CLEANING

# ### a) Removing Missing values

# In[18]:


df.isnull().any()
# Dữ liệu không có missing values nên ta không cần phải xử lí


# ### b) Drop duplicate values

# In[19]:


rows = df.shape[0]
df.drop_duplicates(subset = df.columns.values[:-1], keep= 'first',inplace = True)
print(rows-df.shape[0],'duplicated Rows has been removed')


# ### c) Removing Outliers

# In[20]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[21]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(data = df,linewidth = 2.5, width = 0.50)
plt.show()


# In[22]:


df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape


# In[23]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(data = df,linewidth = 2.5, width = 0.50)
plt.show()


# In[24]:


#Splitting the data into dependent and independent variables
y = df.Outcome
X = df.drop('Outcome', axis = 1)
def train_test_data(X,y,test_sizes,random_states):
    X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size = test_sizes,random_state = random_states)
    return X_train ,X_test , y_train, y_test
X_train ,X_test , y_train, y_test = train_test_data(X,y,0.25,1)


# In[25]:



X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 1)


# In[26]:



def Robust_Scaler(X_train,X_test):
    from sklearn.preprocessing import RobustScaler
    X_scale = RobustScaler()
    X_scale.fit(X_train)
    X_train_scaled = X_scale.transform(X_train)
    X_scale.fit(X_test)
    X_test_scaled = X_scale.transform(X_test)
    return X_train_scaled,X_test_scaled

def Kmean_accuracy(X_train,X_test,y_test,random_states):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state = random_states)
    kmeans.fit(X_train)
    y_pred=kmeans.predict(X_test)
    return  metrics.accuracy_score(y_test, y_pred)
def Birch_accuracy(X_train,X_test,y_test,bf):
    from sklearn.cluster import Birch
    brc = Birch(n_clusters=2,branching_factor=bf )
    brc.fit(X_train)
    y_pred= brc.fit_predict(X_test)
    return  metrics.accuracy_score(y_test, y_pred)
def MiniBatch_Accuracy(X_train,X_test,y_test,random_states):
    from sklearn.cluster import MiniBatchKMeans
    Mnb = MiniBatchKMeans(n_clusters=2,random_state = random_states )
    Mnb.fit(X_train)
    y_pred= Mnb.fit_predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)



# In[27]:


X_train ,X_test , y_train, y_test = train_test_data(X,y,0.25,42)
X_train_scaled,X_test_scaled = Robust_Scaler(X_train,X_test)


# In[28]:


print('Kmeans:',Kmean_accuracy(X_train,X_test,y_test,81))
print('MiniBatch accuracy:',MiniBatch_Accuracy(X_train_scaled,X_test_scaled,y_test,81))
print('Birch Accuracy:',Birch_accuracy(X_train_scaled,X_test_scaled,y_test,81))

# -------summary-----
# Kmeans: 0.4125
# MiniBatch accuracy: 0.8125
# Birch Accuracy: 0.675


# In[ ]:




