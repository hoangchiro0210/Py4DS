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


df = pd.read_csv('xAPI-Edu-Data.csv')
df.head()


# ## 1.EDA

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.describe(include ='object')


# In[8]:


df.describe()


# In[9]:


for i in range(1,len(df.columns)):
    print(df.iloc[:,i].value_counts())


# In[10]:


#histograms for each variable in df
hist = df.hist(bins=10,figsize =(10,10))
plt.tight_layout()
plt.savefig('Histogram of 4 features')


# In[ ]:





# In[11]:


sns.pairplot(df, hue = 'Class')
plt.tight_layout()
plt.savefig('[Pairplot of 4 features')

# ở đồ thị này, class L có đồ thị lệch phải , class H có đồ thị lệch trái cho thấy rằng các chỉ số về class L đều thấp hơn class M và H 


# In[12]:


plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths= .1, cmap= 'YlGnBu', annot = True)
plt.yticks(rotation = 0)

plt.tight_layout()
plt.savefig('Heatmap of 4 features')

# ở đồ thị này ta thấy rằng chỉ số tương quan giữa các cột đều không cao


# In[13]:


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[14]:


df = df.drop(to_drop,axis =1 )


# In[15]:


Result = sns.countplot(x='Class', data = df, linewidth  = 2 ,edgecolor = sns.color_palette('dark'))

# ở đồ thị này ta thấy rằng số học sinh thuộc clss M chiếm phần lớn , còn lại là class L,H
# Dữ liệu khá cân bằng 


# In[17]:


plt.subplots(figsize = (20,8))
df['raisedhands'].value_counts().sort_index().plot.bar()
plt.title('No. of times vs No. of students raised their hands on particular time')
plt.xlabel('No. of times,students raised their hands', fontsize =14)
plt.ylabel('No. of students, on particular times',fontsize = 14)
plt.tight_layout()
plt.savefig('Raise hand vs times')


# In[18]:


Raised_hand = sns.boxplot(x= 'Class', y = 'raisedhands',data =df)
plt.tight_layout()
plt.savefig('Boxplot of class and Raised hands')


# In[19]:


plt.subplots(figsize = (20,8))
sns.countplot(x = 'StudentAbsenceDays',data = df, hue ='Class', palette= 'bright')
plt.legend()
plt.tight_layout()
plt.savefig('Hist of Student Absence days')


# ## 2,Data Cleaning

# In[20]:


from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
for column in df.columns :
    df[column]= labelencoder.fit_transform(df[column])


# ### a) Removing Missing values

# In[21]:


df.isnull().sum()
# Dữ liệu không có missing values nên ta không cần phải xử lí


# ### b) Drop duplicate values

# In[22]:


rows = df.shape[0]
df.drop_duplicates(subset = df.columns.values[:-1], keep= 'first',inplace = True)
print(rows-df.shape[0],'duplicated Rows has been removed')


# ### c) Removing Outliers

# In[23]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[24]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(data = df,linewidth = 2.5, width = 0.50)
plt.tight_layout()
plt.savefig('Boxplot before removing outliers')


# In[25]:


row = df.shape[0]
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

print(rows-df.shape[0],'Outliers has been removed')


# In[26]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(data = df,linewidth = 2.5, width = 0.50)
plt.tight_layout()
plt.savefig('Boxplot after removing outliers')


# In[27]:


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
to_drop


# In[28]:


df = df.drop(to_drop,axis = 1)


# In[ ]:





# In[29]:


def train_test_data(X,y,test_sizes,random_states):
    X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size = test_sizes,random_state = random_states)
    return X_train ,X_test , y_train, y_test
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
    kmeans = KMeans(n_clusters=3, random_state = random_states)
    kmeans.fit(X_train)
    y_pred=kmeans.predict(X_test)
    return  metrics.accuracy_score(y_test, y_pred)
def Birch_accuracy(X_train,X_test,y_test,b_factor):
    from sklearn.cluster import Birch
    brc = Birch(n_clusters=3,branching_factor=b_factor )
    brc.fit(X_train)
    y_pred= brc.fit_predict(X_test)
    return  metrics.accuracy_score(y_test, y_pred)
def MiniBatch_Accuracy(X_train,X_test,y_test,random_states):
    from sklearn.cluster import MiniBatchKMeans
    Mnb = MiniBatchKMeans(n_clusters=3,random_state = random_states )
    Mnb.fit(X_train)
    y_pred= Mnb.fit_predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)


# In[30]:


y = df.Class
X = df.drop('Class', axis = 1)
X_train ,X_test , y_train, y_test = train_test_data(X,y,0.25,1)


# In[31]:



X_train_scaled,X_test_scaled = Robust_Scaler(X_train,X_test)


# In[ ]:





# In[32]:


X_train ,X_test , y_train, y_test = train_test_data(X,y,0.25,92)

X_train_scaled,X_test_scaled = Robust_Scaler(X_train,X_test)


# In[33]:


print('Minibatch accuracy:',MiniBatch_Accuracy(X_train_scaled,X_test_scaled,y_test,58))
print('Birch accuracy:',Birch_accuracy(X_train_scaled,X_test_scaled,y_test,58))
print('Kmeans accuracy:',Kmean_accuracy(X_train_scaled,X_test_scaled,y_test,58))

#-------SUMMARY-------
'''
Minibatch accuracy: 0.7415730337078652
Birch accuracy: 0.5280898876404494
Kmeans accuracy: 0.2696629213483146
'''


# In[ ]:




