#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import sciki-learn metrics module for accuracy calculation
from sklearn.model_selection import KFold,cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# In[2]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[3]:


df = pd.read_csv('mushrooms.csv')
df.head()


# The target label used in the dataset is class. This contains the data about whether a mushroom is edible or not. The target label is stored in another variable and is dropped from the dataframe

# In[4]:


y = df['class']
X = df.drop('class', axis = 1)
y.head()


# In[5]:


df.dtypes


# In[6]:


from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
for column in df.columns :
    df[column]= labelencoder.fit_transform(df[column])


# In[7]:


df.dtypes


# In[8]:


df.head()


# In[9]:


df.describe()


# # Quick look at the characteristics of the data 

# The violin plot below represents the distribution of the classification characteristics. It is possible to see that 'gill-color' property of the mushroom breaks to 2 part , one below 3 and one above 3 , that may contribute the classification 

# In[10]:


df_div = pd.melt(df,'class',var_name = 'Characteristics')
fig, ax = plt.subplots(figsize = (10,5))
p = sns.violinplot(ax = ax , x = 'Characteristics', y = 'value', hue = 'class', split = True , data = df_div,inner = 'box')
df_no_class = df.drop(['class'], axis = 1)
p.set_xticklabels(rotation = 90 , labels = list(df_no_class.columns))


# Is the data balanced ?

# In[11]:


plt.figure()
pd.Series(df['class']).value_counts().sort_index().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('class')
plt.title('Number of posionous/ edible mushrooms ( 0 = edible, 1 = posionous)')


# The dataset is balanced 

# Let's look at the correlation between the variables

# In[12]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1 , cmap = 'YlGnBu', annot = True)
plt.yticks(rotation = 0)


# ## 2.Feature Selection 

# In[13]:


# # Split dataset in features and target variable 
y = df['class']  #Target variable
X = df.drop('class', axis = 1) # Features
y.head()


# ## 3. Splitting Data

# In[14]:


# Split dataset into training set and test set 
# 70% training and 30% test
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state =1 )


#  ## 4. CART

# In[15]:


# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree CLassifier
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset 
y_pred= clf.predict(X_test)


# In[16]:

print('--------------------CART-----------------------')
print('CART(Tree Prediction) Accuracy: {}'.format(sum(y_pred== y_test)/len(y_pred)))
print("CART(Tree Prediction) Accuracy by calling metrics",metrics.accuracy_score(y_test,y_pred))


# ## 5. Compute the metrics(accuracy score)

# In[17]:

print('-------------------Compute the metrics(accuracy score)-----------------------')
clf = DecisionTreeClassifier()
# Fit decision tree classifier
clf = clf.fit(X_train,y_train)
# Predict testset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print('SVM Accuracy : ', metrics.accuracy_score(y_test,y_pred))
print('\n')
# Evaluate a score by cross validation
scores = cross_val_score(clf,X,y,cv =5)
print('scores = {} \n final score = {} \n '.format(scores,scores.mean()))
print('\n')


# ## 6. Random Forest

# In[18]:


# Fit Random Forest Classifier
print('--------------------Random Forest-----------------------')
rdf = RandomForestClassifier()
rdf.fit(X_train,y_train)
# Predict testset
y_pred = rdf.predict(X_test)
#Evaluate performance of the model
print('RDF : ',metrics.accuracy_score(y_test,y_pred))
print('\n')
# Evaluate a score by cross- validation
scores = cross_val_score(rdf,X,y,cv=5)
print('scores = {} \n final score = {} \n '.format(scores,scores.mean()))
print('\n')


# ## 7.Logistic Regression

# In[19]:


print('--------------------Logistic Regression-----------------------')
# Fit logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train,y_train)
# Predict testset
y_pred = lr.predict(X_test)
# Evaluate performance of the model 
print('LR: ', metrics.accuracy_score(y_test,y_pred))
# Evaluate a score by cross-validation
scores = cross_val_score(lr,X,y,cv=5)
print('scores = {} \n final score = {} \n'.format(scores,scores.mean()))
print('\n')


# In[ ]:





# In[ ]:




