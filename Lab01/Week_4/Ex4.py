#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load libraries
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


#load dataset
df = pd.read_csv('xAPI-Edu-Data.csv')
df


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
for column in df.columns :
    df[column]= labelencoder.fit_transform(df[column])

print(df.info())
# # Feature Selection

# In[7]:


# split dataset in features and target variable
feature_col = ['gender', 'NationalITy','StageID', 'GradeID','SectionID','raisedhands','VisITedResources','ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'Class']
X = df[feature_col[0:-1]]
y = df[feature_col[-1]]


# # Splitting Data
# 

# In[8]:


# Split dataset into training set and test set 
# 70% training and 30% test
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state =1 )


# # CART
# 

# In[9]:


# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree CLassifier
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset 
y_pred= clf.predict(X_test)


# In[10]:

print('---------------------CART--------------------')
print('CART(Tree Prediction) Accuracy: {}'.format(sum(y_pred== y_test)/len(y_pred)))
print("CART(Tree Prediction) Accuracy by calling metrics",metrics.accuracy_score(y_test,y_pred))


# # Compute the metrics(accuracy score)

# In[11]:

print('---------------------Compute the metrics--------------------')
clf = DecisionTreeClassifier(criterion = 'entropy')
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


# # Random Forest
# 

# In[12]:

print('---------------------Random Forest--------------------')
# Fit Random Forest Classifier
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


# #  Logistic Regression

# In[13]:


print('---------------------Logistic Regression--------------------')
# Fit logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train,y_train)
# Predict testset
y_pred = lr.predict(X_test)
# Evaluate performance of the model 
print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
# Evaluate a score by cross-validation
scores = cross_val_score(lr,X,y,cv=5)
print('scores = {} \n final score = {} \n'.format(scores,scores.mean()))
print('\n')


# In[ ]:




