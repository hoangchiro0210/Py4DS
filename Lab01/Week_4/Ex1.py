#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


dataset_pd = pd.read_csv('spam.csv')


# In[3]:


dataset_np = np.genfromtxt('spam.csv', delimiter = ',')


# In[4]:


print(dataset_pd.shape)
print(dataset_np.shape)


# In[5]:


dataset_pd.head(5)


# In[6]:


dataset_np[0:5, :]


# In[7]:


dataset_np[:, 0:5]


# In[8]:


X= dataset_np[:,:-1]
y= dataset_np[:,-1]


# In[9]:


X


# In[10]:


print(X.shape)
print(y.shape)


# In[11]:


print(X[0:5,:])
print(y[0:5])
print(y[-5:])


# # Split the train and testsets 

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)


# In[14]:


print(X_train,'\n\n', X_test)
print(y_train,'\n\n', y_test)


# In[15]:


# Import Ml Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# In[16]:


# Import metrics to evaluate the performance of each model 
from sklearn import metrics 


# In[17]:


# Import libraries for cross validation
from sklearn.model_selection import KFold,cross_val_score


# # CART

# In[18]:


clf = DecisionTreeClassifier()


# ### Train the model by fit 
# ### Test after training 
# ### compute the metrics (accuracy score )

# In[19]:


# Fit decision tree classifier 
clf = clf.fit (X_train,y_train)


# In[20]:


# Predict testset
y_pred = clf.predict(X_test)


# In[ ]:





# In[21]:


# Evaluate performance on the model
print('--------------------CART-----------------------')
print('CART(Tree Prediction) Accuaracy : {}'.format(sum(y_pred == y_test)/len(y_pred)))
print('CART(Tree Prediction) Accuaracy by calling metrics:',metrics.accuracy_score(y_test,y_pred))


# In[22]:


# clf = DecisionTreeClassifier(criterion = 'entropy')
clf = DecisionTreeClassifier()


# ## Train the model by .fit
# ## Test after training
# ## Compute the metrics (accuracy score)
# 

# In[23]:





# Fit Decision Tree Classifier
print('-------------------Compute the metrics (accuracy score)------------------------')
clf = clf.fit(X_train,y_train)
# Predict testset
y_pred = clf.predict(X_test)
#Evaluate performance of the model
print('SVM Accuracy : ', metrics.accuracy_score(y_test,y_pred))
print('\n')
# Evaluate a score by cross.validation
scores = cross_val_score(clf,X,y,cv =5)
print('scores = {} \n final score = {} \n'.format(scores, scores.mean()))
print('\n')


# # Random Forest
# 

# In[24]:


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


# # Logistic Regression
# 

# In[25]:




# Fit logistic Regression Classifier
print('--------------------Logistic Regression-----------------------')
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




