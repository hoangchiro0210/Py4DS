#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import sciki-learn metrics module for accuracy calculation
import eli5 #Calculating and Displaying importance using the eli5 library
from eli5.sklearn import PermutationImportance
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


# # Loading data 
# ## load the required PIma Indian Diabetes dataset using pandas' read CSV function

# In[3]:


col_names= ['Pregnancies', 'Glucose', 'Bp','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
#load dataset
df = pd.read_csv('diabetes.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# # 2.Feature Selection
# ## Dividing given columns into 2 types of variable dependent( or target variable) and independent variable( or feature variable)

# In[6]:


# split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin','BMI','Age','Glucose', 'BloodPressure','DiabetesPedigreeFunction']
X = df[feature_cols] # Features
y = df.Outcome # Target variable


# In[ ]:





# # 3. Splitting Data 
# 
# ### To understand model performance ,dividing the datasets into a training set and test set in a good strategy
# ### Split the dataset by using function 

# In[7]:


# Split dataset into training set and test set 
# 70% training and 30% test
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state =1)


# # 4. Building Decision Tree Model  
# ### Create a Decision Tree Model using Scikit-learn

# In[8]:


# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree CLassifier
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset 
y_pred= clf.predict(X_test)


# ### Return an explanation of estimator parameters (weights). Use this function to show classifier weights. 

# In[9]:


perm = PermutationImportance(clf,random_state=1).fit(X_test, y_test)
eli5.show_weights(perm,feature_names = X_test.columns.tolist())


# # 5. Evaluating Model
# ### Let's estimate, how accurately the classifier or model can predict the type of cultivars.
# ### Accuracy can be computed by comparing actual test set values and predicted values.
# 

# In[10]:

print('--------------------CART-----------------------')
# Model Accuracy , how often is the classifier collect ?
print('Accuracy :', metrics.accuracy_score(y_test,y_pred))


# # You can improve this accuaracy by tuning the parametersin the Decision Tree Algorithm
# 

# # 6. Visualizing Decision Trees

# In[11]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus


# In[12]:


dot_data = StringIO()


export_graphviz(clf, out_file=dot_data,
filled=True, rounded=True,
special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())


# # 7. Random Forest

# In[13]:


# Fit Random Forest Classifier
print('\n')
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


# # 8.Logistic Regression

# In[14]:

print('\n')
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




