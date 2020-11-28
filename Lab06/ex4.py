

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

X_train = pd.read_csv('Santander_train.csv',nrows=35000)
X_test = pd.read_csv('Santander_test.csv',nrows=15000)

X_train.head()

print(X_train.shape)
print(X_test.shape)

X_train.drop(labels=['TARGET'], axis=1, inplace = True)



def Constant_Features(x_train, x_test,threshold=0):

  from sklearn.feature_selection import VarianceThreshold
  sel = VarianceThreshold(threshold=0)
  sel.fit(x_train)  # fit finds the features with zero variance
  x_train = sel.transform(x_train)
  x_test = sel.transform(x_test)
  return  x_train,x_test

def Quasi_ConstantFeatures(x_train,x_test):
    threshold=0.98
    # create empty list
    quasi_constant_feature = []

    # loop over all the columns
    for feature in x_train.columns:

        # calculate the ratio.
        predominant = (x_train[feature].value_counts() / np.float(len(x_train))).sort_values(ascending=False).values[0]
        
        # append the column name if it is bigger than the threshold
        if predominant >= threshold:
            quasi_constant_feature.append(feature)   
            
    # print(quasi_constant_feature)

    # drop the quasi constant columns
    x_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    x_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    return x_train,x_test

def duplicate_feature(x_train,x_test):
  # transpose the feature matrice
  train_features_T = x_train.T

  # print the number of duplicated features
  print(train_features_T.duplicated().sum())

  # select the duplicated features columns names
  duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
  # drop those columns
  
  x_train.drop(labels=duplicated_columns, axis=1, inplace=True)
  x_test.drop(labels=duplicated_columns, axis=1, inplace=True)
  return x_train,x_test

def RecursiveFE(x_train,y_train):
  from sklearn.feature_selection import RFE 
  from sklearn.ensemble import RandomForestClassifier
  # define model
  rfc = RandomForestClassifier(n_estimators=100)
  rfe = RFE(estimator=rfc, n_features_to_select=3)
  # fit the model
  rfe.fit(x_train, y_train)
  # transform the data
  x_train, y_train = rfe.transform(x_train, y_train)
  x_test, y_test = rfe.transform(x_test, y_test)
  return x_train,x_test 

def mutual_info(x_train,x_test):
  # creating set to hold the correlated features
  corr_features = set()

  # create the correlation matrix (default to pearson)
  corr_matrix = x_train.corr()

  # optional: display a heatmap of the correlation matrix
  # plt.figure(figsize=(11,11))
  # sns.heatmap(corr_matrix)

  for i in range(len(corr_matrix .columns)):
      for j in range(i):
          if abs(corr_matrix.iloc[i, j]) > 0.8:
              colname = corr_matrix.columns[i]
              corr_features.add(colname)
              
  x_train.drop(labels=corr_features, axis=1, inplace=True)
  x_test.drop(labels=corr_features, axis=1, inplace=True)
  return x_train,x_test

def PCA_(x_train):
  import numpy as np
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(x_train)
  x_train = pca.transform(x_train)
  return x_train

def AUC_Score(X_train,y_train):
  from sklearn.model_selection import KFold
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score
  model = RandomForestClassifier()
  kfold = KFold(n_splits=2, random_state=42) 
  scores = cross_val_score(model, X_train_C, y_train, cv=kfold, scoring='roc_auc')
  return  scores.mean()

X_train = pd.read_csv('Santander_train.csv',nrows=35000)
X_test = pd.read_csv('Santander_test.csv',nrows=15000)
y_train = X_train['TARGET']
X_train.drop(labels=['TARGET'], axis=1, inplace = True)

from sklearn.naive_bayes import GaussianNB

print ('----------------------CONSTANT FEATURES---------------')
X_train_C, X_test_C = Constant_Features(X_train,X_test)
print(X_train_C.shape)
print(X_test_C.shape)
X_train_C, X_test_C = Constant_Features(X_train,X_test)
print( 'AUC Score: ',AUC_Score(X_train_C,y_train))


print ('----------------------Quasi_Constant FEATURES---------------')
X_train_Q, X_test_Q  = Quasi_ConstantFeatures(X_train,X_test)

print(X_train_Q.shape)
print(X_test_Q.shape)
print( 'AUC Score: ',AUC_Score(X_train_Q,y_train))

print ('----------------------duplicate FEATURES---------------')
X_train_D, X_test_D = duplicate_feature(X_train,X_test)
print(X_train_D.shape)
print(X_test_D.shape)
print( 'AUC Score: ',AUC_Score(X_train_D,y_train))

print ('----------------------Mutual Info FEATURES---------------')
X_train_M, X_test_M = mutual_info(X_train,X_test)
print(X_train_M.shape)
print(X_test_M.shape)
print( 'AUC Score: ',AUC_Score(X_train_M,y_train))

print ('----------------------PCA---------------')
X_train_PCA = PCA_(X_train)
print(X_train_PCA.shape)
print( 'AUC Score: ',AUC_Score(X_train_PCA,y_train))