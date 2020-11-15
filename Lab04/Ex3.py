# Introduction
'''
This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled
mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field
Guide to North American Mushrooms (1981). Each species is identified as definitely edible,
definitely poisonous, or of unknown edibility and not recommended.
'''

# Import
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split # Import train_test_split function
#import model cluster
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering,MiniBatchKMeans
# Import metrics to evaluate the perfomance of each model
from sklearn.metrics import accuracy_score
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)

# Read dataset
path = "/home/qbaro/Work/Python_for_scientist/Lab4/Py4DS_Lab4_Dataset/mushrooms.csv"
dataset = pd.read_csv(path)
print(dataset.head(10))
print("*"*100)
print("\n")


# Exploring
print("Shape of data: ",dataset.shape,"\n")
print("*"*100)
print("\n")
print("Information of data \n",dataset.info())
print("*"*100)
print("\n")
print("describe data: \n",dataset.describe())
print("*"*100)
print("\n")
dataset=dataset.drop(["veil-type"],axis=1)
'''
From table describe we can see column 'veil-type' have only value 0 
and doesn't tell us anything useful. So we can drop that column.
'''
for i in dataset.columns:
    print("\n*", i, "*")
    print(dataset[i].value_counts())
    print("*"*30)
print("\n")
# Draw Label 'Class' Countplot
fig = plt.Figure()
sns.countplot(x = 'class', data = dataset)
plt.savefig('Countplot_Label_Class.jpg')
'''
This plot shows that label is balanced with 2 value p(poisonous) and e(edible ).
'''
# Draw Countplot feature
plot = dataset[[col for col in dataset.columns]]
f, ax = plt.subplots(ncols = 4, nrows = int(len(plot.columns)/4), figsize=(10,len(plot.columns)/3))
for i, c in zip(ax.flatten(), plot.columns):
    sns.countplot(plot[c], ax = i)
f.tight_layout()
plt.savefig("Countplot_each_column.jpg")
## Missing values
missing_values = dataset.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(dataset))*100
print(pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing']))
dataset = dataset[dataset['stalk-root'] != '?']
'''
Data don't have missing value
'''
print("*"*100)
# Remove outlier
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\n")
print('shape after Remove outlier \n',dataset.shape)
print("*"*100)
print("\n")
### Drop Duplicates

rows = dataset.shape[0]
dataset.drop_duplicates(subset = dataset.columns.values[:-1], keep= 'first',inplace = True)
print(rows-dataset.shape[0],'duplicated Rows has been removed')
print("shape after drop duplicates",dataset.shape)
print("*"*100)


X=dataset.drop(['class'], axis=1)
Y=dataset['class']

# Encoding

labelencoder=LabelEncoder()
for column in X.columns:
    X[column] = labelencoder.fit_transform(X[column])
Y = labelencoder.fit_transform(Y)
print(X.dtypes)
print("\n")


# spliting train-test data

x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2,random_state = 0)


# Model, predict and estimate the result: 
# MiniBatchKMeans model

#Initialize the model
MBK = MiniBatchKMeans(n_clusters=2,random_state=391)
#Fit our model on the X dataset
MBK.fit(x_train)
#Calculate which mushrooms fall into which clusters
y_prep = MBK.predict(x_test)
print("accuracy of MiniBatchKMeans:", accuracy_score(y_test, y_prep))

# KMEANS model
KM = KMeans(n_clusters=2,random_state=390)
KM.fit(x_train)
y_prep = KM.predict(x_test)
print("accuracy of KMeans:", accuracy_score(y_test, y_prep))

# Agglomerative model 
app = AgglomerativeClustering(n_clusters=2,linkage= "average")
app.fit(x_train)
y_prep =app.fit_predict(x_test)
print("accuracy of Agglomerative:", accuracy_score(y_test, y_prep))
'''
Summary:
    accuracy of MiniBatchKMeans: 0.6172539489671932
    accuracy of birch: 0.637910085054678
    accuracy of Agglomerative: 0.755771567436209
'''







