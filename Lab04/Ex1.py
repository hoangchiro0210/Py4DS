# Import
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Import train_test_split function
#import model cluster
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
# Import metrics to evaluate the perfomance of each model
from sklearn.metrics import accuracy_score
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)

# Read datasset
path = '/home/qbaro/Work/Python_for_scientist/Lab4/Py4DS_Lab4_Dataset/spam.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.replace(' ','')
print(df.head(10))

# Exploring
print("Shape of data: ",df.shape,"\n")
print("*"*100)
print("\n")
print("Information of data \n",df.info())
print("*"*100)
print("\n")
print("describe data: \n",df.describe())
print("*"*100)
print("\n")


## Missing values
missing_values = df.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(df))*100
print(pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing']))
'''
Data don't have missing value
'''
print("*"*100)
# Draw Label 'spam' Countplot
fig = plt.Figure()
sns.countplot(x = '1', data = df)
plt.savefig('Countplot_label.png')
'''
This plot shows that label isn't balanced with 2 value 0 and 1.
'''
# Draw boxplot
boxplot = df[[col for col in df.columns]]
f, ax = plt.subplots(ncols = 4, nrows = int(len(boxplot.columns)/4), figsize=(10,len(boxplot.columns)/3))
for i, c in zip(ax.flatten(), boxplot.columns):
    sns.boxplot(boxplot[c], ax = i)
f.tight_layout()
plt.savefig("Boxplot_each_column_before_remove_outliers.jpg")


### Drop Duplicates

rows = df.shape[0]
df.drop_duplicates(subset = df.columns.values[:-1], keep= 'first',inplace = True)
print(rows-df.shape[0],'duplicated Rows has been removed')
print("shape after drop duplicates",df.shape)
print("*"*100)
# Remove outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\n")
print('shape after Remove outlier \n',df.shape)
print("*"*100)
print("\n")

X=df.drop(['1'], axis=1)
Y=df['1']

# spliting train-test data

x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2,random_state = 1)


# Model, predict and estimate the result:
 
# KMEANS model
#Initialize the model
kmeans = KMeans(n_clusters=2,random_state=0)
#Fit our model on the X dataset
kmeans.fit(x_train)
#Calculate which mushrooms
y_prep = kmeans.predict(x_test)
print("accuracy of k-means:", accuracy_score(y_test, y_prep))

#BIRCH model
birch = Birch(n_clusters=2)
birch.fit(x_train)
y_prep = birch.predict(x_test)
print("accuracy of birch:", accuracy_score(y_test, y_prep))

# Agglomerative model 
app = AgglomerativeClustering(n_clusters=2,linkage= "average")
app.fit(x_train)
y_prep =app.fit_predict(x_test)
print("accuracy of Agglomerative:", accuracy_score(y_test, y_prep))
'''
Summary:
    accuracy of k-means: 0.8695652173913043
    accuracy of birch: 0.8695652173913043
    accuracy of Agglomerative: 0.8695652173913043
'''