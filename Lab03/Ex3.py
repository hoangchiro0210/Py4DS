## Library import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import metrics to evaluate the perfomance of each model
from sklearn import metrics

# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#load dataframe
df = pd.read_csv('FIFA2018Statistics.csv')
print(df.head(10))

print(df.describe())

print(df.shape)

numerical_features   = df.select_dtypes(include = [np.number]).columns
categorical_features = df.select_dtypes(include= [np.object]).columns

for i in range(1,26):
    print(df.iloc[:,i].value_counts())
    print("*"*20)

sns.countplot(x="Man of the Match",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))
plt.title('Countplot')
plt.show()
'''
đồi thị label 'Man of the Match' is balanced with 2 values 'Yes' and 'No'
'''
#****************************
### Correlation
plt.figure(figsize=(30,14))
sns.heatmap(df[numerical_features].corr(), annot=True,robust=True, yticklabels=1)
plt.title('Correlation')
plt.show()

print(df.head())

# encode target variable 'Man of the match' into binary format
df['Man of the Match'] = df['Man of the Match'].map({'Yes': 1, 'No': 0})

### Correlation of label

corr = df.corr()
corr = corr.filter(items=['Man of the Match'])
plt.figure(figsize=(15,8))
plt.title('Correlation of label')
sns.heatmap(corr, annot=True)
plt.show()

'''
 'Man of the Match' có hệ số tương quan cao với 'Goal Scored', 'On-Target', 'Corners', 'Attempts', 'free Kicks', 'Yellow Card', 'red', 'Fouls Committed', 'Own goal Time'
'''
### Boxplot
var = ['Man of the Match','Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red',\
       'Fouls Committed']
dummy_data = df[var]
plt.figure(figsize=(22,10))
sns.boxplot(data = dummy_data)
plt.title('Boxplot')
plt.show()

'''
#### Theo như boxplot:

* 1 giá trị ngoại lai trong cột Goal scored
* 2 trong On_Target
* 1 trong Corners
* 2 trong Attempts
* 3 trong Yellow Card
* 1 trong red

##### Số lượng giá trị ngoại lai không quá lớn và vượt quá phạm vi so với tiêu chuẩn nên sẽ không ảnh hưởng nhiều nếu xóa chúng.
'''

## Missing values
missing_values = df.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(df))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])

### Remove Missing Value
'''
Vì 'own Goal Time' và 'Own goals' có hơn 90% giá trị thiếu, điều này sẽ dẫn đến mô hình dự đoán sai hướng. Nên xóa chúng là lựa chọn tốt nhất.
Xóa "Corners', 'Fouls Committed' và 'On-Targets' vì có mức độ tuyến tính thấp và không ảnh hưởng đến việc phân loại.
'''
df.drop(['Own goal Time', 'Own goals', '1st Goal', 'Date', 'Corners','Fouls Committed', 'On-Target'], axis = 1, inplace= True)

### Label Encoder

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Cat_Colums = df.dtypes.pipe(lambda X: X[X=='object']).index
for col in Cat_Colums:
    df[col] = label.fit_transform(df[col])

### Drop Duplicates

rows = df.shape[0]
df.drop_duplicates(subset = df.columns.values[:-1], keep= 'first',inplace = True)
print(rows-df.shape[0],'duplicated Rows has been removed')

print('Check null value :',df.isnull().any().any())

print(df.shape)
print(df.head())


### Remove outlier

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df.shape)
print(df.head())

#split dataset in features and target variable
X = df.drop('Man of the Match',axis = 1)
y = df['Man of the Match']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

### Model, predict and estimate the result:


# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)
# Predict testset
y_pred=rdf.predict(X_test)
# Evaluate performance of the model
print("accuracy RDF before using scaler:  ", metrics.accuracy_score(y_test, y_pred))

### Data RobustScaler

from sklearn.preprocessing import RobustScaler

X_scale = RobustScaler()
X_scale.fit(X_train)
X_train_scaled = X_scale.transform(X_train)
X_scale.fit(X_test)
X_test_scaled = X_scale.transform(X_test)


# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_scaled, y_train)
# Predict testset
y_pred=rdf.predict(X_test_scaled)
# Evaluate performance of the model
print("accuracy RDF using RobustScaler:  ", metrics.accuracy_score(y_test, y_pred))

### Data Normalizer

from sklearn.preprocessing import Normalizer

# normalizer
norm = Normalizer()
norm = norm.fit(X_train)
X_train_norm = norm.transform(X_train)
norm = norm.fit(X_test)
X_test_norm = norm.transform(X_test)


# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_norm, y_train)
# Predict testset
y_pred=rdf.predict(X_test_norm)
# Evaluate performance of the model
print("accuracy RDF using Normalizer:  ", metrics.accuracy_score(y_test, y_pred))

### Data StandardScaler

from sklearn.preprocessing import StandardScaler

St = StandardScaler()
St = St.fit(X_train)
X_train_St = St.transform(X_train)
St = St.fit(X_test)
X_test_St = St.transform(X_test)


# Fit Random Forest Classifier
rdf = RandomForestClassifier(random_state=1)
rdf.fit(X_train_norm, y_train)
# Predict testset
y_pred=rdf.predict(X_test_norm)
# Evaluate performance of the model
print("accuracy RDF using StandardScaler:  ", metrics.accuracy_score(y_test, y_pred))

### Data MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

MM = MinMaxScaler()
MM = MM.fit(X_train)
X_train_MM = MM.transform(X_train)
MM = MM.fit(X_test)
X_test_MM = MM.transform(X_test)


# Fit Random Forest Classifier
rdf = RandomForestClassifier(random_state=1)
rdf.fit(X_train_norm, y_train)
# Predict testset
y_pred=rdf.predict(X_test_norm)
# Evaluate performance of the model
print("accuracy RDF using MinMaxScaler:  ", metrics.accuracy_score(y_test, y_pred))




