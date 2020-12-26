import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.stats import loguniform
from pandas import read_csv,set_option
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)

# Encode
def Encoder(data):
    '''
    Encode target data
    Parameters:
        data: dataframe nead encode
    Returns:
        data: datafram encoded
    '''
    labelencoder=LabelEncoder()
    data_colums = data.dtypes.pipe(lambda X: X[X=='object']).index
    for column in data_colums:
        data[column] = labelencoder.fit_transform(data[column])
    return data



# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
print('*'*60)
print("The sonar dataset is a standard machine learning dataset comprising \
208 rows of data with 60 numerical input variables and a target variable with \
two class values, e.g. binary classification. The dataset involves predicting \
whether sonar returns indicate a rock or simulated mine.")
print('>> Shape of data: ',dataframe.shape)
print('*'*60)
print('>> Information of data: \n')
print(dataframe.info())
print('*'*60)

print('>> Check null in data \n',dataframe.isnull().sum())
print('>> Check duplicate in data :',dataframe.duplicated().sum())
print('*'*60)

# Grid Search


# split into input and output elements
dataframe = Encoder(dataframe)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
# define model
model = SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('>>Before tuning:')
print(classification_report(y_test,y_pred))
'''accuracy : 0.79 '''
print('*'*60)
# define model
model = SVC()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['kernel'] = ['linear', 'poly', 'rbf']
space['gamma'] = ['scale', 'auto']
space['shrinking'] = [True, False]
space['C'] = [15,30,40]
# define search
search = GridSearchCV(model,space ,scoring='accuracy', cv=cv)
# execute search    
result = search.fit(X_train, y_train)
# summarize result
print('>> After tuning')
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
# prediction
pred = search.predict(X_test)
print(classification_report(y_test,pred))
'''accuracy: 0.84'''
