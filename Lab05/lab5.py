# 18110053_NguyenQuocBao

## libary import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings filter
from warnings import simplefilter
# Import LabelEncoder function
from sklearn.preprocessing import LabelEncoder
# import scikit-learn Normalizer for scaling data
from sklearn.preprocessing import Normalizer
# Import train_test_split function 
from sklearn.model_selection import train_test_split
# Import metrics to evaluate the perfomance of each model
from sklearn.metrics import accuracy_score
# Import model 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)


## Cleaning data
def remove_outlier(data):
    '''
    If data overcome [Q1-1.5IQR, Q3+1.5IQR], remove them.
    Parameters:    
        data : dataframe need remove outlier
    returns:
        data : dataframe removed outlier
    '''
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))).any(axis = 1)]
    return data

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

# Build model
def Model_LogisticRegression(X_train, Y_train, X_test, Y_test):
    '''
    Build model Machine Learning from data available
    Parameters:
        + X_train : DataFrame
            DataFrame is used to train
        + Y_train : DataFrame
            DataFrame is used to train with label
        + X_test : DataFrame
            DataFrame is used to test
        + y_test : DataFrame
            DataFrame is used to compare with the predicted label
    Returns: float
            This function returns the accuracy between y_test and y_pred
    '''
    LR = LogisticRegression(random_state=58)
    LR.fit(X_train, Y_train)
    y_pred_LR = LR.predict(X_test)
    acc_LR = accuracy_score(y_pred_LR, Y_test)
    return acc_LR

def Model_DecisionTree(X_train, Y_train, X_test, Y_test):
    '''
    Build model Machine Learning from data available
    Parameters:
        + X_train : DataFrame
            DataFrame is used to train
        + Y_train : DataFrame
            DataFrame is used to train with label
        + X_test : DataFrame
            DataFrame is used to test
        + y_test : DataFrame
            DataFrame is used to compare with the predicted label
    Returns: float
            This function returns the accuracy between y_test and y_pred
    '''
    dtree = DecisionTreeClassifier(random_state=58)
    dtree.fit(X_train, Y_train)
    y_pred_tree = dtree.predict(X_test)
    acc_DT = accuracy_score(y_pred_tree, Y_test)
    return acc_DT

def Model_SVC(X_train, Y_train, X_test, Y_test):
    '''
    Build model Machine Learning from data available
    Parameters:
        + X_train : DataFrame
            DataFrame is used to train
        + Y_train : DataFrame
            DataFrame is used to train with label
        + X_test : DataFrame
            DataFrame is used to test
        + y_test : DataFrame
            DataFrame is used to compare with the predicted label
    Returns: float
        This function returns the accuracy between y_test and y_pred
    '''
    svc = SVC(random_state=0, kernel='linear')
    svc.fit(X_train,Y_train)
    y_pred_svc = svc.predict(X_test)
    acc_svc = accuracy_score(y_pred_svc,Y_test)
    return acc_svc

def Model_RandomForest(X_train, Y_train, X_test, Y_test):
    '''
    Build model Machine Learning from data available
    Parameters:
        + X_train : DataFrame
            DataFrame is used to train
        + Y_train : DataFrame
            DataFrame is used to train with label
        + X_test : DataFrame
            DataFrame is used to test
        + y_test : DataFrame
            DataFrame is used to compare with the predicted label
    Returns: float
        This function returns the accuracy between y_test and y_pred
    '''
    RDF = RandomForestClassifier(random_state=23)
    RDF.fit(X_train, Y_train)
    Y_pred_RDF = RDF.predict(X_test)
    acc_RDF = accuracy_score(Y_pred_RDF, Y_test)
    return acc_RDF



# main
def main():
    # ************************************************ #
    '''                 Read Data                     '''
    # ************************************************ #
    path_train = '/home/qbaro/Work/Python_for_scientist/Lab5/Py4DS_Lab5/Py4DS_Lab5_Dataset/titanic_train.csv' 
    path_test = '/home/qbaro/Work/Python_for_scientist/Lab5/Py4DS_Lab5/Py4DS_Lab5_Dataset/titanic_test.csv'
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    # Information of data train
    print("*"*80)
    print(">> Information of data train \n")
    print(train.info())

    # Shape of data train
    print("*"*80)
    print(">> Shape of data train: ", train.shape)

    # Describe Data train
    print("*"*80)
    print(">> Describe data train \n", train.describe())

    # Print head of data 
    print("*"*80)
    print(train.head(3))
    print("\n")

    # Check missing value
    print("*"*80)
    print(">> Check missing value")
    missing_values = train.isnull().sum().sort_values(ascending = False)
    percentage_missing_values = (missing_values/len(train))*100
    print(pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing']))
    '''
    There are many missing value in feature 'Age','Cabin'
    '''

    # fill with mean
    print("*"*80)
    train['Age'] = train['Age'].fillna(train['Age'].mean())
    print(">> After fill with mean \n", train.isna().sum())


    # ************************************************ #
    '''                     EDA                       '''
    # ************************************************ #

    # Draw countplot
    fig, ax = plt.subplots(1,3,figsize=(20,8))
    sns.countplot(x = 'Survived', data = train,ax = ax[0], palette='icefire')
    sns.countplot(x = 'Survived', hue = 'Sex', data = train, ax = ax[1])
    sns.countplot(x = 'Survived', hue = 'Pclass', data = train, ax = ax[2], palette = 'rocket')
    plt.savefig('Countplot')
    '''
        The first chart shows the survivor count imbalance low with 2 value 0(No) , 1(Yes).
        The seconde chart shows Survivor Count based on Sex because of priority for women and children
        , so the number of men alive is low.
        The third chart shows Survivor count based on Class of Passenger, Most passenger in Pclass = 1,2 survived,
        passenger in Pclass = 3 had a majority of the total passengers but most did not survive.
    '''
    # Draw heatmap
    fig = plt.figure(figsize=(13,10))
    sns.heatmap(data = train.corr(),annot=True,cmap='rocket',linewidths=0.2)
    plt.yticks(rotation = 0)
    plt.savefig('heatmap')
    '''
        Most features are backward correlated and independent of each other.
    '''

    # ************************************************ #
    '''                Cleaning data                  '''
    # ************************************************ #
    
    ## Drop feature
    train = train.drop(['Name','PassengerId' ,'Cabin', 'Embarked','Ticket'], axis=1)
    test = test.drop(['Name', 'PassengerId','Cabin', 'Embarked','Ticket'], axis=1)
    '''
        We decide drop this features, because of the following reason:
        Cabin have too many missing values, although it might useful, we can know about information cabin position filled with water first.
        Ticket have too many mixed data type and doesn't help us figure out survival of the passengers.
        Embarked,Name, PassengerId doesn't effect to survival.
    '''

    ## Drop Duplicates
    print("*"*80)
    print(">> Drop Duplicates")
    rows = train.shape[0]
    train.drop_duplicates(subset = train.columns.values[:-1], keep= 'first',inplace = True)
    print(rows-train.shape[0],'duplicated Rows has been removed')
    print("shape after drop duplicates",train.shape)
    
    ## Remove outlier
    print("*"*80)
    print(">> Shape before remove outlier: ", train.shape)
    train = remove_outlier(train)
    print(">> Shape after remove outlier: ", train.shape)

    ## Encoder
    print("*"*80)
    print(">> Encoding data")
    train = Encoder(train)
    print(train.dtypes)
    
    ## Splitting Training data
    X = train.drop(['Survived'],axis=1)
    y = train['Survived']
    X_train, X_test , Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=9)
    # print(">> Shape data before build model ", train.shape)

    # ************************************************ #
    '''               Build Model                     '''
    # ************************************************ #
    print("*"*80)
    ## LogisticRegression
    acc_LR = Model_LogisticRegression(X_train, Y_train, X_test, Y_test)
    print("Logistic Regression gives an Accuracy of ",acc_LR*100,"% on the traing set.")


    ## DecisionTreeClassifier
    acc_DT = Model_DecisionTree(X_train, Y_train, X_test, Y_test)
    print("Decision Tree gives an Accuracy of ",acc_DT*100,"% on the traing set.")

    ## SVC
    acc_svc = Model_SVC(X_train, Y_train, X_test, Y_test)
    print("SVC gives an Accuracy of ",acc_svc*100,"% on the traing set.")

    ## RandomForestClassifier
    acc_RDF = Model_RandomForest(X_train, Y_train, X_test, Y_test)
    print("Random Forest gives an Accuracy of ",acc_RDF*100,"% on the traing set.")
    '''
        Summary:
            Logistic Regression gives an Accuracy of  81.17647058823529 % on the traing set.
            Decision Tree gives an Accuracy of  83.52941176470588 % on the traing set.
            SVC gives an Accuracy of  76.47058823529412 % on the traing set.
            Random Forest gives an Accuracy of  77.64705882352942 % on the traing set.
    '''




    # ************************************************ #
    '''               Test data                    '''
    # ************************************************ #
    print("\n")
    print("="*130)
    print(">> Train with data test")
    
    # Information of data train 
    print("*"*80)
    print(">> Information of data test ")
    print(test.head(3))
    print("*"*80)
    print(test.info())

    # Describe data train 
    print(">> Describe data train ")
    print(test.describe())
    print("*"*80)

    # Shape data test 
    print(">> Shape data test ",test.shape)
    print("*"*80)

    # Check missing value
    print(">> Check missing value")
    print(test.isnull().sum())

    # fill with mean
    print("*"*80)
    test['Age'] = test['Age'].fillna(test['Age'].median())
    test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
    print(">> After fill with median \n", train.isnull().sum())

    # Encoder
    print("*"*80)
    print(">> Encoding data")
    test = Encoder(test)
    print(test.dtypes)
    
    
    # build model
    print("*"*80)
    RDF = RandomForestClassifier(random_state=23)
    RDF.fit(X, y)
    y_pred_RDF = RDF.predict(test)
    print(y_pred_RDF)

    print("*"*80)
    print("score ",RDF.score(X,y))

   
    



    # ## test
    # import operator
    # stats = {}
    # for i in range(1,100):
    #     for j in range(1,200):
    #         X_train, X_test , Y_train, Y_test = train_test_split(train.drop(['Survived'],axis=1),train['Survived'],test_size=0.2,random_state=i)
    #         dtree = SVC(random_state=j,kernel='linear')
    #         dtree.fit(X_train, Y_train)
    #         y_pred_tree = dtree.predict(X_test)
    #         acc_DT = accuracy_score(y_pred_tree, Y_test)
    #         stats[j] = acc_DT
    #         print(stats[j],j,i)
    # print(max(stats.items(), key=operator.itemgetter(1))[0])


if __name__ == "__main__":
    main()

