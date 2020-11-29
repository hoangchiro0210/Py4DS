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
# import warnings filter
from warnings import simplefilter, filterwarnings
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'

# Constant Features that show single values in all the observations in the dataset. 
#    These features provide no information that allows ML models to predict the target.
def Constant_Features(x_train, x_test,threshold=0):
    """
    Removing Constant Features using Variance Threshold
    Input: threshold parameter to identify the variable as constant
         train data (pd.Dataframe) 
         test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # import and create the VarianceThreshold object.
    from sklearn.feature_selection import VarianceThreshold
    vs_constant = VarianceThreshold(threshold=threshold)

    # select the numerical columns only.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]

    # fit the object to our data.
    vs_constant.fit(numerical_x_train)

    # get the constant colum names.
    constant_columns = [column for column in numerical_x_train.columns
                        if column not in numerical_x_train.columns[vs_constant.get_support()]]

    # detect constant categorical variables.
    constant_cat_columns = [column for column in x_train.columns
                            if (x_train[column].dtype == "O" and len(x_train[column].unique())  == 1 )]

    # concatenating the two lists.
    all_constant_columns = constant_cat_columns + constant_columns

    # drop the constant columns
    x_train = x_train.drop(labels=all_constant_columns, axis=1, inplace=True)
    x_test = x_test.drop(labels=all_constant_columns, axis=1, inplace=True)
    return x_train, x_test

def correlation(x_train,x_test,method='pearson',min_periods = 1):
    '''
    Compute pairwise correlation of columns, excluding NA/null values.
    Then Removing high feature correlation 
    Input:
        train data (pd.Dataframe) 
        test data (pd.Dataframe)
        method: {‘pearson’, ‘kendall’, ‘spearman’}. Default 'pearson'
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
        min_periods : int 
            Minimum number of observations required per pair of columns to have a valid result.
            Currently only available for pearson and spearman correlation. default = 1
    Output: train data, test data after applying filter methods
    '''
    # creating set to hold the correlated features
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = x_train.corr(method=method)

    # optional: display a heatmap of the correlation matrix
    plt.figure(figsize=(11,11))
    sns.heatmap(corr_matrix,annot=True)
    plt.savefig("heatmap")

    for i in range(len(corr_matrix .columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
    print(corr_features)

    x_train.drop(labels=corr_features, axis=1, inplace=True)
    x_test.drop(labels=corr_features, axis=1, inplace=True)
    return x_train,x_test

# Duplicated Features: features of the same geometry type that are collocated and optionally share attributes
def Duplicated_Features(x_train,x_test):
    """
      Removing Duplicated Features 
      Input:
             train data (pd.Dataframe) 
             test data (pd.Dataframe)
      Output: train data, test data after applying filter methods
    """
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

# Principal Component Analysis (PCA)
def pca(X,n_components=None,random_state = None):
    '''
    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    The input data is centered but not scaled for each feature before applying the SVD
    Input:
        X: data (pd.Dataframe) 
        n_components: int, float, None or str, default = None
            Number of components to keep
        random_state: int, RandomState instance, default=None
            Pass an int for reproducible results across multiple function calls
    Output:
        X data after applying filter methods
    '''
    # import the required functions and object.
    import numpy as np
    from sklearn.decomposition import PCA
    # define model
    pca = PCA(n_components=n_components)
    # fit on train,test set
    pca.fit(X)
    X = pca.transform(X)
    return X

# Recursive Feature Elimination (RFE)
def recursive_rdc(x_train,y_train,x_test,y_test,n_features_to_select=None):
    '''
    Select Features using recursive model RandomForestClassifier
      Input:
             train data (pd.Dataframe)
                 + X_train : data train
                 + y_train : label train
             test data (pd.Dataframe)
                 + X_test: data test
                 + y_test: label test
            n_features_to_select : int or None (default=None)
                    The number of features to select. If `None`, half of the features
                    are selected.
      Output: train data, test data after applying filter methods
    '''
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    rfe = RFE(estimator=rfc, n_features_to_select=n_features_to_select)
    # fit the model
    rfe.fit(x_train, y_train)
    rfe.fit(x_test,y_test)
    # transform the data
    x_train = rfe.transform(x_train)
    x_test = rfe.transform(x_test)
    return x_train,x_test
# Quasi Constant_Features
def Quasi_Constant_Features(x_train, x_test,threshold=0.98):
    """
      Removing Quasi-Constant Features with Threshold
      Input: threshold parameter to identify the variable as constant. Default = 0.98
             train data (pd.Dataframe) 
             test data (pd.Dataframe)
      Output: train data, test data after applying filter methods
    """
    # create empty list
    quasi_constant_feature = []

    # loop over all the columns
    for feature in x_train.columns:

        # calculate the ratio.
        predominant = (x_train[feature].value_counts() / np.float(len(x_train))).sort_values(ascending=False).values[0]

        # append the column name if it is bigger than the threshold
        if predominant >= threshold:
            quasi_constant_feature.append(feature)   
        
    print(quasi_constant_feature)
    
    # drop the quasi constant columns
    x_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    x_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    
    return x_train,x_test
def AUC_Score(X_train,y_train,X_test,y_test):
    '''
    calculated accuracy 
    Input:
             train data (pd.Dataframe)
                 + X_train : data train
                 + y_train : label train
             test data (pd.Dataframe)
                 + X_test: data test
                 + y_test: label test
    Output:
            accuracy 
    '''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=12)
    model.fit(X_train,y_train)
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_pred,y_test)
    return acc

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

# main
def main():
    # ************************************************ #
    '''                 Read Data                     '''
    # ************************************************ #
    path = '/home/qbaro/Work/Python_for_scientist/Lab6/Py4DS_Lab6/Py4DS_Lab6_Dataset/data.csv'
    data = pd.read_csv(path,index_col=0,nrows=1000)
    # Information of data 
    print("*"*80)
    print(">> Information of data train \n")
    print(data.info())

    # Shape of data train
    print("*"*80)
    print(">> Shape of data : ", data.shape)

    # Describe Data train
    print("*"*80)
    print(">> Describe data train \n", data.describe())

    # Print head of data 
    print("*"*80)
    print(data.head(3))
    print("\n")

    # Check missing value
    print("*"*80)
    print(">> Check missing value")
    missing_values = data.isnull().sum().sort_values(ascending = False)
    percentage_missing_values = (missing_values/len(data))*100
    print(pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing']))
    '''
    There are many missing value 
    '''
    print("Shape before drop missing value: ", data.shape)
    data = data.drop(['Loaned From'],axis = 1)
    data = data.dropna()
    print("Shape after drop missing value: ", data.shape)

     ## Encoder
    print("*"*80)
    print(">> Encoding data")
    data = Encoder(data)
    print(data.dtypes)

    # split label
    X = data.drop(['International Reputation'], axis=1)
    y = data['International Reputation']

    # Split train test
    print("*"*80)
    print(">> Split train test \n")
    X_train,X_test,Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    print(">> Shape train",X_train.shape)
    print(">> Shape test",X_test.shape)

    # teachnical Feature Selection vs Dimensionality Reduction
    print ('----------------------PCA---------------')
    X_train_PCA = pca(X_train,n_components=len(X_test.columns))
    X_test_PCA = pca(X_test,n_components=len(X_test.columns))
    print(">> Shape after run pca",X_train_PCA.shape)
    print(">> Shape after run pca",X_test_PCA.shape)
    print('>> Accuracy: ',AUC_Score(X_train,Y_train,X_test,Y_test))

    print ('----------------------Duplicate FEATURES---------------')
    X_train_D, X_test_D = Duplicated_Features(X_train,X_test)
    print(">> Shape after run ",X_train_D.shape)
    print(">> Shape after run ",X_test_D.shape)
    print('>> AUC Score: ',AUC_Score(X_train_D,Y_train,X_test_D,Y_test))

    print ('----------------------Quasi_Constant_Features---------------')
    X_train_Q, X_test_Q =  Quasi_Constant_Features(X_train, X_test,threshold=0.98)
    print(">> Shape after run ",X_train_Q.shape)
    print(">> Shape after run ",X_test_Q.shape)
    print('>> AUC Score: ',AUC_Score(X_train_Q,Y_train,X_test_Q,Y_test))
    
    print ('----------------------Correlation FEATURES---------------')
    X_train_c, X_test_c = correlation(X_train,X_test)
    print(">> Shape after run ",X_train_c.shape)
    print(">> Shape after run ",X_test_c.shape)
    print('>> AUC Score: ',AUC_Score(X_train_c,Y_train,X_test_c,Y_test))




if __name__ == "__main__":
    main()