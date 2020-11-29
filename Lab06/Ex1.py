import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import train_test_split function 
from sklearn.model_selection import train_test_split
# Import LabelEncoder function
from sklearn.preprocessing import LabelEncoder
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'


#****************************************************************#
'''              Dimensionality Reduction                   '''
#****************************************************************#

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

# Quasi-Constant Features in which a value occupies the majority of the records.
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

#****************************************************************#
'''                Correlation Filter Methods                   '''
#****************************************************************#


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
                print(colname)

    x_train.drop(labels=corr_features, axis=1, inplace=True)
    x_test.drop(labels=corr_features, axis=1, inplace=True)
    return x_train,x_test

#****************************************************************#
'''                     Select feature                          '''
#****************************************************************#

# Mutual Information
def Mutual_Information(x_train,y_train,select_k='all',indices=True):
    '''
    Select Features using mutual_info_classif 
      Input:
             train data (pd.Dataframe) 
             test data (pd.Dataframe)
             select_k: int or “all”, optional, default='all'
                        Number of top features to select. The “all” option bypasses selection,
                        for use in a parameter search.
            indices : boolean (default True)
                    If True, the return value will be an array of integers, rather
                    than a boolean mask.
      Output: train data, test data after applying filter methods
    '''
    # import the required functions and object.
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest
    # select the number of features you want to retain.
    select_k = select_k

    # get only the numerical features.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]


    # create the SelectKBest with the mutual info strategy.
    selection = SelectKBest(mutual_info_classif, k=select_k).fit(numerical_x_train, y_train)

    # display the retained features.
    features = x_train.columns[selection.get_support(indices=indices)]
    print(features)

# Chi-squared Score

def Chi_squared_Score(x_train,y_train,select_k='all'):
    '''
    Select Features using chi2
      Input:
             train data (pd.Dataframe) 
             test data (pd.Dataframe)
             select_k: int or “all”, optional, default='all'
                        Number of top features to select. The “all” option bypasses selection,
                        for use in a parameter search.
      Output: train data, test data after applying filter methods
    '''
    # import the required functions and object.
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    # change this to how much features you want to keep from the top ones.
    select_k = select_k

    # apply the chi2 score on the data and target (target should be binary).  
    selection = SelectKBest(chi2, k=select_k).fit(x_train, y_train)

    # display the k selected features.
    features = x_train.columns[selection.get_support()]
    print(features)

# Using SelectFromModel
def select_from_model(X_train, y_train, X_test, y_test):
    '''
    Select Features using model RandomForestClassifier
      Input:
             train data (pd.Dataframe)
                 + X_train : data train
                 + y_train : label train
             test data (pd.Dataframe)
                 + X_test: data test
                 + y_test: label test
      Output: train data, test data after applying filter methods
    '''
    # import the required functions and object.
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    # feature extraction
    select_model = SelectFromModel(rfc)
    # fit on train,test set
    fit_train = select_model.fit(X_train, y_train)
    fit_test = select_model.fit(X_test, y_test)
    # transform train,test set
    model_features_train = fit_train.transform(X_train)
    model_features_test = fit_test.transform(X_train)
    return model_features_train, model_features_test

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
    # transform the data
    x_train = rfe.transform(x_train)
    x_test = rfe.transform(x_test)
    return x_train,x_test

def main():
    data = {'A':[0,0,0,0,0,0,2,1,3,0,0,0,0,0,0],'B':[1,1,1,1,1,0, 2, 2, 3,1,1,0, 1, 4, 3],'C':['a','b','a','b','a','a', 'b', 'b', 'b','a','a','b', 'a', 'a', 'a'],'D':[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]}
    df = pd.DataFrame(data)
    print(">> create data ")
    
    # Shape of data train
    print("*"*80)
    print(">> Shape of data : ", df.shape)
    
    # Print head of data 
    print("*"*80)
    print(df)
    print("\n")
    
    # Drop label
    X = df.drop(columns = ['D'])
    Y = df['D']
    
    # Split train test
    print("*"*80)
    print(">> Split train test \n")
    X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=None)
    print(">> Shape train",X_train.shape)
    print(">> Shape test",X_test.shape)

    # Giving examples to demonstrate your function works
    # Constant_Features
    print("*"*80)
    print(">> Example")
    print(">> Shape train before run :",X_train.shape)
    print(">> Constant_Features ")
    x_train,x_test = Constant_Features(X_train,X_test,0.5)
    print(x_train)
    # print(">> Shape train after run :",x_train.shape)
    
    # Quasi_Constant_Features
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> Quasi_Constant_Features ")
    x_train,x_test = Quasi_Constant_Features(X_train,X_test)
    print(x_train)
    print(">> Shape train after run :",x_train.shape)

    # Duplicated_Features
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> Quasi_Constant_Features ")
    x_train,x_test = Duplicated_Features(X_train,X_test)
    print(x_train)
    print(">> Shape train after run :",x_train.shape)


    # Correlation
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> Quasi_Constant_Features ")
    x_train,x_test = correlation(X_train,X_test)
    print(x_train)
    print(">> Shape train after run :",x_train.shape)


    # Selection feature
    print("*"*80)
    print(">>               Selection feature               ")
    
    print(">> Shape train before run :",X_train.shape)
    print("Mutual_Information")
    # Mutual_Information
    print(">> Shape train before run :",X_train.shape)
    print(">> Mutual_Information")
    Mutual_Information(X_train,Y_train)
    
    # Encode
    print("*"*80)
    print(">> encode")
    label = LabelEncoder()
    Cat_Colums = X_train.dtypes.pipe(lambda X: X[X=='object']).index
    for col in Cat_Colums:
        X_train[col] = label.fit_transform(X_train[col])
        X_test[col] = label.fit_transform(X_test[col])
    print(">> After encode ")
    print(X_train)

    # Chi_squared_Score
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> Chi_squared_Score")
    Chi_squared_Score(X_train,Y_train)

    # select_from_model
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> select_from_model")
    x_train, x_test = select_from_model(X_train,Y_train,X_test,Y_test)
    print(">> after run select_from_model")
    print(">> train:")
    print(x_train)
    print(">> test")
    print(x_test)

    # Principal Component Analysis (PCA)
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> Principal Component Analysis (PCA)")
    print(">> train after run pca \n",pca(X_train,n_components=3))
    print(">> test after run pca \n",pca(X_test,n_components=3))

    # Recursive Feature Elimination (RFE)
    print("*"*80)
    print(">> Shape train before run :",X_train.shape)
    print(">> Recursive Feature Elimination (RFE)")
    x_train, y_train = recursive_rdc(X_train,Y_train,X_test,Y_test)
    print(">> train after run RFE \n")
    print(x_train)
    print(">> test after run RFE \n")
    print(x_test)




    

    




    



if __name__ == "__main__":
    main()
