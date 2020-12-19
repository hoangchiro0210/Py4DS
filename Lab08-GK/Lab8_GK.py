# Library import
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Import train_test_split function 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Import metrics to evaluate the perfomance of each model
from sklearn.metrics import accuracy_score

pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,\
              'display.max_columns', None)
# Function for PCA
def findNumPC(X, thres = 0.95):
    '''
        Find the number of major component vectors
        -----------------------
        Parameters:
            X : numpy-ndarray
                the original matrix 
            thres: float, default = 0.95
                Threshold of information we want to keep from data 
                (by default we keep 95% of the original data)
        -----------------------
        Return:
            the number of major component vectors
    '''
    # Standardize the dataset
    X_std = (X - X.mean(axis = 0))/X.std(axis = 0, ddof = 1)
    # Calculate the covariance matrix for the features in the dataset.
    cov_mat = np.cov(X_std.T, bias = 0)
    # Calculate the eigenvalues and eigenvectors for the covariance matrix.
    eigenvals, eigenvectors = np.linalg.eig(cov_mat)
    eigenvals = sorted(eigenvals,reverse=True)
    # Sum the eigenvalues
    cumsum = np.cumsum(eigenvals)
    # Percentage calculation
    cumsum /= cumsum[-1]
    # To visualize the amount of information obtained for each individual vector
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 8), sharex = True)
    axes[0].plot(range(len(eigenvals)), eigenvals, marker = '.', color = 'b', label = 'Eigenvalue')
    axes[1].plot(range(len(cumsum)), cumsum, marker = '.', color = 'r', label = 'Cumulative propotion')
    axes[0].legend()
    axes[1].legend()
    plt.title('Scree Graph')
    plt.show()

    # Repeats each element in the cumsum, 
    # when the incremental percentage just reaches the threshold, we stop
    for i, val in enumerate(cumsum):
        if val >= thres:
            return i + 1

def PCA_Method(X, n_components=None):
    '''
    Linear dimensionality reduction using Singular Value Decomposition of the data 
    to project it to a lower dimensional space.
    --------------------------    
        Parameters:
            X : numpy-ndarray
                the original matrix 
            n_components: int or None. default = None.
                the number of major component vectors
    --------------------------    
        Return:
            Transform matrix by the matrix of eigenvectors.
    '''
    # Standardize the dataset
    X_std = (X - X.mean(axis = 0))/X.std(axis = 0, ddof = 1)
    # Calculate the covariance matrix for the features in the dataset.
    cov_mat = np.cov(X_std.T, bias = 0)
    # Calculate the eigenvalues and eigenvectors for the covariance matrix.
    eigen_val, eigen_vectors = np.linalg.eig(cov_mat)
    # Pick k eigenvalues and form a matrix of eigenvectors.
    top_eigen_vectors = eigen_vectors[:,:n_components]
    # Transform the original matrix by the matrix of eigenvectors.
    transformed_data = np.matmul(np.array(X_std),top_eigen_vectors)
    return transformed_data

# function main  
def main():
    # Read data
    '''
    Dataset information:
        These data are the results of a chemical analysis of wines grown in the same region in Italy 
        but derived from three different cultivars. The analysis determined the quantities of 13 constituents found 
        in each of the three types of wines.
    '''
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)

    #Rename columns
    df_wine.columns = ['Quatily', 'Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline'] 
    # Print 5 rows on top
    print('\n')
    print('>> 5 rows on top')
    print(df_wine.head())
    print('*'*50)

    #shape of data, 178 rows, 13 features and 1 label
    print('>> Shape of data: ',df_wine.shape)
    print('*'*50)
    # split label
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    # Review the steps of PCA algorithm:
    '''
    1. Standardize the dataset.
    2. Calculate the covariance matrix for the features in the dataset.
    3. Calculate the eigenvalues and eigenvectors for the covariance matrix.
    4. Sort eigenvalues and their corresponding eigenvectors.
    5. Pick k eigenvalues and form a matrix of eigenvectors.
    6. Transform the original matrix by the matrix of eigenvectors.
    '''
    # Standardize the dataset
    X_std = (X - X.mean(axis = 0))/X.std(axis = 0, ddof = 1)
    # For easy visualization, we will use 2 vectors
    X_PCA = PCA_Method(X,2)
    print('>> Shape of Data After Run PCA :', X_PCA.shape)
    print('*'*50)
    
    # Visualize result
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    fig, ax = plt.subplots(figsize = (10, 8))

    for l, c, m in zip(np.unique(y), colors, markers):
        ax.scatter(X_PCA[y == l, 0], X_PCA[y == l, 1], c = c, label = l, marker = m)
    ax.set(xlabel = 'PC 1', ylabel = 'PC 2')
    plt.legend(loc='lower left')
    plt.title('Data After Run PCA')
    plt.savefig('After_PCA')

    # Using library sklearn

    from sklearn.decomposition import PCA
    
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_std)


    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    fig, ax = plt.subplots(figsize = (10, 8))

    for l, c, m in zip(np.unique(y), colors, markers):
        ax.scatter(X_pca[y == l, 0]*(-1), X_pca[y == l, 1], c = c, label = l, marker = m)
    ax.set(xlabel = 'PC 1', ylabel = 'PC 2')
    plt.legend(loc='lower left')
    plt.title('Using library sklearn')
    plt.savefig('sklearn_PCA')

    #*****************************************************
    '''             Effecting when use PCA             '''
    #*****************************************************
    print('>> Before use PCA, Shape of data:', X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    from sklearn.tree import DecisionTreeClassifier
    DT = DecisionTreeClassifier(random_state=1)
    DT.fit(X_train,y_train)
    y_pred = DT.predict(X_test)
    acc = accuracy_score(y_pred,y_test)
    print('Accuracy of Decision Tree ' ,acc)
    print('*'*50)
    # It takes 10 vectors ourselves to retain 95% of the data's information
    print(">> The number of major component vectors retain 95% of the data's information:",findNumPC(X,thres=0.95))
    print('*'*50)
    pca = PCA(n_components = 10)
    X_pca = pca.fit_transform(X_std)
    print('>> After use PCA, Shape of data:', X_pca.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=1)
    DT = DecisionTreeClassifier(random_state=1)
    DT.fit(X_train,y_train)
    y_pred = DT.predict(X_test)
    acc = accuracy_score(y_pred,y_test)
    print('Accuracy of Decision Tree ' ,acc)


if __name__ == "__main__":
    main()