#%% THIS IS UN-EDITED
import pandas as pd
import numpy as np


class DecisionTreeLeaf:

    def __init__(self, value):
        self.value = value

    # This method computes the prediction for this leaf node. This will just return a constant value.
    def predict(self, x):
        return self.value

    # Utility function to draw a tree visually using graphviz.
    def draw_tree(self, graph, node_counter, names):
        node_id = str(node_counter)
        val_str = f'{self.value:.4g}' if isinstance(self.value, float) else str(self.value)
        graph.node(node_id, val_str, style='filled')
        return node_counter+1, node_id
        
    def __eq__(self, other):
        if isinstance(other, DecisionTreeLeaf):
            return self.value == other.value
        else:
            return False
        
class DecisionTreeBranch:

    def __init__(self, feature, threshold, low_subtree, high_subtree):
        self.feature = feature
        self.threshold = threshold
        self.low_subtree = low_subtree
        self.high_subtree = high_subtree

    # For a branch node, we compute the prediction by first considering the feature, and then 
    # calling the upper or lower subtree, depending on whether the feature is or isn't greater
    # than the threshold.
    def predict(self, x):
        if x[self.feature] <= self.threshold:
            return self.low_subtree.predict(x)
        else:
            return self.high_subtree.predict(x)

    # Utility function to draw a tree visually using graphviz.
    def draw_tree(self, graph, node_counter, names):
        node_counter, low_id = self.low_subtree.draw_tree(graph, node_counter, names)
        node_counter, high_id = self.high_subtree.draw_tree(graph, node_counter, names)
        node_id = str(node_counter)
        fname = f'F{self.feature}' if names is None else names[self.feature]
        lbl = f'{fname} > {self.threshold:.4g}?'
        graph.node(node_id, lbl, shape='box', fillcolor='yellow', style='filled, rounded')
        graph.edge(node_id, low_id, 'False')
        graph.edge(node_id, high_id, 'True')
        return node_counter+1, node_id

#%% 
from graphviz import Digraph
from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABC, abstractmethod

class DecisionTree(ABC, BaseEstimator):

    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        
    # As usual in scikit-learn, the training method is called *fit*. We first process the dataset so that
    # we're sure that it's represented as a NumPy matrix. Then we call the recursive tree-building method
    # called make_tree (see below).
    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            self.names = X.columns
            X = X.to_numpy()
        elif isinstance(X, list):
            self.names = None
            X = np.array(X)
        else:
            self.names = None
        Y = np.array(Y)        
        self.root = self.make_tree(X, Y, self.max_depth)
        
    def draw_tree(self):
        graph = Digraph()
        self.root.draw_tree(graph, 0, self.names)
        return graph
    
    # By scikit-learn convention, the method *predict* computes the classification or regression output
    # for a set of instances.
    # To implement it, we call a separate method that carries out the prediction for one instance.
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return [self.predict_one(x) for x in X]

    # Predicting the output for one instance.
    def predict_one(self, x):
        return self.root.predict(x)        

    # This is the recursive training 
    def make_tree(self, X, Y, max_depth):

        # We start by computing the default value that will be used if we'll return a leaf node.
        # For classifiers, this will be the most common value in Y.
        default_value = self.get_default_value(Y)

        # First the two base cases in the recursion: is the training set completely
        # homogeneous, or have we reached the maximum depth? Then we need to return a leaf.

        # If we have reached the maximum depth, return a leaf with the majority value.
        if max_depth == 0:
            return DecisionTreeLeaf(default_value)

        # If all the instances in the remaining training set have the same output value,
        # return a leaf with this value.
        if self.is_homogeneous(Y):
            return DecisionTreeLeaf(default_value)

        # Select the "most useful" feature and split threshold. To rank the "usefulness" of features,
        # we use one of the classification or regression criteria.
        # For each feature, we call best_split (defined in a subclass). We then maximize over the features.
        n_features = X.shape[1]
        _, best_feature, best_threshold = max(self.best_split(X, Y, feature) for feature in range(n_features))
        
        if best_feature is None:
            return DecisionTreeLeaf(default_value)

        # Split the training set into subgroups, based on whether the selected feature is greater than
        # the threshold or not
        X_low, X_high, Y_low, Y_high = self.split_by_feature(X, Y, best_feature, best_threshold)

        # Build the subtrees using a recursive call. Each subtree is associated
        # with a value of the feature.
        low_subtree = self.make_tree(X_low, Y_low, max_depth-1)
        high_subtree = self.make_tree(X_high, Y_high, max_depth-1)

        if low_subtree == high_subtree:
            return low_subtree

        # Return a decision tree branch containing the result.
        return DecisionTreeBranch(best_feature, best_threshold, low_subtree, high_subtree)
    
    # Utility method that splits the data into the "upper" and "lower" part, based on a feature
    # and a threshold.
    def split_by_feature(self, X, Y, feature, threshold):
        low = X[:,feature] <= threshold
        high = ~low
        return X[low], X[high], Y[low], Y[high]
    
    # The following three methods need to be implemented by the classification and regression subclasses.
    
    @abstractmethod
    def get_default_value(self, Y):
        pass

    @abstractmethod
    def is_homogeneous(self, Y):
        pass

    @abstractmethod
    def best_split(self, X, Y, feature):
        pass

#%% HERE THE EDITING STARTS


class TreeRegression(DecisionTree, RegressorMixin):

    def __init__(self, max_depth=10, criterion='var_red'):
        super().__init__(max_depth)
        self.criterion = criterion
        
    def fit(self, X, Y):
        if self.criterion == 'var_red': # ADDED THIS AS THE ONLY CRITERION FUNCTION
            self.criterion_function = variance_reduction_scorer
        else:
            raise Exception(f'Unknown criterion: {self.criterion}')
        super().fit(X, Y)
        self.classes_ = sorted(set(Y))

    # Select a default value that is going to be used if we decide to make a leaf.
    # We will select the most common value.
    def get_default_value(self, Y): # CHANGED THIS --------------------------------------------------
        return np.mean(Y) 
    
    # Checks whether a set of output values is homogeneous. 
    def is_homogeneous(self, Y): # CHANGED THIS -----------------------------------------------------
        return np.var(Y) <= 0 # THIS COULD BE CHANGED
        
    # Finds the best splitting point for a given feature. We'll keep frequency tables (Counters)
    # for the upper and lower parts, and then compute the impurity criterion using these tables.
    # In the end, we return a triple consisting of
    # - the best score we found, according to the criterion we're using
    # - the id of the feature
    # - the threshold for the best split
    def best_split(self, X, Y, feature):

        # Create a list of input-output pairs, where we have sorted
        # in ascending order by the input feature we're considering.
        sorted_indices = np.argsort(X[:, feature])        
        X_sorted = list(X[sorted_indices, feature])
        Y_sorted = list(Y[sorted_indices])

        n = len(Y)
        
        # Keep track of the best result we've seen so far.
        max_score = -np.inf
        max_i = None

        # CREATE TWO SETS
        low_distr = list() # ONE EMPTY
        high_distr = list(Y_sorted) # ONE OF THE WHOLE SET
        
        # GET THEIR VARIANCES
        #low_var = 0 # SINCE EMPTY
        Y_var = np.var(Y) # SAVE IT SO WE DO NOT NEED TO COMPUTE AGAIN
        #high_var = Y_var # SINCE WHOLE SET
        
        # INITIALISE (USED TO REDUCE COMPUTATIONS OF VARIANCE)
        low_sum = 0
        low_sum2 = 0
        high_sum = sum(Y_sorted)
        high_sum2 = sum([n**2 for n in Y_sorted])

        # Go through all the positions (excluding the last position).
        for i in range(0, n-1):

            # Input and output at the current position.
            x_i = X_sorted[i]
            y_i = Y_sorted[i]
            
            # UPPDATE THE SETS
            low_distr.append(y_i) # ADD THE CURRENTS TO THE *EMPTIER* SET
            high_distr.remove(y_i) # REMOVE IT FROM THE *FULLER* SET
            
            # UPDATE THE VARIANCES OF THE TWO SETS
            low_sum += y_i
            low_sum2 += y_i**2
            low_var = low_sum2/(i+1) - (low_sum/(i+1))**2
            #print('Error in Var(low): ', low_var - np.var(low_distr))
            high_sum -= y_i
            high_sum2 -= y_i**2
            high_var = high_sum2/(n-i-1) - (high_sum/(n-i-1))**2
            #print('Error in Var(high): ', high_var - np.var(high_distr))
            #low_var = np.var(low_distr)
            #high_var = np.var(high_distr)

            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            #x_next = XY[i+1][0]
            x_next = X_sorted[i+1] # ABBUNDANT?
            if x_i == x_next:
                continue

            # Compute the homogeneity criterion for a split at this position.
            score = self.criterion_function(i, n, low_var, high_var, Y_var)

            # If this is the best split, remember it.
            if score > max_score:
                max_score = score
                max_i = i

        # If we didn't find any split (meaning that all inputs are identical), return
        # a dummy value.
        if max_i is None:
            return -np.inf, None, None

        # Otherwise, return the best split we found and its score.
        split_point = 0.5*(X_sorted[max_i] + X_sorted[max_i+1])
        return max_score, feature, split_point
    

def variance_reduction_scorer(i, n, low_var, high_var, Y_var): # DEFINED THIS FUNCTION
    return Y_var - (i+1)*low_var/n - (n-i-1)*high_var/n # VARIANCE REDUCTION FUNCTION FROM ASSIGNMENT DESCRIPTION

#%% Generate Data
np.random.seed(0)
def make_some_data(n):
    x = np.random.uniform(-5, 5, size=n)
    Y = (x > 1) + 0.1*np.random.normal(size=n)
    X = x.reshape(n, 1) # X needs to be a 2-dimensional matrix
    return X, Y
    
[X, Y] = make_some_data(100)

import matplotlib.pyplot as plt
plt.plot(X,Y,'.')

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

#%% Applying Method to Generated data
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13]
n = len(max_depth)
test_score = np.zeros(n,)
train_score = np.zeros(n,)

i = 0
best = np.Inf
for dep in max_depth:
    #print(i)
    clf = TreeRegression(max_depth=dep)
    
    clf_train_score = cross_validate(clf, Xtrain, Ytrain, scoring='neg_mean_squared_error')
    train_score[i] = -np.mean(clf_train_score['test_score'])
    #print('train_score: ', train_score[i])
    
    clf.fit(Xtrain, Ytrain)
    test_score[i] = mean_squared_error(Ytest, clf.predict(Xtest))
    #print('test_score: ', test_score[i])

    #print('test_score: ', test_score[i])
    if train_score[i] < best:
        best = train_score[i]
        best_dep = dep
    i += 1

print('Best mean cross validation: ', best)
print('Best max depth: ', best_dep)  



clf = TreeRegression(max_depth=best_dep)
clf.fit(Xtrain, Ytrain)
Yguess = clf.predict(Xtest)
print('mean square error: ',mean_squared_error(Ytest, clf.predict(Xtest)))
clf.draw_tree()
#%% Getting Russian appartments data

# Read the CSV file using Pandas.
alldata = pd.read_csv('C:/Users/kyhng/Desktop/Applied_ML/PA1/sberbank.csv')

# Convert the timestamp string to an integer representing the year.
def get_year(timestamp):
    return int(timestamp[:4])
alldata['year'] = alldata.timestamp.apply(get_year)

# Select the 9 input columns and the output column.
selected_columns = ['price_doc', 'year', 'full_sq', 'life_sq', 'floor', 'num_room', 'kitch_sq', 'full_all']
alldata = alldata[selected_columns]
alldata = alldata.dropna()

# Shuffle.
alldata_shuffled = alldata.sample(frac=1.0, random_state=0)

# Separate the input and output columns.
X = alldata_shuffled.drop('price_doc', axis=1)
# For the output, we'll use the log of the sales price.
Y = alldata_shuffled['price_doc'].apply(np.log)
plt.plot(X,Y,'.')
# Split into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

#%% Applying Method to Generated data
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

m1 = DummyRegressor()
m1_score = cross_validate(m1, Xtrain, Ytrain, scoring='neg_mean_squared_error')
print('dummy score: ', -m1_score['test_score'])

#%%
max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]

i = 0
best = np.Inf
for dep in max_depth:
    print(i)
    i += 1
    clf = TreeRegression(max_depth=dep)
    clf_score = cross_validate(clf, Xtrain, Ytrain, scoring='neg_mean_squared_error')
    mean_score = np.mean(clf_score['test_score'])
    if mean_score < best:
        best = mean_score
        best_dep = dep

print('Best mean mean square error: ', best)
print('Best max depth: ', best_dep)  



clf = TreeRegression(max_depth=best_dep)
clf.fit(Xtrain, Ytrain)
print('mean square error: ',mean_squared_error(Ytest, clf.predict(Xtest)))
clf.draw_tree()


#%% Underfit/overfit
from sklearn.model_selection import cross_validate

max_depth = [1,2,3,4,5,6,7,8,9,10,11,12]
n = len(max_depth)
test_score = np.zeros(n,)
train_score = np.zeros(n,)

i = 0
for dep in max_depth:
    print(i)
    clf = TreeRegression(max_depth=dep)
    
    clf_train_score = cross_validate(clf, Xtrain, Ytrain, scoring='neg_mean_squared_error')
    train_score[i] = -np.mean(clf_train_score['test_score'])
    print('train_score: ', train_score[i])
    
    clf.fit(Xtrain, Ytrain)
    test_score[i] = mean_squared_error(Ytest, clf.predict(Xtest))
    print('test_score: ', test_score[i])

    i += 1

#%%
plt.plot(max_depth,train_score,'.-',color='red')
plt.plot(max_depth,test_score,'.-',color='blue')
plt.legend(['train','test'])

#%%
clf = TreeRegression(max_depth=25)
clf.fit(Xtrain, Ytrain)
score = mean_squared_error(Ytrain, clf.predict(Xtrain))
print('score: ', score)