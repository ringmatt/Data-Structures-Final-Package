#* ClassifierAlgorithm.py
#*
#* ANLY 555 Spring 2022
#* Final Project
#*
#* Due on: 2022-03-22
#* Author(s): Matt Ring
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*

import pandas as pd
import numpy as np
import re
from scripts.TreeNodes import *

class ClassifierAlgorithm:
    '''
    The fundamental classifier object. 
    Creates the foundations for training and testing classifier algorithms.
    '''
    
    def __init__(self):
        '''
        Instantiates the ClassifierAlgorithm object.
        '''
        
    def train(self):
        '''
        Trains the classification model
        '''

        # Tests whether the method works
        # REMOVE IN FUTURE UPDATES!
        print("train method invoked\n")
        
    def test(self):
        '''
        Tests the classification model.
        '''
        
        # Tests whether the method works.
        # REMOVE IN FUTURE UPDATES!
        print("test method invoked\n")

class simpleKNNClassifier(ClassifierAlgorithm):
    '''
    Creates a basic KNN classifer algorithm which can be trained and tested.
    '''
    
    def __init__(self, k = 5, distanceMetric = "euclidean"):
        '''
        Instantiates the KNN classifier class.

        Parameters
        ----------
        k : int
            Number of closest training instances to use when predicting an instances class

        distanceMetric : str
            String representing what distance metric to use. Either "euclidean" or "mahalanobis".
        '''

        # Check the type of k and the distanceMetric are correct
        if not isinstance(k, int):
            raise TypeError("'k' must be an integer.")
        elif not isinstance(distanceMetric, str):
            raise TypeError("'distanceMetric' must be of type str")

        # Raise an error if an incorrect data metric is provided
        if distanceMetric not in ["euclidean", "mahalanobis"]:
            raise ValueError("'distanceMetric' must be either 'euclidean' or 'mahalanobis'.")

        # Save the knn-specific member attributes
        self.k = k
        self.distanceMetric = distanceMetric

    def train(self, trainingData, trueLabels):
        '''
        Stores the data and labels.

        Parameters
        ----------
        trainingData : Pandas DataFrame
            A dataframe of all features used to for prediction.

        trueLabels : str
            A string defining the column storing the labels.
        '''

        # Check if the parameters are of the correct type
        if not isinstance(trainingData, pd.DataFrame):
            raise TypeError("The training data must be a Pandas DataFrame.")
        elif not isinstance(trueLabels, str):
            raise TypeError("'trueLabels' must be a string.")

        # Check to ensure the labels exist within the dataset
        if trueLabels not in trainingData.columns.tolist():
            raise IndexError("'trueLabels' must be the name of a column in the training data.")

        # Ensure the training data's indices have been reset
        trainingData = trainingData.reset_index(drop = True)
        
        # Save to member attributes
        self._train_df = trainingData.drop([trueLabels], axis = 1)
        self._labels = trainingData[trueLabels]
        
    def test(self, testData):
        '''
        Returns the most common label of the k closest training instances.

        Parameters
        ----------
        testData : Pandas DataFrame
            Set of observations in the same format as the training data.
            Will be labeled based on nearby training instances. Must not
            include a column for labels.

        Returns
        -------
        predicted_labels : list
            A list of the most common label among training neighbors for each test instance.
        '''
        
        # Check the type of the test data
        if not isinstance(testData, pd.DataFrame):
            raise TypeError("The test data must be a Pandas DataFrame.")

        # Ensure that the indices for the test data have been reset
        testData = testData.reset_index(drop = True)

        # Ensure that the test and training data have the same features
        if set(self._train_df.columns) != set(testData.columns):
            raise ValueError("The test and train data do not have the same columns")

        # Calculate the distances for both training and testing data using either Mahalanobis or Euclidean distance
        if self.distanceMetric == "euclidean":

            distances_train = self._distanceEuclid(self._train_df)
            distances_test = self._distanceEuclid(testData)
        
        elif self.distanceMetric == "mahalanobis":

            distances_train = self._distanceMahal(self._train_df)
            distances_test = self._distanceMahal(testData)

        # Initializes the predicted labels list
        predicted_labels = [None] * len(testData)

        # Calculate distances between each test and training instance
        for test_index in testData.index:

            # Create a list of distances
            distances = [None] * len(self._train_df)

            # Cycle through each training instance
            for train_index in self._train_df.index:

                # Determine distance between instances
                distances[train_index] = abs(distances_test[test_index] - distances_train[train_index])

            # Use bubble sort to get the k-smallest values and indexes faster than other sorts
            kNeighbors, kLabels = self._bubbleSortK(distances, self._labels.tolist())

            # Select the most common label of the nearest neighbors
            predicted_labels[test_index] = max(set(kLabels), key=kLabels.count)

        # Return the predicted labels
        return predicted_labels

    def _mode(self, _list):
        '''
        Returns the most common label of the k closest training instances.

        Parameters
        ----------
        _list : list
            A list of items to find the mode from.

        Returns
        -------
        mode : item from list
            A list of the most common label among training neighbors for each test instance.
        '''

        # Create a blank dictionary
        items = {}

        # Cycle through each element in the list
        for item in _list:

            # Try to increment the given key
            try:
                items[item] = item[item] + 1
            # If it doesn't exist, create the key
            except:
                items[item] = 1

        # Find the most frequent label
        mode = max(items, key=item.get)

        # Return the mode
        return mode

    def _bubbleSortK(self, toSort, labels):
        '''
        Sorts a numeric list in ascending order using bubble sort, but only finds the k smallest values this way.

        Parameters
        ----------
        toSort : list
            An unsorted list of numeric values.

        labels : list
            Corresponding labels of these instances.

        Returns
        -------
        toSort : list
            A list sorted in ascending order.

        labels : list
            A list of labels for the closest instances.
        '''

        # Check to ensure all variables are of the right type
        if not isinstance(toSort, list):
            raise TypeError("'toSort' must be of type list.")
        elif not isinstance(labels, list):
            raise TypeError("'labels' must be of type list.")

        # Ensure the distances and labels lists are the same length
        if len(toSort) != len(labels):
            raise ValueError("The distance and label lists must be of the same length.")

        # Loops k times
        for i in range(self.k):

            # Loop through each element except those already sorted
            for j in range(0, len(toSort) - i - 1):
                
                # Swap neighboring elements if the left value is greater than the right
                if toSort[j] < toSort[j + 1]:
                    toSort[j], toSort[j + 1] = toSort[j + 1], toSort[j]
                    labels[j], labels[j + 1] = labels[j + 1], labels[j]

        # At this point, the k smallest numbers should be to the right
        return toSort[-self.k:], labels[-self.k:]

    def _distanceEuclid(self, df):
        '''
        Calculates the Euclidean distance for all points in the dataset.

        Parameters
        ----------
        df : Pandas DataFrame
            Set of training observations to calculate the distance for.

        Returns
        -------
        vals : list
            A list of Euclidean distances for each instance.
        '''

        # Initialize distance column as zeros
        df["dist"] = 0

        # Square every value in the dataframe and add to a new distance column
        for col in df.columns:
            df["dist"] = df["dist"] + df[col]**2
        
        # Take the square root of all values in the distance column
        df["dist"] = df["dist"]**(1/2)

        # Change to a list
        vals = df["dist"].tolist()

        # Returns the Euclidean distances
        return vals

    def _distanceMahal(self, df):
        '''
        Calculates the Mahalanobis distance for all points in the dataset.

        Parameters
        ----------
        df : Pandas DataFrame
            Set of training observations to calculate the distance for.

        Returns
        -------
        vals : list
            A list of Mahalanobis distances for each instance.
        '''

        # Calculates the mean-adjusted values for each feature
        y_mu = df - df.mean(axis = 0)

        # Calculates the inverse covariance matrix of the dataset
        cov = np.cov(df.values.T)
        inv_covmat = np.linalg.inv(cov)

        # Multiplies the mean-adjusted values by the inverted covariance matrix
        temp = np.dot(y_mu, inv_covmat)

        # Multiplies by the transpose of the mean-adjusted values
        temp = np.dot(temp, y_mu.T)

        # Extracts the distances and takes the square root
        vals = np.sqrt(temp.diagonal()).tolist()

        # Returns the Mahalanobis distances
        return vals

class decisionTreeClassifier(ClassifierAlgorithm, TreeNodeABC):
    '''
    Creates a basic decision tree classifier that will work for both qualitative and quantitative datasets.
    '''
   
    def __init__(self, infoGainThreshold = 0.1):
        '''
        Instantiates the decision tree classifier.

        Parameters
        ----------
        infoGainThreshold : float
            What threshold of information gain should be met for splitting to occur.
            Ranges from 0 to 2^c where c is the number of classes.
        '''

        # Check if the parameters are of the correct type
        if not isinstance(infoGainThreshold, float):
            raise TypeError("'infoGainThreshold' must be of type float.")

        # Initializes relevant attributes
        self.infoGainThreshold = infoGainThreshold
        self.root_node = None
        
    def train(self, trainingData, trueLabels):
        '''
        Stores the data and labels, creates a root node, and calls the recursive training function to build the tree.
        Saves the root node to the model.

        Parameters
        ----------
        trainingData : Pandas DataFrame
            A dataframe of all features used to for prediction.

        trueLabels : str
            A string defining the column storing the labels.
        '''

        # Check if the parameters are of the correct type
        if not isinstance(trainingData, pd.DataFrame):
            raise TypeError("The training data must be a Pandas DataFrame.")
        elif not isinstance(trueLabels, str):
            raise TypeError("'trueLabels' must be a string.")

        # Check to ensure the labels exist within the dataset
        if trueLabels not in trainingData.columns.tolist():
            raise IndexError("'trueLabels' must be the name of a column in the training data.")

        # Save to member attributes
        self._train_df = trainingData.drop([trueLabels], axis = 1)
        self._labels = trainingData[trueLabels].tolist()

        # Creates a blank root node
        # Sets the entropy to its maximum and the prediction to the most common class
        node = TreeNodeABC(self._train_df .index.values.tolist(), 2**len(set(self._labels)), max(set(self._labels), key=self._labels.count))

        # Finds the best split with the current data
        best_split_type, best_ig, [best_left_ent, best_right_ent], [best_left_pred, best_right_pred], best_feat, best_val = node.bestSplit(self._train_df, self._labels)

        # Creates a root node of this split's type
        if best_split_type in (np.dtype('int64'), np.dtype('float64')):
            self.root_node = QuantNode(self._train_df.index.values.tolist(), 2**len(set(self._labels)), max(set(self._labels), key=self._labels.count), infoGainThreshold = self.infoGainThreshold)
        
        # If qualitative...
        elif isinstance(best_split_type, str) or isinstance(best_split_type, object):
            self.root_node = QualNode(self._train_df.index.values.tolist(), 2**len(set(self._labels)), max(set(self._labels), key=self._labels.count), infoGainThreshold = self.infoGainThreshold)

        # Using this node, begin recursively training the decision tree
        self.__recursive_train(self.root_node)
        
    def test(self, testData):
        '''
        Returns the predicted label for each test instance.

        Parameters
        ----------
        testData : Pandas DataFrame
            Set of observations in the same format as the training data.
            Will be labeled based on nearby training instances. Must not
            include a column for labels.

        Returns
        -------
        predicted_labels : list
            A list of predicted labels for each test instance.
        '''
        
        # Check the type of the test data
        if not isinstance(testData, pd.DataFrame):
            raise TypeError("The test data must be a Pandas DataFrame.")

        # Ensure that the indices for the test data have been reset
        testData = testData.reset_index(drop = True)

        # Ensure that the test and training data have the same features
        if set(self._train_df.columns) != set(testData.columns):
            raise ValueError("The test and train data do not have the same columns")

        # Initialize the list of predicted labels
        predicted_labels = [None] * len(testData)

        # Cycle through each test instance
        for i in range(len(testData)):
            
            # Call the recursive test function to get a predicted label for each instance
            # Make sure to start with the root node here
            predicted_labels[i] = self.__recursive_test(self.root_node, testData.iloc[i])

        # Return the predicted labels
        return predicted_labels

    def __recursive_train(self, node):
        '''
        Cycles through nodes, splitting and adding children with each step until
        a information gain threshold is reached.

        Parameters
        ----------
        node : QualNode or QuantNode
            A qualitative or quantitative node to be split.
        '''

        # Check if this node has more than one instance
        if node.isPure():
            
            # If true, return nothing
            return

        # Find the best split for this node to maximize information gain
        split_type, ig, child_ent, child_prediction, col, val = node.bestSplit(self._train_df, self._labels)
        
        # Split if it would produce a high enough information gain
        if ig > node.infoGainThreshold:
            # performs the split on the optimal feature and condition
            # returns DTNode subclasses matching the typeof argument
            left, right = node.split(self._train_df, split_type, ig, child_ent, child_prediction, col, val)
            # call train_recursive on the left and right child nodes
            self.__recursive_train(left)
            self.__recursive_train(right)

    def __recursive_test(self, node, data):
        '''
        Iterates through the tree with a single row of data to find a predicted value.

        Parameters
        ----------
        node : QualNode or QuantNode
            A node of the model's tree.

        data : Pandas Series
            A single row/instance from a dataframe.

        Returns
        -------
        predicted_label : int, str, or object
            A value representing the most common label in the leaf node where the instance ended up.
        '''

        # Check that all parameters are of the correct type
        if not (isinstance(node, QualNode) or isinstance(node, QuantNode)):
            raise TypeError("'node' must be of type QualNode or QuantNode.")
        elif not isinstance(data, pd.Series):
            raise TypeError("'data' must be of type Pandas Series.")

        # If the given node is a leaf node, return its prediction
        if not node.parent():
            return node.prediction
        
        # Check whether this instance should go to the left or right of the split
        if node.goLeft(data):
            return self.__recursive_test(node.left_child, data)
        else:
            return self.__recursive_test(node.right_child, data)

    def __str__(self):
        '''
        Calls the toString function to print the tree.

        Returns
        -------
        self.__toString(self.root) : str
            A string which can be passed to http://mshang.ca/syntree/ to print the decision tree.
        '''
        return self.__toString(self.root_node)
    
    def __toString(self, node):
        '''
        Recursively creates the string which will display the tree http://mshang.ca/syntree/.

        Parameters
        ----------
        node : QualNode or QuantNode
            A node of the model's tree.

        Returns
        -------
        sss : str
            A string representation of the decision tree.
        '''

        # If this node has no children, return the prediction
        if not node.parent():
            return ' [' + str(node.prediction) + ']'

        # Return this node and all children
        else:
            sss = '[' + str(node.feature) + ':' + str(node.condition)
            sss += self.__toString(node.left_child)
            sss += self.__toString(node.right_child)
            sss += ']'
            return sss

class kdTreeKNNClassifier(simpleKNNClassifier, decisionTreeClassifier, kdTreeNodeABC):
    '''
    Creates a KNN classifier optimized for large, low-dimensionality datasets via
    the use of a KD-tree data structure.
    '''
    
    def train(self, trainingData, trueLabels):
        '''
        Stores the data and labels, creates a root node, and calls the recursive training function to build the KD tree.
        Saves the root node to the model.

        Parameters
        ----------
        trainingData : Pandas DataFrame
            A dataframe of all features used to for prediction.

        trueLabels : str
            A string defining the column storing the labels.
        '''

        # Check if the parameters are of the correct type
        if not isinstance(trainingData, pd.DataFrame):
            raise TypeError("The training data must be a Pandas DataFrame.")
        elif not isinstance(trueLabels, str):
            raise TypeError("'trueLabels' must be a string.")

        # Check to ensure the labels exist within the dataset
        if trueLabels not in trainingData.columns.tolist():
            raise IndexError("'trueLabels' must be the name of a column in the training data.")

        # Save to member attributes
        self._train_df = trainingData.drop([trueLabels], axis = 1)
        self._labels = trainingData[trueLabels].tolist()

        # Select the first dimension to split on
        dim = self._train_df.columns[1]

        # Store the first dimension's type
        split_type = self._train_df[dim].dtype

        # Find the median instance along this dimension
        median_inst = self.__get_median_index(self._train_df[dim])

        # Create the root node by type
        if split_type in (np.dtype('int64'), np.dtype('float64')):

            # Create a quantitative root node
            self.root_node = kdQuantNode(self._train_df.index.values.tolist())

        elif isinstance(split_type, str) or isinstance(split_type, object):

            # Create a qualitative root node
            self.root_node = kdQualNode(self._train_df.index.values.tolist())

        # Use the median instance to initialize the root node
        self.root_node.instance = median_inst
        self.root_node.prediction = self._labels[median_inst]
        self.root_node.feature = dim
        self.root_node.condition = self._train_df[dim].loc[median_inst]

        # Using this node, begin recursively training the kd tree
        self.__recursive_train(self.root_node)  
      
    def test(self, testData):
        '''
        Returns the predicted label for each test instance.

        Parameters
        ----------
        testData : Pandas DataFrame
            Set of observations in the same format as the training data.
            Will be labeled based on nearby training instances. Must not
            include a column for labels.

        Returns
        -------
        predicted_labels : list
            A list of predicted labels for each test instance.
        '''
        
        # Check the type of the test data
        if not isinstance(testData, pd.DataFrame):
            raise TypeError("The test data must be a Pandas DataFrame.")

        # Ensure that the indices for the test data have been reset
        testData = testData.reset_index(drop = True)

        # Ensure that the test and training data have the same features
        if set(self._train_df.columns) != set(testData.columns):
            raise ValueError("The test and train data do not have the same columns")

        # Initialize the list of predicted labels
        predicted_labels = [None] * len(testData)

        # Initialize the test and train distances
        if self.distanceMetric == "euclidean":
            distances_test = self._distanceEuclid(testData)
            distances_train = self._distanceEuclid(self._train_df)
        elif self.distanceMetric == "mahalanobis":
            distances_test = self._distanceMahal(testData)
            distances_train = self._distanceMahal(self._train_df)

        # Cycle through each test instance
        for i in range(len(testData)):
            
            # Call the recursive test function to get a list of instances along the path
            path = self.__recursive_test(self.root_node, testData.iloc[i])

            # Extract the distances and labels along the path
            distances = [None] * len(path)
            path_labels = [None] * len(path)
            for j in range(len(path)):
                distances[j] = abs(distances_test[i] - distances_train[path[j].instance])
                path_labels[j] = self._labels[path[j].instance]             

            # Find the k-closest instances
            kNeighbors, kLabels = self._bubbleSortK(distances, path_labels)

            # Select the most common label of the nearest neighbors
            predicted_labels[i] = max(set(kLabels), key=kLabels.count)

        # Return the predicted labels
        return predicted_labels  
              
    def __recursive_train(self, node, depth = 2):
        '''
        Cycles through nodes, splitting and adding children with each step until
        each node represents a single instance.

        Parameters
        ----------
        node : QualNode or QuantNode
            A qualitative or quantitative node to be split.

        depth : int
            Vertical location in the kd tree. Used to select the splitting feature.
        '''

        # Check if this node is empty
        if node == None:
            
            # If true, return nothing
            return

        # Check if this node has one instance
        elif len(node.indices) == 1:

            # Set this node's instance to itself and its label, then return nothing
            node.instance = node.indices[0]
            node.prediction = self._labels[node.instance]
            return

        # Subset the data
        df = self._train_df.loc[node.indices]

        # Select the dimension to split on
        dim = df.columns[depth % len(df.columns)]

        # Store the dimension's type
        split_type = df[dim].dtype

        # Find the median instance, then use it to set this node's 
        # # prediction, feature, and split condition
        node.instance = self.__get_median_index(df[dim])
        node.prediction = self._labels[node.instance]
        node.feature = dim
        node.condition = df[dim].loc[node.instance]

        # performs the split on the optimal feature and condition
        # returns DTNode subclasses matching the typeof argument
        left, right = node.split(df, split_type)

        # Call train_recursive on the left and right child nodes
        self.__recursive_train(left, depth+1)
        self.__recursive_train(right, depth+1)

    def __recursive_test(self, node, data, path = None):
        '''
        Iterates through the tree with a single row of data to find a predicted value.

        Parameters
        ----------
        node : QualNode or QuantNode
            A node of the model's tree.

        data : Pandas Series
            A single row/instance from a dataframe.

        path : list
            A list of training indexes for each node this test instance has traversed over.

        Returns
        -------
        path : list
            A list of each node this test instance has traversed over.
        '''

        # Check that all parameters are of the correct type
        if not (isinstance(node, kdQualNode) or isinstance(node, kdQuantNode)):
            raise TypeError("'node' must be of type QualNode or QuantNode.")
        elif not isinstance(data, pd.Series):
            raise TypeError("'data' must be of type Pandas Series.")
        elif (not isinstance(path, list)) and (not path == None):
            raise TypeError("'path' must be of type list or None.")

        # Add this node's training index to the path
        if path == None:
            path = [node]
        else:
            path.append(node)

        # If the given node is a leaf node, return the path
        if not node.parent():
            return path
        
        # Check whether this instance should go to the left or right of the split
        # Add the other child node for each split to get all nodes 1 away from the direct path
        if (node.goLeft(data)) and (node.left_child != None):

            # Add the right child node if it exists
            if node.right_child != None:
                path.append(node.right_child)

            return self.__recursive_test(node.left_child, data, path)
        elif node.right_child != None:

            # Add the left child node if it exists
            if node.left_child != None:
                path.append(node.left_child)

            return self.__recursive_test(node.right_child, data, path)

    def __get_median_index(self, df):
        '''
        Finds the index for the median of a particular column in a dataframe.

        Parameters
        ----------
        df : Pandas Series
            A column of a dataframe to find the median value on.
        '''
        
        # Ranks each row of the dataframe by its value
        ranks = df.rank(pct=True)

        # Finds the one closest to the median and returns
        close_to_median = abs(ranks - 0.5)
        return int(close_to_median.idxmin())

    def __str__(self):
        '''
        Calls the toString function to print the tree.

        Returns
        -------
        self.__toString(self.root) : str
            A string which can be passed to http://mshang.ca/syntree/ to print the decision tree.
        '''
        return self.__toString(self.root_node)

    def __toString(self, node):
        '''
        Recursively creates the string which will display the tree http://mshang.ca/syntree/.

        Parameters
        ----------
        node : QualNode or QuantNode
            A node of the model's tree.

        Returns
        -------
        sss : str
            A string representation of the decision tree.
        '''

        # If this node has no children, return only the instance
        if not node.parent():
            return ' [' + str(node.prediction) + ']'

        # Return this node and all children
        else:
            sss = '[' + str(node.prediction)
            sss += self.__toString(node.left_child)
            sss += self.__toString(node.right_child)
            sss += ']'
            return sss

# if __name__ == "__main__":

#     # Import the testing data
#     test_df = pd.read_csv("test_data/knn_test_data_2.csv")

#     # Set a random seed
#     np.random.seed(123)

#     # Prepare the training data
#     train_indexes = np.random.choice(len(test_df), replace = False, size = int(.85*len(test_df)))
#     train_data = test_df.iloc[train_indexes].reset_index(drop = True)

#     # Set up the test data and labels
#     test_indexes = test_df.index[~test_df.index.isin(train_indexes)]
#     test_data = test_df.loc[test_indexes].reset_index(drop = True)
#     test_labels = test_data["species"]
#     test_data = test_data.drop(["species"], axis = 1)

#     # Create the KD-KNN Classifier
#     knn = kdTreeKNNClassifier(k = 5, distanceMetric = "euclidean")
#     knn.train(trainingData = train_data, trueLabels = "species")
#     pred_labels = knn.test(testData = test_data)

#     # Compare to the actual labels
#     score = 0
#     for i in range(len(pred_labels)):
#         if test_labels[i] == pred_labels[i]:
#             score += 1
#     score = score/len(pred_labels)
#     score

#     # Print the tree
#     print(knn)