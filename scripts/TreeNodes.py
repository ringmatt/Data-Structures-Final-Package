#* TreeNodes.py
#*
#* ANLY 555 Spring 2022
#* Final Project: Deliverable 4
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

#=====================================================================
# Creates the basic decision tree node classes
#=====================================================================

# Import necessary packages
import numpy as np
import pandas as pd

# Creates the ABC tree node base class

class TreeNodeABC():
    '''
    The base class for decision tree nodes.
    '''

    def __init__(self, indices, entropy, prediction, infoGainThreshold = 0.1):
        '''
        Initializes a tree node.

        Parameters
        -------
        indices : list
            Integer numbers corresponding the each instance's index in the data.

        entropy : float
            The entropy of this node. A measure of node purity.

        prediction : int, str, or object
            The most common/probable class in this node.

        infoGainThreshold : float
            What threshold of information gain should be met for splitting to occur.
            Ranges from 0 to 2^c where c is the number of classes.
        '''

        # Check the types of all parameters
        if not isinstance(indices, list):
            raise TypeError("'indices' must be of type int.")
        elif not (isinstance(entropy, float) or isinstance(entropy, int)):
            raise TypeError("'entropy' must be of type float or int.")
        elif not (isinstance(prediction, int) or isinstance(prediction, str) or isinstance(prediction, object)):
            raise TypeError("'prediction' must be of type int, str, or object.")
        elif not isinstance(infoGainThreshold, float):
            raise TypeError("'infoGainThreshold' must be of type float.")

        # Check input value types
        self.prediction = prediction
        self.indices = indices
        self.entropy = entropy
        self.infoGainThreshold = infoGainThreshold

        # These will be filled out if child nodes are created
        self.condition = None
        self.feature = None
        self.left_child = None
        self.right_child = None

    def parent(self):
        '''
        Determines whether left or right children nodes exist.

        Returns
        -------
        children : bool
            Whether the node is a parent node/has child nodes.
        '''

        # Check if the left and right nodes exist
        if self.left_child != None and self.right_child != None:

            # Return that the nodes exist
            return True

        # If one child does not exist, return false
        else:
            return False

    def isPure(self):
        '''
        Checks if the current entropy is zero.

        Returns
        -------
        purity : bool
            Whether the node contains only one type of label.
        '''

        # If entropy is zero, there is only one class so return true
        if self.entropy == 0:
            return True
        else: 
            return False 

    def bestSplit(self, data, labels):
        '''
        Finds and returns the best feature to split the data on.

        Parameters
        -------
        data : Pandas DataFrame
            A dataframe of instances to split on. Should NOT include the outcome feature/labels.

        labels : list
            A list of the class labels for these instances.

        Returns
        -------
        best_split_type : data type
            Represents either quantitative (int or float) or qualitative (str or object) data types

        best_ig : float
            The information gain of the returned split.
        
        child_entropy : list
            The entropy for each child node as floats.

        child_predictions : list
            The class prediction for each child node.

        best_feat : str
            Name of the feature which produces the best split.

        best_val : str or int
            Condition upon which to split the data. Type depends on the type of split being performed.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be of type Pandas DataFrame.")
        elif not isinstance(labels, list):
            raise TypeError("'labels' must be of type list.")

        # Subset the data to the correct indices
        df = data.iloc[self.indices]

        # Initalize the best entropy as negative infinity
        best_ent = float('inf')

        # Initalize the best feature, value to split on, info gain, and split type as None
        best_feat, best_val, best_ig, best_split_type = [None]*4

        # Initalize the entropies and predictions of the best child nodes as None
        best_left_ent, best_right_ent, best_left_pred, best_right_pred = [None]*4

        # Cycle through the features
        for col in df.columns:

            # Initialize the best value to split on and info gain for the current column as None
            best_val_temp, best_ig_temp = [None]*2

            # Initialize the current best entropy for this feature and child nodes as infinity
            best_ent_temp, best_left_ent_temp, best_right_ent_temp = [float('inf')]*3

            # Initialize the current best predictions for the left and right child nodes as None
            best_left_pred_temp, best_right_pred_temp = [None]*2         

            # Save the feature's type
            split_type_temp = df[col].dtype

            # Check if the feature is quantitative
            if split_type_temp in (np.dtype('int64'), np.dtype('float64')):

                # If quantitative, cycle through 10 percentiles of split points
                for i in df[col].quantile(np.arange(0.1,1,0.1)).values:

                    # Split the data and extract labels
                    left_node_labels = [labels[j] for j in df.loc[df[col] <= i].index.values.tolist()]
                    right_node_labels = [labels[j] for j in df.loc[df[col] > i].index.values.tolist()]

                    # Calculate the entropy for each child node
                    left_ent_temp, left_pred_temp = self.__entropy(left_node_labels)
                    right_ent_temp, right_pred_temp = self.__entropy(right_node_labels)

                    # Get a weighted average of the entropy
                    curr_ent = len(left_node_labels)/len(df)*left_ent_temp + len(right_node_labels)/len(df)*right_ent_temp

                    # If lower than the current best entropy, set this to the best entropy, split, child nodes' entropy, and info gain
                    if curr_ent < best_ent_temp:
                        best_val_temp = i
                        best_ent_temp = curr_ent
                        best_left_ent_temp = left_ent_temp
                        best_right_ent_temp = right_ent_temp
                        best_left_pred_temp = left_pred_temp
                        best_right_pred_temp = right_pred_temp
                        best_ig_temp = self.entropy - curr_ent

            elif isinstance(split_type_temp, str) or isinstance(split_type_temp, object):

                # If qualitative, cycle through all possible split values
                for i in df[col].unique():

                    # Split the data and extract labels
                    left_node_labels = [labels[j] for j in df.loc[df[col] != i].index.values.tolist()]
                    right_node_labels = [labels[j] for j in df.loc[df[col] == i].index.values.tolist()]

                    # Calculate the entropy for each child node
                    left_ent_temp, left_pred_temp = self.__entropy(left_node_labels)
                    right_ent_temp, right_pred_temp = self.__entropy(right_node_labels)

                    # Get a weighted average of the entropy
                    curr_ent = len(left_node_labels)/len(df)*left_ent_temp + len(right_node_labels)/len(df)*right_ent_temp

                    # If lower than the current best entropy, set this to the best entropy, split and child nodes
                    if curr_ent < best_ent_temp:
                        best_val_temp = i
                        best_ent_temp = curr_ent
                        best_left_ent_temp = left_ent_temp
                        best_right_ent_temp = right_ent_temp
                        best_left_pred_temp = left_pred_temp
                        best_right_pred_temp = right_pred_temp
                        best_ig_temp = self.entropy - curr_ent

            # If this feature's entropy is lower than the prior features, replace the best feature, entropy, split, split type, and info gain
            if best_ent_temp < best_ent:
                best_feat = col
                best_ent = best_ent_temp
                best_val = best_val_temp
                best_split_type = split_type_temp
                best_ig = best_ig_temp

                # Save each child node's entropy and predictions
                best_left_ent = best_left_ent_temp
                best_right_ent = best_right_ent_temp
                best_left_pred = best_left_pred_temp
                best_right_pred = best_right_pred_temp

        # Return the best split
        return best_split_type, best_ig, [best_left_ent, best_right_ent], [best_left_pred, best_right_pred], best_feat, best_val

    def split(self, data, split_type, best_ig, child_ent, child_prediction, col, val):
        '''
        Splits the data and returns two new nodes based on the type of the split.

        Parameters
        ----------
        data : Pandas DataFrame
            A dataframe of instances to split on. Should NOT include the outcome feature/labels.

        split_type : int, float, str, or object
            Represents either quantitative (int or float) or qualitative (str or object) data types.

        best_ig : float
            The information gain for this particular split.
        
        child_ent : list
            A list of left and right child node entropy values.

        child_prediction : list
            A list of left and right child node predicted classes.

        col : str
            Name of the feature which produces the best split.

        val : int, float, str, or object
            Condition upon which to split the data. Type depends on the type of split being performed.

        Returns
        -------
        left_node & right_node: QualNode or QuantNode
            Two new nodes based on the split criteria.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be of type Pandas Dataframe.")
        elif not (isinstance(split_type, int) or isinstance(split_type, float) or isinstance(split_type, str) or isinstance(split_type, object)):
            raise TypeError("'split_type' must be of type int, float, str, or object.")
        elif not isinstance(best_ig, float):
            raise TypeError("'best_ig' must be of type float.")
        elif not isinstance(child_ent, list):
            raise TypeError("'child_ent' must be of type list.")
        elif not isinstance(child_prediction, list):
            raise TypeError("'child_prediction' must be of type list.")
        elif not isinstance(col, str):
            raise TypeError("'col' must be of type str.")
        elif not (isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, object)):
            raise TypeError("'val' must be of type int, float, str, or object.")

        # Split only if the information gain is large enough (entropy decreased sufficiently)
        # Otherwise return empty child nodes
        if best_ig < self.infoGainThreshold:

            # Set the child nodes to None
            self.left_child = None
            self.right_child = None

            # Return each node as None
            return None, None

        # Subset the data to the correct parent node indices
        df = data.iloc[self.indices]

        # Set the condition and feature attributes for this node
        self.condition = val
        self.feature = col

        # Check if the split is quantitative
        if split_type in (np.dtype('int64'), np.dtype('float64')):

            # Split the data
            left_data = df.loc[df[col] <= val]
            right_data = df.loc[df[col] > val]

            # Create the left and right nodes
            left_node = QuantNode(left_data.index.values.tolist(), child_ent[0], child_prediction[0], infoGainThreshold = self.infoGainThreshold)
            right_node = QuantNode(right_data.index.values.tolist(), child_ent[1], child_prediction[1], infoGainThreshold = self.infoGainThreshold)

        # If qualitative...
        elif isinstance(split_type, str) or isinstance(split_type, object):

            # Split the data
            left_data = df.loc[df[col] != val]
            right_data = df.loc[df[col] == val]

            # Create the left and right nodes
            left_node = QualNode(left_data.index.values.tolist(), child_ent[0], child_prediction[0], infoGainThreshold = self.infoGainThreshold)
            right_node = QualNode(right_data.index.values.tolist(), child_ent[1], child_prediction[1], infoGainThreshold = self.infoGainThreshold)

        # Save as child nodes
        self.left_child = left_node
        self.right_child = right_node

        # Return each node
        return left_node, right_node

    def __entropy(self, labels):
        '''
        Calculates the entropy for a given node.

        Parameters
        ----------
        labels : list
            A list of the class values per instance.

        Returns
        -------
        entropy: float
            The entropy of the set of instances.

        best_pred: int, str, or object
            The most common label for this split.
        '''

        # Check the types of all parameters
        if not isinstance(labels, list):
            raise TypeError("'labels' must be of type list.")

        # Determine the number of instances  
        n_labels = len(labels)

        # If empty, return an entropy of zero and predicted value of None
        if n_labels == 0:
            return 0, None

        # Calculate the probability of a given instance being each label
        value, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels

        # Extract the most likely class
        best_pred = None
        best_pred_prob = 0
        for i in range(0,len(probs)):
            if probs[i] > best_pred_prob:
                best_pred_prob = probs[i]
                best_pred = value[i]

        # If there is only one class, set entropy to zero and return that value
        if len(value) == 1:
            return 0, labels[0]

        # Initalize entropy to zero
        entropy = 0

        # Compute entropy as the negative sum of probabilities for each class
        # times the log of that class's probability
        for i in probs:
            entropy -= i * np.log(i)

        # Return the entropy
        return entropy, best_pred

# Creates the qualitative tree node class

class QualNode(TreeNodeABC):
    '''
    A decision tree node implemented for splits on qualitative features.
    '''

    def goLeft(self, data):
        '''
        Determines whether a row/instance from a dataframe has a feature value not equal to this node's condition.

        Parameters
        -------
        data : Pandas Series
            A single row/instance from a dataframe.

        Returns
        -------
        left : bool
            A boolean which is true when one should traverse to the left node of the tree.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.Series):
            raise TypeError("'data' must be of type Pandas Series.")

        # Compare this instance's value to this node's condition
        if data[self.feature] != self.condition:

            # Move to left node
            return True
        else:

            # Move to right node
            return False

# Creates the quantitative tree node class

class QuantNode(TreeNodeABC):
    '''
    A decision tree node implemented for splits on quantitative features.
    '''

    def goLeft(self, data):
        '''
        Determines whether a row/instance from a dataframe has a feature value less than or equal to this node's condition.

        Parameters
        -------
        data : Pandas Series
            A single row/instance from a dataframe.

        Returns
        -------
        left : bool
            A boolean which is true when one should traverse to the left node of the tree.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.Series):
            raise TypeError("'data' must be of type Pandas Series.")

        # Compare this instance's value to this node's condition
        if data[self.feature] <= self.condition:

            # Move to left node
            return True
        else:

            # Move to right node
            return False

# Creates a KD Tree ABC Node class

class kdTreeNodeABC(TreeNodeABC):
    '''
    The base class for kd tree knn classifier nodes.
    '''

    def __init__(self, indices):
        '''
        Initializes a KD tree node.

        Parameters
        -------
        indices : list
            Integer numbers corresponding the each instance's index in the data.
        '''

        # Check the types of all parameters
        if not isinstance(indices, list):
            raise TypeError("'indices' must be of type list.")

        # Check input value types
        self.indices = indices

        # These will be filled out during splitting
        self.instance = None
        self.prediction = None
        self.condition = None
        self.feature = None
        self.left_child = None
        self.right_child = None

    def split(self, data, split_type):
        '''
        Splits the data, saves the median instance as this node's value,
        and returns two new nodes based on the type of the split.

        Parameters
        ----------
        data : Pandas DataFrame
            A dataframe of instances to split on. Should NOT include the outcome feature/labels.

        split_type : int, float, str, or object
            Represents either quantitative (int or float) or qualitative (str or object) data types.

        Returns
        -------
        left_node & right_node: QualNode or QuantNode
            Two new nodes based on the split criteria.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be of type Pandas Dataframe.")
        elif not (isinstance(split_type, int) or isinstance(split_type, float) or isinstance(split_type, str) or isinstance(split_type, object)):
            raise TypeError("'split_type' must be of type int, float, str, or object.")

        # Remove the median value and save as this node's prediction
        df = data.drop(index = self.instance)

        # Check if the split is quantitative
        if split_type in (np.dtype('int64'), np.dtype('float64')):

            # Split the data
            left_data = df.loc[df[self.feature] <= self.condition]
            right_data = df.loc[df[self.feature] > self.condition]

            # Create the left and right nodes IF indices exist for that node
            if not left_data.empty:
                left_node = kdQuantNode(left_data.index.values.tolist())
            else: 
                left_node = None
            
            if not right_data.empty:
                right_node = kdQuantNode(right_data.index.values.tolist())
            else: 
                right_node = None

        # If qualitative...
        elif isinstance(split_type, str) or isinstance(split_type, object):

            # Split the data
            left_data = df.loc[df[self.feature] != self.condition]
            right_data = df.loc[df[self.feature] == self.condition]

            # Create the left and right nodes IF indices exist for that node
            if not left_data.empty:
                left_node = kdQualNode(left_data.index.values.tolist())
            else: 
                left_node = None

            if not right_data.empty:
                right_node = kdQualNode(right_data.index.values.tolist())
            else: 
                right_node = None

        # Save as child nodes
        self.left_child = left_node
        self.right_child = right_node

        # Return each node
        return left_node, right_node

# Creates kd tree versions of the qualitative tree node class

class kdQualNode(kdTreeNodeABC):
    '''
    A decision tree node implemented for splits on qualitative features.
    '''

    def goLeft(self, data):
        '''
        Determines whether a row/instance from a dataframe has a feature value not equal to this node's condition.

        Parameters
        -------
        data : Pandas Series
            A single row/instance from a dataframe.

        Returns
        -------
        left : bool
            A boolean which is true when one should traverse to the left node of the tree.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.Series):
            raise TypeError("'data' must be of type Pandas Series.")

        # Compare this instance's value to this node's condition
        if data[self.feature] != self.condition:

            # Move to left node
            return True
        else:

            # Move to right node
            return False

# Creates kd tree versions of the quantitative tree node class

class kdQuantNode(kdTreeNodeABC):
    '''
    A decision tree node implemented for splits on quantitative features.
    '''

    def goLeft(self, data):
        '''
        Determines whether a row/instance from a dataframe has a feature value less than or equal to this node's condition.

        Parameters
        -------
        data : Pandas Series
            A single row/instance from a dataframe.

        Returns
        -------
        left : bool
            A boolean which is true when one should traverse to the left node of the tree.
        '''

        # Check the types of all parameters
        if not isinstance(data, pd.Series):
            raise TypeError("'data' must be of type Pandas Series.")

        # Compare this instance's value to this node's condition
        if data[self.feature] <= self.condition:

            # Move to left node
            return True
        else:

            # Move to right node
            return False

# if __name__ == "__main__":

#     # Import the testing data
#     test_df = pd.read_csv("test_data/knn_test_data_2.csv")

#     # Set a random seed
#     np.random.seed(123)

#     # Prepare the data
#     test_data = test_df.drop(["species"], axis = 1)
#     test_labels = test_df["species"].tolist()

#     # Setup the indices for these nodes
#     indices = test_data.index.values.tolist()

#     # Max entropy is 2 to the power of the number of classes, use this to start
#     entropy = 2**len(set(test_labels))

#     # Initial prediction will be the most common class
#     prediction = max(set(test_labels), key=test_labels.count)

#     # Create a qualitative and quantitative node
#     qual_node = QualNode(indices, entropy, prediction)
#     quant_node = QuantNode(indices, entropy, prediction)

#     # Check that all attributes are present (i.e. That initialization worked)
#     print("Init test: ", qual_node.prediction, ", ", qual_node.indices[:10], ", ", 
#                             qual_node.entropy, ", ", qual_node.infoGainThreshold, ", ",
#                             qual_node.condition, ", ", qual_node.feature, ", ",
#                             qual_node.left_child, ", ", qual_node.right_child)

#     # Check the most basic functions: those that return true or false
#     print("Parents? Qual: ", qual_node.parent(), ", Quant:", quant_node.parent())
#     print("Pure? Qual: ", qual_node.isPure(), ", Quant:", quant_node.isPure())

#     # Check the advanced functions: those that split the data
#     split_type, best_ig, child_ent, child_prediction, col, val = quant_node.bestSplit(test_data, test_labels)
#     print(split_type, best_ig, child_ent, child_prediction, col, val, sep = ", ")

#     # Here's a good gut check of the prior results
#     print("\nThis split makes sense based on the following:")
#     test_df.groupby("species").mean()

#     # Now let's make this split
#     left_node, right_node = quant_node.split(test_data, split_type, best_ig, child_ent, child_prediction, col, val)

#     # Finally, let's check that these nodes are functioning properly
#     left_node.prediction