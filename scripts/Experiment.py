#* Experiment.py
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

import matplotlib.pyplot as plt
import numpy as np
from scripts.ClassifierAlgorithm import *
import pandas as pd
import seaborn as sns

class Experiment:
    '''
    A class which runs cross validation and creates confusion matrices for 
    classifaction algorithms.
    '''

    def __init__(self, df, labels, classifiers):
        '''
        Creates the Experiment class instance.

        Parameters
        ----------
        df : Pandas DataFrame
            The dataset which will be experimented on.
        
        labels : str
            A string representing the labels column in the dataset

        classifiers : list
            A list of classifier objects.
        '''

        # Check to ensure the parameters are of the correct type
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'df' must be of type Pandas DataFrame.")
        elif not isinstance(labels, str):
            raise TypeError("'labels' must be of type str.")
        elif not isinstance(classifiers, list):
            raise TypeError("'classifiers' must be of type list.")

        # Save each parameter as a member attribute
        self._df = df
        self._labels = labels
        self._classifiers = classifiers
    
    def crossValidation(self, kFolds = 5, seed = 123):
        '''
        Runs cross validation tests using a specified number of folds.

        Parameters
        ----------
        kFolds : int
            The number of folds in the training data used for cross-validation.
        
        seed : int
            An integer that ensures each run returns the same results despite randomness.

        Returns
        -------
        pred_labels : dictionary
            Returns a dictionary where the keys are the classifier
            and the values are lists of predicted labels.

        true_labels : list
            Returns a list of the true labels.
        '''
        
        # Check to ensure the parameters are of the correct type
        if not isinstance(kFolds, int):
            raise TypeError("'kFolds' must be of type int.")
        elif not isinstance(seed, int):
            raise TypeError("'seed' must be of type int.")

        # Shuffle the dataset
        df = self._df.sample(frac = 1, random_state = seed)

        # Prepare a dictionary to store the results for each classifier
        pred_labels = {}

        # Save the test labels
        true_labels = df[self._labels]

        # Cycle through each classifier
        for c in self._classifiers:

            # Initialize a predicted labels list
            temp_labels = [None] * len(true_labels)

            # Cycle through the number of k-folds
            for k in range(kFolds):

                # Subset the data to the ith subset
                test_data = df.iloc[
                    int(k*len(df)/kFolds):int((k+1)*len(df)/kFolds)]
                
                # Extract non-test data to the training set
                train_data = df.iloc[~test_data.index]

                # Train the classifier
                c.train(train_data, self._labels)
                
                # Save to this fold's results
                temp_labels[
                    int(k*len(df)/kFolds):int((k+1)*len(df)/kFolds)] = c.test(test_data.drop([self._labels], axis = 1))
            
            # Save the the dictionary of classifier results
            pred_labels[c] = temp_labels

        # Returns the dictionary of predicted labels
        return pred_labels, true_labels.tolist()
        
    def score(self, pred_labels, true_labels):
        '''
        Calculates the score for the each classifier.

        Parameters
        ----------
        pred_labels : dictionary
            A dictionary where the keys are the classifier
            and the values are lists of predicted labels.

        true_labels : list
            A list of the true labels.

        Returns
        -------
        scores : dictionary
            Returns a dictionary where the keys are the classifiers
            and the values are accuracy scores.
        '''
        
        # Check to ensure the parameters are of the correct type
        if not isinstance(pred_labels, dict):
            raise TypeError("'pred_labels' must be of type dict.")
        elif not isinstance(true_labels, list):
            raise TypeError("'true_labels' must be of type list.")

        # Initialize the scores dictionary
        scores = {}

        # Cycle through the classifiers
        for classifier in self._classifiers:

            # Initialize the score
            score = 0
            
            # Cycle through the true labels
            for i in range(len(true_labels)):

                # If the label is correct, add to the score
                if true_labels[i] == pred_labels[classifier][i]:
                    score += 1
            
            # Divide by the total number of true labels
            scores[classifier] = score/len(true_labels)

        # Return the scores
        return scores

    def confusionMatrix(self, pred_labels, true_labels, titles):
        '''
        Determines the confusion matrix for each classifier.

        Parameters
        ----------
        pred_labels : dictionary
            A dictionary where the keys are the classifier
            and the values are lists of predicted labels.

        true_labels : list
            A list of the true labels.

        titles : dictionary
            A dictionary of titles for each classifier.
        '''
        
        # Ensure proper parameter types
        if not isinstance(pred_labels, dict):
            raise TypeError("'pred_labels' must be of type dict.")
        elif not isinstance(true_labels, list):
            raise TypeError("'true_labels' must be of type list.")

        # Determine unique labels
        labels = self._df[self._labels].unique().tolist()

        # Create a crosswalk dictionary relating labels to indicies
        cw = {labels[i]: i for i in range(len(labels))}

        # Initialize a dictionary of to store each classifier's matrix
        cm = {}

        # Calculate confusion matrix for each classifier

        # Cycle through the classifiers
        for c in self._classifiers:

            # Initialize a blank matrix of the correct size
            cm[c] = [[0 for i in range(len(labels))] for i in range(len(labels))]

            # Cycle through true and predicted labels
            for i in range(len(true_labels)):

                # Extract true and predicted labels for this instance
                t = true_labels[i]
                p = pred_labels[c][i]

                # Add result to appropriate key in the dictionary
                # Extracts classifier's matrix, then finds corresponding
                # indicies for given true and predicted labels
                cm[c][cw[t]][cw[p]] = cm[c][cw[t]][cw[p]] + 1

            # Convert the matrix to a dataframe and plot using seaborn
            sns.heatmap(
                pd.DataFrame(cm[c], columns = labels, index = labels), 
                annot=True)
            plt.title(titles[c])
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.show()

# if __name__ == "__main__":

#     # Import the testing data
#     test_df = pd.read_csv("test_data/knn_test_data_2.csv")

#     # Initialize the experiment
#     knn_e = simpleKNNClassifier(k = 5, distanceMetric = "euclidean")
#     knn_m = simpleKNNClassifier(k = 5, distanceMetric = "mahalanobis")
#     exp = Experiment(test_df, "species", [knn_e, knn_m])

#     # Run cross validation
#     pred_labels_test, true_labels_test = exp.crossValidation()

#     # Return scores
#     exp.score(pred_labels_test, true_labels_test)

#     # Plot confusion matrices
#     exp.confusionMatrix(pred_labels_test, true_labels_test, {knn_e: "Euclidean", knn_m: "Mahalanbois"})