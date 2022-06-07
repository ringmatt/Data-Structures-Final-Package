#* test03.py
#*
#* ANLY 555 Spring 2022
#* Final Project: Deliverable 3
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
# Testing the DataSet Class & Subclasses
#=====================================================================

# Imports the DataSet class and all subclasses
from scripts.ClassifierAlgorithm import *
from scripts.Experiment import *
import numpy as np

# Creates a function to test the loading, cleaning, and exploring of the ABC Classifier class
def ClassifierAlgorithmTests():
    print("===========================================================\n")
    print("Classifier object:\n")
    c_alg = ClassifierAlgorithm()
    c_alg.train()
    c_alg.test()

# Creates a function to test the cleaning, and exploring of the simple KNN classifier subclass
def knnClassifierTests(test_df, labels, trainPercent = 0.85):
    # Set up the training data
    train_indexes = np.random.choice(len(test_df), replace = False, size = int(trainPercent*len(test_df)))
    train_data = test_df.iloc[train_indexes].reset_index(drop = True)

    # Set up the test data
    test_indexes = test_df.index[~test_df.index.isin(train_indexes)]
    test_data = test_df.loc[test_indexes].reset_index(drop = True)
    test_labels = test_data[labels]
    test_data = test_data.drop([labels], axis = 1)
    
    print("===========================================================\n")
    print("Simple KNN Classifier object:\n")
    
    # Create the KNN Classifier
    knn = simpleKNNClassifier(k = 5, distanceMetric = "euclidean")
    knn.train(trainingData = train_data, trueLabels = labels)
    pred_labels = knn.test(testData = test_data)
    print("First 5 predicted labels:\n")
    print(pred_labels[0:5], "\n")

    # Compare to the actual labels
    score = 0
    for i in range(len(pred_labels)):
        if test_labels[i] == pred_labels[i]:
            score += 1
    score = score/len(pred_labels)
    print("Accuracy of the KNN classifier:\n", score)
    
# Creates a function to test the cleaning, and exploring of the Experiment class
def ExperimentTests(test_df, labels, classifiers, titles):
    print("===========================================================\n")
    print("Experiment object:\n")
    exp = Experiment(test_df, labels, classifiers)

    # Run cross validation
    pred_labels_test, true_labels_test = exp.crossValidation()

    # Return scores
    scores = exp.score(pred_labels_test, true_labels_test)

    print("Scores per classifier:\n", scores)

    # Plot confusion matrices
    exp.confusionMatrix(pred_labels_test, true_labels_test, {classifiers[i]: titles[i] for i in range(len(classifiers))})
    
# Runs all of the testing functions
if __name__=="__main__":

    # Import the testing data
    df = pd.read_csv("test_data/knn_test_data_2.csv")

    # Set a random seed
    np.random.seed(123)

    # Run classifier class and knn subclass tests
    ClassifierAlgorithmTests()
    knnClassifierTests(df, "species")

    # Initialize classifiers for the experiment
    knn_e = simpleKNNClassifier(k = 5, distanceMetric = "euclidean")
    knn_m = simpleKNNClassifier(k = 5, distanceMetric = "mahalanobis")

    # Run experiment class tests
    ExperimentTests(df, "species", [knn_e, knn_m], ["Euclidean", "Mahalanobis"])
    