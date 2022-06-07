#* test03.py
#*
#* ANLY 555 Spring 2022
#* Final Project: Deliverable 5
#*
#* Due on: 2022-04-25
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
# Testing the kdTree KNN Classifier
#=====================================================================

# Imports the kdTree KNN CLasifier
from scripts.ClassifierAlgorithm import *

# Creates a function to test the loading, cleaning, and exploring of the ABC Classifier class
def kdKNNTests(df, labels):

    # Prepare the training data
    train_indexes = np.random.choice(len(df), replace = False, size = int(.85*len(df)))
    train_data = df.iloc[train_indexes].reset_index(drop = True)

    # Set up the test data and labels
    test_indexes = df.index[~df.index.isin(train_indexes)]
    test_data = df.loc[test_indexes].reset_index(drop = True)
    test_labels = test_data[labels]
    test_data = test_data.drop([labels], axis = 1)

    print("===========================================================\n")
    print("Decision Tree Classifier object:\n")

    # Create the KD Tree KNN Classifier
    kdKNN = kdTreeKNNClassifier()
    kdKNN
    kdKNN.train(trainingData = train_data, trueLabels = labels)
    pred_labels = kdKNN.test(testData = test_data)
    print("First 8 actual labels:\n")
    print(test_labels[0:8].tolist(), "\n")
    print("First 8 predicted labels:\n")
    print(pred_labels[0:8], "\n")

    # Compare to the actual labels
    score = 0
    for i in range(len(pred_labels)):
        if test_labels[i] == pred_labels[i]:
            score += 1
    score = score/len(pred_labels)
    print("Accuracy of the KD Tree KNN classifier:\n", score)

    return 
    
# Runs all of the testing functions
if __name__ == "__main__":

    # Import the testing data
    test_df = pd.read_csv("test_data/knn_test_data_2.csv")

    # Set a random seed
    np.random.seed(123)

    # Run tests on the decision tree classifier
    kdKNNTests(test_df, "species")