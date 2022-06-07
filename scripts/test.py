#* test.py
#*
#* ANLY 555 Spring 2022
#* Final Project
#*
#* Due on: 2022-02-13
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
# Testing DataSet Class 
#=====================================================================

# Imports the DataSet class and all subclasses
from DataSet import (DataSet, 
                     QuantDataSet, 
                     QualDataSet,
                     TextDataSet, 
                     TimeSeriesDataSet)

# Creates functions to test the Dataset class and all subclasses

def DataSetTests():
    print("===========================================================\n")
    print("DataSet object:\n")
    data = DataSet("")
    data.clean()
    data.explore()

def QuantDataSetTests():
    print("===========================================================\n")
    print("QuantDataSet object:\n")
    data = QuantDataSet("")
    data.clean()
    data.explore()
    
def QualDataSetTests():
    print("===========================================================\n")
    print("QualDataSet object:\n")
    data = QualDataSet("")
    data.clean()
    data.explore()
    
def TextDataSetTests():
    print("===========================================================\n")
    print("TextDataSet object:\n")
    data = TextDataSet("")
    data.clean()
    data.explore()
    
def TimeSeriesDataSetTests():
    print("===========================================================\n")
    print("TimeSeriesDataSet object:\n")
    data = TimeSeriesDataSet("")
    data.clean()
    data.explore()

#=====================================================================
# Testing Classifier Class 
#=====================================================================

# Imports the ClassifierAlgorithm class and all subclasses

from ClassifierAlgorithm import (ClassifierAlgorithm,
                                 simpleKNNClassifier,
                                 kdTreeKNNClassifier)

# Creates functions to test the ClassifierAlgorithm class and all subclasses
                                        
def ClassifierAlgorithmTests():
    print("===========================================================\n")
    print("ClassifierAlgorithm object:\n")
    classifier = ClassifierAlgorithm()
    classifier.train()
    classifier.test()

def simpleKNNClassifierTests():
    print("===========================================================\n")
    print("simpleKNNClassifier object:\n")
    classifier = simpleKNNClassifier()
    classifier.train()
    classifier.test()
    
def kdTreeKNNClassifierTests():
    print("===========================================================\n")
    print("kdTreeKNNClassifier object:\n")
    classifier = kdTreeKNNClassifier()
    classifier.train()
    classifier.test()
        
#=====================================================================
# Testing Experiment Class 
#=====================================================================

# Imports the Experiment class

from Experiment import Experiment

# Creates functions to test the Experiment class

def ExperimentTests():
    print("===========================================================\n")
    print("Experiment object:\n")
    experiment = Experiment()   
    experiment.score()
    experiment.runCrossVal()
    
# Runs all test functions
    
if __name__=="__main__":
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    kdTreeKNNClassifierTests()
    ExperimentTests()