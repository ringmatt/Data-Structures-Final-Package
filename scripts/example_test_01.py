#=====================================================================
# Testing script for Deliverable 1: Source Code Framework
#=====================================================================

#=====================================================================
# Testing DataSet Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from Del_1_SourceCode_Framework import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

def DataSetTests():
    print("DataSet Instantiation invokes both the __load() and the\
__readFromCSV() methods....")
    data = DataSet(" ")
    print("==============================================================")
    print("Check ABC member attributes...")
    print("DataSet._delim:", data._delim)
    print("DataSet._newLine:", data._newLine, "This should be on a newLine.\n")
    print("QuantDataSet.naChar:", data.naChar)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Instantiating the DataSet class again both the load()\
and the readFromCSV() methods run.")
    data = DataSet(" ")
    print("Now call DataSet.clean()...")
    data.clean()
    print("===========================================================")
    print("Now call DataSet.explore()...")
    data.explore()
    print("\n\n")

def QuantDataSetTests():
    data = QuantDataSet(" ")
    print("Check inheritence ...")
    print("QuantDataSet._newLine:",data._newLine,"This should be on a new line.")
    print("===========================================================")
    print("Check member methods...\n")
    print("Check that clean and explore methods have been overriden...\n")
    print("QuantDataSet.clean():")
    data.clean()
    print("QuantDataSet.explore():")
    data.explore()
    print("\n\n")
    
def QualDataSetTests():
    data = QualDataSet(" ")
    print("Check inheritence ...")
    print("QualDataSet._newLine:",data._newLine,"This should be on a new line.")
    print("===========================================================")
    print("Check QualDataSet member attributes...")
    print("QualDataSet.ordered(): (Should be None by default)")
    print(data.ordered)
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("QuanlDataSet.clean():")
    data.clean()
    print("QuanlDataSet.explore():")
    data.explore()
    print("\n\n")
    
def TextDataSetTests():
    data = TextDataSet(" ")
    print("Check inheritence ...")
    print("TextDataSet._newLine:",data._newLine,"This should be on a new line.")
    print("===========================================================")
    print("Check TextDataSet member attributes...")
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TextDataSet.clean():")
    data.clean()
    print("TextDataSet.explore():")
    data.explore()
    print("\n\n")
    
def TimeSeriesDataSetTests():
    data = TimeSeriesDataSet("filename","weekly")
    print("Check inheritence ...")
    print("TimeSeriesDataSet._newLine:",data._newLine,"This should be on a new line.")
    print("===========================================================")
    print("Check TimeSeriesDataSet member attributes...")
    print("TimeSeriesDataSet.timeDelta (Should be weekly):",data.timeDelta)
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TimeSeriesDataSet.clean():")
    data.clean()
    print("TimeSeriesDataSet.explore():")
    data.explore()
    print("\n\n")

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from Del_1_SourceCode_Framework import (ClassifierAlgorithm,
                                simpleKNNClassifier,kdTreeKNNClassifier,
                                hmmClassifier,graphKNNClassifier)
                                        
def ClassifierAlgorithmTests():
    print("ClassifierAlgorithm Instantiation....")
    classifier = ClassifierAlgorithm("response")
    print("==============================================================")
    print("Check ABC member attributes...")
    print("ClassifierAlgorithm._response:", classifier._response)
    print("ClassifierAlgorithm._predictors (default should be none):")
    print(classifier._predictors)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("ClassifierAlgorithm.train(data):")
    print(classifier.train(x))
    print("ClassifierAlgorithm.test(data):")
    print(classifier.test(x))
    print("ClassifierAlgorithm.transform(data):")
    print(classifier.transform(x))
    print("===========================================================\n\n")

def simpleKNNClassifierTests():
    print("simpleKNNClassifier Instantiation....")
    classifier = simpleKNNClassifier("response","k")
    print("==============================================================")
    print("Check member attributes...")
    print("simpleKNNClassifierAlgorithm._response:", classifier._response)
    print("simpleKNNClassifier._predictors (default should be none):")
    print(classifier._predictors)
    print("simpleKNNClassifier._k:",classifier._k)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("simpleKNNClassifier.train(data):")
    print(classifier.train(x))
    print("simpleKNNClassifier.test(data):")
    print(classifier.test(x))
    print("simpleKNNClassifier.transform(data):")
    print(classifier.transform(x))
    print("===========================================================\n\n")

def kdTreeKNNClassifierTests():
    print("kdTreeKNNClassifier Instantiation....")
    classifier = kdTreeKNNClassifier("response")
    print("==============================================================")
    print("Check member attributes...")
    print("kdTreeKNNClassifier._response:", classifier._response)
    print("kdTreeKNNClassifier._predictors (default should be none):")
    print(classifier._predictors)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("kdTreeKNNClassifier.train(data):")
    print(classifier.train(x))
    print("kdTreeKNNClassifier.test(data):")
    print(classifier.test(x))
    print("kdTreeKNNClassifier.transform(data):")
    print(classifier.transform(x))
    print("===========================================================\n\n")
    
def hmmClassifierTests():
    print("hmmClassifier Instantiation....")
    classifier = hmmClassifier("response")
    print("==============================================================")
    print("Check member attributes...")
    print("hmmClassifier._response:", classifier._response)
    print("hmmClassifier._predictors (default should be none):")
    print(classifier._predictors)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("hmmClassifier.train(data):")
    print(classifier.train(x))
    print("hmmClassifier.test(data):")
    print(classifier.test(x))
    print("hmmClassifier.transform(data):")
    print(classifier.transform(x))
    print("===========================================================\n\n")
    
def graphKNNClassifierTests():
    print("graphKNNClassifier Instantiation....")
    classifier = graphKNNClassifier("response")
    print("==============================================================")
    print("Check member attributes...")
    print("graphKNNClassifier._response:", classifier._response)
    print("graphKNNClassifier._predictors (default should be none):")
    print(classifier._predictors)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("graphKNNClassifier.train(data):")
    print(classifier.train(x))
    print("graphKNNClassifier.test(data):")
    print(classifier.test(x))
    print("graphKNNClassifie.transform(data):")
    print(classifier.transform(x))
    print("===========================================================\n\n")

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from Del_1_SourceCode_Framework import Experiment

def ExperimentTests():
    print("Experiment class instantiation (Experiment(classifier,data))...")
    experiment = Experiment("classifier","data")
    print("==============================================================")
    print("Check member attributes...")
    print("Experiment._classifier:",experiment._classifier)
    print("Experiment._data:",experiment._data)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.score():")
    experiment.score()
    print("Experiment.runCrossVal(numFolds,statistic):")
    experiment.runCrossVal("numFolds", "statistic")
    print("==============================================================")
    print("Experiment.ROC_curve(): (This also calls the private method\
Experiment.__confusionMatrix())")
    experiment.ROC_curve()
    
    
def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    kdTreeKNNClassifierTests()
    hmmClassifierTests()
    graphKNNClassifierTests()
    ExperimentTests()
    
if __name__=="__main__":
    main()
