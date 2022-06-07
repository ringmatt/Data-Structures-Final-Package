#* test02.py
#*
#* ANLY 555 Spring 2022
#* Final Project: Deliverable 2
#*
#* Due on: 2022-03-01
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
from DataSet import (DataSet, 
                     QuantDataSet, 
                     QualDataSet,
                     TextDataSet, 
                     TimeSeriesDataSet)

# Creates a function to test the loading, cleaning, and exploring of the ABC DataSet class
def DataSetTests():
    print("===========================================================\n")
    print("DataSet object:\n")
    data = DataSet()

# Creates a function to test the cleaning, and exploring of the quantitative DataSet subclass
def QuantDataSetTests(filename, columns):
    print("===========================================================\n")
    print("QuantDataSet object:\n")
    data = QuantDataSet(filename)
    data.clean()
    print(data.summary())
    data.explore(columns)
    
# Creates a function to test the cleaning, and exploring of the qualitative DataSet subclass
def QualDataSetTests(filename, columns):
    print("===========================================================\n")
    print("QualDataSet object:\n")
    data = QualDataSet(filename)
    data.clean()
    print(data.summary())
    data.explore(columns)
    
# Creates a function to test the cleaning, and exploring of the text DataSet subclass
def TextDataSetTests(filename, column):
    print("===========================================================\n")
    print("TextDataSet object:\n")
    data = TextDataSet(filename)
    data.clean(column)
    print(data.summary())
    data.explore(column)
    
# Creates a function to test the cleaning, and exploring of the time series DataSet subclass
def TimeSeriesDataSetTests(filename, size, columns):
    print("===========================================================\n")
    print("TimeSeriesDataSet object:\n")
    data = TimeSeriesDataSet(filename)
    data.clean(size)
    print(data.summary())
    data.explore(columns)
    
# Runs all of the testing functions
if __name__=="__main__":
    DataSetTests()
    QuantDataSetTests("project/test_data/quant_data.csv", ["Normalized 45", "Normalized 46", "Normalized 47"])
    QualDataSetTests("project/test_data/qual_data.csv", ["Q4", "Q2"])
    TextDataSetTests("project/test_data/text_data.csv", "text")
    TimeSeriesDataSetTests("project/test_data/time_data.csv", 51, ['1.000000000000000000e+00', '3.512396663427352905e-02'])