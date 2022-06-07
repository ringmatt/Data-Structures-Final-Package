#* DataSet.py
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

# Loads the pandas package for data wrangling
import pandas as pd
# For qualitative visualizations
from pandas.plotting import parallel_coordinates

# Loads the numpy package for array and mathematical capabilities
import numpy as np

# Loads a median filter for the time series data cleaning
from scipy.signal import medfilt

# Used for visualizations in general
import matplotlib.pyplot as plt
import seaborn as sns

# Gathers stopwords, a tokenizer, a lemmatizer from nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# For plotting most common words in corpus
from nltk import FreqDist
 
lemmatizer = WordNetLemmatizer()

class DataSet:
    '''
    The fundamental dataset object, stores data from CSV's. 
    Can clean and explore data.
    '''
    
    def __init__(self, filename = None):
        '''
        Instantiates the DataSet object. Calls load to request a file.
        A file can be given or else it will be requested.
        '''
        # Load the file
        self.__load(filename)
        
    def __str__(self):
        '''
        Prints the DataSet.
            
        Returns
        -------
        str(self.df) : str
            The output of a Pandas DataFrame object.
        '''
        return str(self.df)
    
    def __len__(self):
        '''
        Returns the number of rows (observations) of the DataSet.
            
        Returns
        -------
        len(self.df) : int
            Number of rows in the DataSet.
        '''
        return len(self.df)
        
    def __readFromCSV(self, filename):
        '''
        Determines the file type then loads accordingly. Doesn't actually
        need to be a csv.

        Parameters
        ----------
        filename : str
            A string representing the file path and name of the dataset.
        '''
        
        # Extract part of filename after the period
        file_type = filename.split(".")[1]
        
        # Checks if the file is a csv, then loads as necessary
        if file_type == "csv":
            self.df = pd.read_csv(filename)
            
        # Checks if the file is an Excel file, then loads as necessary
        elif file_type in ("xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"):
            self.df = pd.read_excel(filename)
            
        # Else, try the general read_table function
        else:
            try:
                
                # Calls functions to load a table
                self.df = pd.read_table(filename)
                
            # Throw an error
            except RuntimeError:
                raise RuntimeError('Incorrect file type, must be csv or Excel')
        
    def __load(self, filename):
        '''
        Asks for a filename, passes to the __readFromCSV file.
        '''       
        if filename == None:
            filename = input("File Path: ")
        # Test if the filename is a string
        if isinstance(filename, str):
            
            # Calls functions to load the data
            self.__readFromCSV(filename)

        # Else, try changing the filename to a string
        else:
            try:
                
                # Calls functions to load the data, 
                # coercing the filename to a string
                self.__readFromCSV(str(filename))
                
            # Throw an error
            except TypeError:
                raise TypeError('File name must be a string')
    
    def clean(self):
        '''
        Cleans the dataset. Not implemented in the ABC DataSet class.
        '''
        
        # Tests whether the method works.
        # REMOVE IN FUTURE UPDATES!
        print("clean method invoked\n")
        
    def explore(self):
        '''
        Explores the dataset. Not implemented in the ABC DataSet class.
        '''
        
        # Tests whether the method works.
        # REMOVE IN FUTURE UPDATES!
        print("explore method invoked\n")

    def summary(self):
        '''
        Displays the data type, any applicable measures of central tendency, the miminum, and maximum.

        Central tendancy is mean for numeric data and mode for categorical.
        Maximum is the greatest value for numeric data and the most common for categorical data. 
        For categorical data, The mode should match the maximum, or else the feature is likely a primary key.
        Finally, minimum is the lowest value for the numeric data and least common value for categorical data.
        '''

        # Setup the summary statistics table using datatypes as the first row
        df_summary = pd.DataFrame([self.df.dtypes], columns = self.df.columns, index = ["Data Type"])

        # Prepare lists to hold each value
        central_tendency_val = []
        max_val = []
        min_val = []

        # Cycle through the columns
        for col in df_summary.columns:

            # Determine if they are numeric or categorical,
            # then calculate their central tendancy, maximum, and minimum values
            if df_summary[col].iloc[0] in (np.dtype('int64'), np.dtype('float64')):
                central_tendency_val.append(self.df[col].mean())
                max_val.append(self.df[col].max())
                min_val.append(self.df[col].min())
            elif df_summary[col].iloc[0] in (np.dtype('O'), np.dtype('<U')):
                central_tendency_val.append(self.df[col].mode()[0])
                max_val.append(self.df[col].value_counts().index[0])
                min_val.append(self.df[col].value_counts().index[-1])

        # Create a new dataset of these measures
        df = pd.DataFrame(
            [central_tendency_val, max_val, min_val], 
            columns = self.df.columns, 
            index = ["Central Tendency", "Maximum", "Minimum"])

        # Return the summary
        return df_summary.append(df)
        
class QuantDataSet(DataSet):
    '''
    A version of the DataSet object built for quantitative data. Will remove
    non-quantitative features and fill in missing values.
    Can clean and explore quantitative data.
    '''
    
    def clean(self, fill_method = "mean"):
        '''
        Cleans the quantitative dataset inplace. A method for filling in 
        missing values can be selected but is defaulted to the mean.
        
        Parameters
        ----------
        fill_method : str
            A string of value "mean", "median", or "mode" defining how missing
            values will be filled. Defaults to "mean".
            
        Returns
        -------
        self.df : Pandas DataFrame
            A cleaned version of the DataSet.
        '''
        
        ## Ensure correct datatype for all columns
        
        # Create a list of columns to ignore
        non_quant_cols = []
        
        # Cycle through each column
        for i in range(0,len(self.df.columns)):
            
            # Check if the column is an int or float
            if self.df.dtypes[i] not in (np.dtype('int64'), np.dtype('float64')):
                
                # Try converting the column to a numeric series
                try:
                    
                    self.df[self.df.columns[i]] = pd.to_numeric(
                        self.df[self.df.columns[i]])
                    
                # Otherwise, ignore it
                except:
                    non_quant_cols.append(i)
        
        ## Handle missing values, allow for variety of approaches
        
        # Create list of columns to clean
        
        cols = [i for i in range(0,len(self.df.columns)) if i not in non_quant_cols]
        
        # Calculate the mean of each column and replace NAs with these values
        if fill_method == "mean":
            
            # Cycle through each column
            for i in cols:
                
                # Calculate mean of the column
                col_mean = self.df[self.df.columns[i]].mean()

                # Replace missing values with the column mean
                self.df[self.df.columns[i]] = self.df[self.df.columns[i]].fillna(col_mean)
                
        # Calculate the median of each column and replace NAs with these values
        elif fill_method == "median":
            
            # Cycle through each column
            for i in cols:
                
                # Calculate median of the column
                col_med = self.df[self.df.columns[i]].median()

                # Replace missing values with the column median
                self.df[self.df.columns[i]] = self.df[self.df.columns[i]].fillna(col_med)
            
        # Calculate the mode of each column and replace NAs with these values
        elif fill_method == "mode":
            
            # Cycle through each column
            for i in cols:
                
                # Calculate mode of the column
                col_mode = self.df[self.df.columns[i]].mode()[0]

                # Replace missing values with the column mode
                self.df[self.df.columns[i]] = self.df[self.df.columns[i]].fillna(col_mode)
            
        else:
            
            # Throw an error
            raise ValueError('Improper value for fill_method. Must be "mean", "median", or "mode".')
        
        # Return the cleaned dataframe
        return self.df
        
    def explore(self, columns):
        '''
        Explores the quantitative dataset. Creates one plot analyzing
        distributions and another correlations.
        
        Parameters
        ----------
        columns : list of strings
            Subset of columns to run the exploration on. Must be less than 5.
        '''
        
        # Check if too many columns were passed
        if len(columns) < 5 and len(columns) > 0:
            
            # If fewer than 5 columns, check if they are numeric
            for col in columns:
                
                # Check if the column is an int or float
                if self.df[col].dtypes not in (np.dtype('int64'), np.dtype('float64')):
                    
                    # Try converting the column to a numeric series
                    try:
                        
                        self.df[col] = pd.to_numeric(self.df[col])
                        
                    # Otherwise, throw an error
                    except:
                        raise TypeError("Columns must be numeric.")

        else:
            raise KeyError("Must be fewer than 5 and more than 0 columns.")

        df = self.df[columns]
        
        # Prep the figure style
        sns.set_style("white")
        sns.despine()

        ## Plots a correlation matrix of the subset of provided features
        self.correlation_matrix(df)
        plt.show()

        # Prep the figure style
        sns.set_style("white")
        sns.despine()
        
        ## Plot KDE plots for each feature
        sns.kdeplot(data = df)
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.title("KDE Plot of Feature Distributions")
        # Show the graph
        plt.show() 
            
    def correlation_matrix(self, df):
        '''
        Creates a correlation matrix of quantitative values.

        Parameters
        ----------
        df : pandas DataFrame
            A dataframe of relevant columns for the correlation matrix.
        '''
        # Compute the correlation matrix
        corr = df.corr(method = "pearson")
        annot = corr.copy()
        
        # create three masks
        r0 = corr.applymap(lambda x: '{:.2f}'.format(x))
        r1 = corr.applymap(lambda x: '{:.2f}*'.format(x))
        r2 = corr.applymap(lambda x: '{:.2f}**'.format(x))
        r3 = corr.applymap(lambda x: '{:.2f}***'.format(x))
            
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5, 'label':'Pearson'},
                                                fmt = "",
                                                annot_kws={"size": 20},
                                                vmin = -1,
                                                vmax = 1,
                                                )
        
        plt.xticks(rotation=55)       
        plt.title("Correlation Matrix")
        
class QualDataSet(DataSet):
    '''
    A version of the DataSet object built for qualitative data.
    Can clean and explore qualitative data.
    '''
    
    def clean(self, fill_method = "mode"):
        '''
        Cleans the qualitative dataset inplace. A method for filling in missing 
        values can be selected but is defaulted to the mode.
        
        Parameters
        ----------
        fill_method : str
            A string of value "mode" or a string to replace missing values 
            with. Defaults to "mode".
            
        Returns
        -------
        self.df : Pandas DataFrame
            A cleaned version of the DataSet, but as a dataframe.
        '''
        
        ## Ensure correct datatype for all columns
        
        self.df = self.df.astype('O')
        
        ## Handle missing values, allow for variety of approaches
    
        # Calculate the mode of each column and replace NAs with these values
        if fill_method == "mode":
            
            # Cycle through each column
            for i in range(0,len(self.df.columns)):
                
                # Calculate mode of the column
                col_mode = self.df[self.df.columns[i]].mode()[0]

                # Replace missing values with the column mode
                self.df[self.df.columns[i]] = self.df[self.df.columns[i]].fillna(col_mode)
        
        # If any other string, set missing values
        elif isinstance(fill_method, str):
            
            self.df = self.df.fillna(fill_method, axis = 1)
        
        # If not a string
        else:
            # Throw an error
            raise ValueError('Improper value for fill_method. Must be "mode" or a replacement string.')
        
        # Return the cleaned DataFrame object
        return self.df
        
    def explore(self, columns):
        '''
        Explores the qualitative dataset. 
        Creates a countplot for the first feature and grouped countplot colored by the second feature.
        
        Parameters
        ----------
        columns : str
            Select the columns to visualize. The first will be the x-axis feature and the second will be the color for the 
            grouped countplot. If more than two columns are passed, all extra columns will be ignored.
        '''
        
        # Check if usable columns were passed   
        try:
            # Subsets to relevant columns
            self.df[columns]
        except KeyError:
            raise KeyError("'columns' must be a list of strings representing two columns in the DataSet.")

        # Prep the figure style
        sns.set_style("white")
        sns.despine()

        ## Plots a countplot of the two features
        plt.viridis()
        sns.countplot(y = columns[0], data = self.df)
        plt.title("Countplot")
        plt.show()

        # Prep the figure style
        sns.set_style("white")
        sns.despine()
        
        ## Plot a heatmap for all features
        plt.viridis()
        sns.countplot(y = columns[0], hue = columns[1], data = self.df)
        plt.title("Grouped Countplot")
        plt.show()
        
class TextDataSet(DataSet):
    '''
    A version of the DataSet object built for text data.
    Can clean and explore text data.
    '''
    
    def clean(self, column, additional_stopwords = [], keep_words = []):
        '''
        Cleans the text dataset inplace. Removes stopwords and 
        lemmatizes tokens.
        
        Parameters
        ----------
        column : str
            A string representing the column containing the text data.

        additional_stopwords : list
            List of strings representing additional stopwords.

        keep_words : list
            List of strings representing NLTK stopwords to NOT remove.
            
        Returns
        -------
        self.df : Pandas DataFrame
            A cleaned version of the DataSet, but as a Pandas DataFrame.
        '''

        # If a string, check if the column exists
        if isinstance(column, str):

            # Ensure the column exists and is of type string
            try:
                self.df[column] = self.df[column].astype(str)
                
            except ValueError:
                ValueError("'column' must be the name of a column in the DataSet.")
        # If not of the correct type, throw an error
        else:
            TypeError("'column' must be of type string or None")
        
        ## Remove stop words and lemmatize the text columns
            
        # Combine NLTK stopwords with added stopword
        nltk_sw = set(stopwords.words("english"))
        sw = set.union(nltk_sw, additional_stopwords)

        # Initalize lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Remove words to keep from the set of stopwords
        for word in keep_words:
            sw.remove(word)

        # Make lowercase
        self.df[column] = self.df[column].str.lower()
        
        # Remove punctuations
        self.df[column] = self.df[column].str.replace("[\'!\"#$%&()*+,-/:;<=>@[\\]^_`{|}~?]", "")

        # Remove newline characters
        self.df[column] = self.df[column].str.replace("\n", " ")

        # Remove stopwords
        self.df[column] = self.df[column].apply(
            lambda x: ' '.join([word for word in word_tokenize(x) if word not in (sw)]))
        
        # Lemmatize
        self.df[column] = self.df[column].apply(
           lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
        
        # Return the cleaned DataFrame object
        return self.df
        
    def explore(self, column, n = 10):
        '''
        Explores the data using a lolipop chart of the most common words and 
        a boxplot showing the number of characters per text in the corpus.
        
        Parameters
        ----------
        column : str
            A string representing the column containing the text data.

        n : int
            Number of most frequent words to plot. Defaults to 10.
        '''

        # If a string, check if the column exists
        if isinstance(column, str):

            # Ensure the column exists and is of type string
            try:
                self.df[column] = self.df[column].astype(str)
                
            except ValueError:
                ValueError("'column' must be the name of a column in the DataSet.")
        # If not of the correct type, throw an error
        else:
            TypeError("'column' must be of type string or None")

        # Create a list of all words in corpus
        words = self.df[[column]].stack().str.split(" ").explode().tolist()

        # Determine most frequent words
        fdist = FreqDist(words)
        common_words = fdist.most_common(n)

        # Prep the figure style
        sns.set_style("white")
        sns.despine()

        # Plot a lolipop chart of the most common words
        plt.hlines(y=range(0,n), xmin=0, xmax= [x[1] for x in common_words], color='skyblue')
        plt.plot([x[1] for x in common_words], range(0,n), "o")

        # Adjust titles and labels
        plt.yticks(range(0,n), [x[0] for x in common_words])
        # plt.xticks(range(0, max([x[1] for x in common_words])+1))
        plt.xlabel("Count of Occurances")
        plt.ylabel("Word")
        plt.title("Most Frequent Words in Corpus")
        plt.show()

        # Prep the figure style
        sns.set_style("white")
        sns.despine()

        # Create a density plot of the number of words per text in the corpus
        self.df["words_per_text"] = self.df[column].apply(lambda x: len(word_tokenize(x)))
        sns.kdeplot(self.df["words_per_text"])
        plt.xlabel("Words per Text")
        plt.title("Distribution of Words per Text")
        plt.show()
        
class TimeSeriesDataSet(DataSet):
    '''
    A version of the DataSet object built for time series data.
    Can clean and explore time series data.
    '''
    
    def clean(self, size = None):
        '''
        Cleans the time series dataset inplace. Uses a median filter with an
        adjustable size.
        
        Parameters
        ----------
        size : int
            An odd integer defining the size of the median filter. 
            Defaults to None.
            
        Returns
        -------
        self.df : Pandas DataFrame
            A cleaned version of the DataSet, but as a Pandas DataFrame.
        '''
        
        ## Ensure correct datatype for all columns
        
        # Create a list of columns to ignore
        non_quant_cols = []
        
        # Cycle through each column
        for i in range(0,len(self.df.columns)):
            
            # Check if the column is an int or float
            if self.df.dtypes[i] not in (np.dtype('int64'), np.dtype('float64')):
                
                # Try converting the column to a numeric series
                try:
                    
                    self.df[self.df.columns[i]] = pd.to_numeric(
                        self.df[self.df.columns[i]])
                    
                # Otherwise, ignore it
                except:
                    non_quant_cols.append(i)
                    
        # Create list of columns to clean
        cols = [i for i in range(0,len(self.df.columns)) if i not in non_quant_cols]
        
        ## Apply a median filter to the data
            
        # Cycle through each column
        for i in cols:
            
            # Run a median filter of the requested size
            self.df[self.df.columns[i]] = medfilt(
                self.df[self.df.columns[i]],
                kernel_size = size)
            
        ################# Handle exceptions here?
        
        # Return the cleaned DataFrame object
        return self.df
        
    def explore(self, columns):
        '''
        Explores the time series dataset. Creates line plots and
        violin plots for up to four features.
        
        Parameters
        ----------
        columns : list of strings
            Subset of columns to run the exploration on. Maximum of four.
        '''

        # Check if one to four columns were passed
        if len(columns) > 0 and len(columns) < 5:
            try:
                # Subsets to relevant columns
                df = self.df[columns]
            except KeyError:
                raise KeyError("'columns' must be a list of strings representing columns in the DataSet.")

        # Return an error for stating that more/less columns are needed
        else:
            raise ValueError("'columns' must contain between 1 and 4 strings.")
            
        # Create a color palette
        palette = plt.get_cmap('Dark2')
        
        # Prep the figure style
        sns.set_style("white")
        sns.despine()

        # Plot selected time series data
        num=0
        for column in df:
            num+=1
            plt.plot(np.arange(0,len(df)), df[column], 
            marker='', 
            color=palette(num), 
            linewidth=0.5, alpha=0.9, label=column)

        # Add legend
        plt.legend(loc=2, ncol=2)
        
        # Add titles
        plt.title("Time Series Plots", fontsize=12)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

        # Prep the figure style
        sns.set_style("white")
        sns.despine()

        # Plot the stacked area
        sns.violinplot(data = df)
        plt.title("Distribution(s) of Time Series")
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.show()
          
# if __name__ == "__main__":
    
#     # Test the quantitative dataset
#     df_quant = QuantDataSet("project/test_data/quant_data.csv")
#     df_quant.clean()
#     df_quant.summary()
#     df_quant.explore(columns = ["Normalized 45", "Normalized 46", "Normalized 47"])
    
#     # Test the qualitative dataset
#     df_qual = QualDataSet("project/test_data/qual_data.csv")
#     df_qual.clean()
#     df_qual.summary()
#     df_qual.explore(columns = ["Q4", "Q2"])
    
#     # Test the text dataset
#     df_text = TextDataSet("project/test_data/text_data.csv")
#     df_text.clean(column = "text")
#     df_text.summary()
#     df_text.explore(column = "text")
    
#     # Test the time series dataset
#     df_time = TimeSeriesDataSet("project/test_data/time_data.csv")
#     df_time.clean(size = 51)
#     df_time.summary()
#     df_time.explore(columns = ['1.000000000000000000e+00', '3.512396663427352905e-02'])