#Is the baby name for a boy or for a girl? 

# Import standard packages 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

import re



from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

#import the csv file as a dataframe
#encoding option is essential
df = pd.read_csv('babynames.csv', encoding = "ISO-8859-1")
print(df.shape[0]/5)
print(df.head())


#The first and most important step in using TPOT on any data set is to rename the target class/response variable to class.
df.rename(columns={'sex': 'class'}, inplace=True)
print(df.head())

#TPOT only works with numerical data - so also possibly need to change the names with hash 
df['class'] = df['class'].map({'boy':0,'girl':1})
#df['name'] = df['name'].lower()
#df['name'] = map(lambda x: x.hash(), df['name'])

#name - variable to be studied 
X = df['name'].values
#sex - variable we want to be able to determine 
y = df['class'].values #changed to class from sex 

#For now we will create a train test split - later on we will use 5 fold cross validation 
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 53)



# Print the head of df
print(df.head())
##########################
#TPOT classifier
##########################
#Before: SGDClassifier(max_iter = 5)
from tpot import TPOTClassifier
from gensim.sklearn_api import W2VTransformer
# Transform the training data using only the 'text' column values: count_train
pipeline = Pipeline([
    ('vec', W2VTransformer(size=10, min_count=1, seed=1)),
#    ('vec', CountVectorizer(analyzer ='char', ngram_range = (1,2))),#count vectorizer to create a dtm, ngram_range = (1,1) because we want to classify based on the frequency of letters in a name and letters next to a name. 
#    ('tfidf', TfidfTransformer()), # term frequency–inverse document frequency
    ('clf', TPOTClassifier(generations=8, population_size=4, verbosity = 2,  max_eval_time_mins=0.35, config_dict = 'TPOT sparse', max_time_mins = 1.8)), #classifier verbosity is just how much infomation the classifier prints whilst working because it takes ages 
])


print(pipeline)
#print what the pipeline is doing 
pipeline.fit(X_train, y_train)


#print an accuracy score - even though Theo wants me to 5-fold cross validate lets just do this to check 
print('pipeline score is: ') 
print(pipeline.score(X_test, y_test))

# Compute 5-fold cross-validation scores: cv_scores
#cv_scores = cross_val_score(pipeline, X, y, cv = 5)

# Print the 5-fold cross-validation scores
#print('All cross validation scores: '+ str(cv_scores))

#print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))