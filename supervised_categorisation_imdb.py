#Is the movie review positive or negative? 

# Import standard packages 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score 


from sklearn.preprocessing import StandardScaler
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
df = pd.read_csv('imdb.csv', encoding = "ISO-8859-1")


def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    df['length'] = df['processed'].apply(lambda x: len(x))
    df['text']= df['processed']
    df.drop(['processed'], axis = 1)
    return(df)

df = processing(df)

print(df.head())



features= [c for c in df.columns.values if c  not in ['id', 'length']]
numeric_features= [c for c in df.columns.values if c  not in ['id', 'text']]
target = 'score'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
X_train.head()


# Transform the training data using only the 'text' column values: count_train
text_pipeline = Pipeline([
    ('selector', TextSelector(key='text')),

    ('vec', CountVectorizer(analyzer ='word', ngram_range = (1,3))),#count vectorizer to create a dtm. Tokenize on words
#    ('tfidf', TfidfTransformer()), # term frequency–inverse document frequency - not usefull for text 
    ('clf', SGDClassifier(max_iter = 5)), #classifier
])

num_pipeline = Pipeline([
    ('selector', NumberSelector(key='length')),
    ('standard', StandardScaler())
])


fu_pipeline = FeatureUnion([('text', text_pipeline), ('length', num_pipeline)])

#print what the pipeline is doing 
print(fu_pipeline.fit_transform(X_train, y_train))

#print an accuracy score - even though Theo wants me to 5-fold cross validate lets just do this to check 
print('pipeline score is: ' + str(fu_pipeline.score(X_test, y_test)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(fu_pipeline, X, y, cv = 5)

# Print the 5-fold cross-validation scores
print('All cross validation scores: '+ str(cv_scores))

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))