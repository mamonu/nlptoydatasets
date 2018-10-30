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
#holdoutdf = 

def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    df['length'] = df['processed'].apply(lambda x: len(x))
    df['text']= df['processed']
    df.drop(['processed'], axis = 1)
    return(df)

df = processing(df)

print(df.head())


#list of text AND numeric features
all_features= [c for c in df.columns.values if c  not in ['id', 'score', 'processed']]
#list of numeric features
numeric_features= [c for c in df.columns.values if c  not in ['id', 'text', 'score', 'processed']]
#target variable
target = 'score'
print(numeric_features)
print(all_features)


X_train, X_test, y_train, y_test = train_test_split(df[all_features], df[target], test_size=0.33, random_state=42)
print('X_train is :')
print(X_train.head())


# Transform and classify the training data using only the 'text' column values
text_pipeline = Pipeline([
    ('selector', TextSelector(key='text')),
    ('vec', CountVectorizer(analyzer ='word', ngram_range = (1,3))),#count vectorizer to create a dtm. Tokenize on words
    ('tfidf', TfidfTransformer()), # term frequency–inverse document frequency - not usefull for text 
#    ('clf', SGDClassifier(max_iter = 5)) #classifier THIS ISNT WORKING !!!!!!!!!!!!!!!!!!!
])
print(text_pipeline)

#pipelines are each individualy fitted and transformed
#text_pipeline.fit_transform(X_train)

num_pipeline = Pipeline([
    ('selector', NumberSelector(key='length')),
    ('standard', StandardScaler())
])


#num_pipeline.fit_transform(X_train)


fu_pipeline = FeatureUnion([
	('text', text_pipeline), 
	('length', num_pipeline)
	])

feature_processing = Pipeline([('fu_pipeline', fu_pipeline)])
print(feature_processing)
feature_processing.fit_transform(X_train)


#print what the pipeline is doing 
#print(fu_pipeline.fit_transform(X_train, y_train))

#print an accuracy score - even though Theo wants me to 5-fold cross validate lets just do this to check 
#print('pipeline score is: ' + str(feature_processing.score(X_test, y_test)))

from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('features',fu_pipeline),
    ('classifier', RandomForestClassifier()),
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
print(np.mean(preds == y_test))




# Compute 5-fold cross-validation scores: cv_scores
from sklearn.model_selection import GridSearchCV

hyperparameters = { #'features__text__tfidf__max_df': [0.9, 0.95], # ALSO DOESNT LIKE THESE, PROBABLY BECAUSE I USED TFIDF BEFORE 
                    #'features__text__tfidf__ngram_range': [(1,1), (1,2), (1,3)],
                   'classifier__max_depth': [50, 70],
                    'classifier__min_samples_leaf': [1,2]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv=5)
 
# Fit and tune model
clf.fit(X_train, y_train)

print(clf.best_params_)





#refitting on entire training data using best settings only
clf.refit

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

print(np.mean(preds == y_test))











###############
### Testing ###
###############


#read in the test csv 
df_test = pd.read_csv('imdb.csv', encoding = "ISO-8859-1")

#preprocessing
submission = processing(df_test)
predictions = clf.predict_proba(submission)

preds = pd.DataFrame(data=predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)

#generating a submission file
result = pd.concat([submission[['text']], preds], axis=1)
#result.set_index('id', inplace = True)
print(result.head(10))