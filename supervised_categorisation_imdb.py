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

#for testing 
from sklearn.metrics import log_loss


from nltk import pos_tag 


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


#reg_ex
import re


import pickle

#Some options that will display the data in the prompt in a more desirable format
pd.set_option('display.max_columns', 32)
pd.set_option('display.width', 800)



from sklearn.base import BaseEstimator, TransformerMixin

# class TextSelector(BaseEstimator, TransformerMixin):
#     """
#     Transformer to select a single column from the data frame to perform additional transformations on
#     Use on text columns in the data
#     """
#     def __init__(self, key):
#         self.key = key

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X[self.key]
    
# class NumberSelector(BaseEstimator, TransformerMixin):
#     """
#     Transformer to select a single column from the data frame to perform additional transformations on
#     Use on numeric columns in the data
#     """
#     def __init__(self, key):
#         self.key = key

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X[[self.key]]



class AdhocStats(BaseEstimator, TransformerMixin):
    """Extract features from each zoopla csv row for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, mycolumn):
        return [{'length': len(text),
                 'num_sentences': text.count('.'),

                }
                for text in mycolumn]


adhocstatspipe = Pipeline([
             
                ('handpickedfeatures', AdhocStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ('classifier', LogisticRegression())     # or any other classifier deemed useful!
    
])



#import the csv file as a dataframe
#encoding option is essential
df_total = pd.read_csv('imdb.csv', encoding = "ISO-8859-1")
df = df_total.iloc[0:int((df_total.shape[0]/5)*4)]
holdoutdf = df_total.iloc[int((df_total.shape[0]/5)*4):-1]

#print(df.tail())
#print(holdoutdf.head())


def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    df['length'] = df['processed'].apply(lambda x: len(x))
    df['text']= df['processed']
    df.drop(['processed'], axis = 1)
    return(df)

df = processing(df)

#print(df.head())


#list of text AND numeric features
#all_features= [c for c in df.columns.values if c  not in ['id', 'score', 'processed']]
#list of numeric features
#numeric_features= [c for c in df.columns.values if c  not in ['id', 'text', 'score', 'processed']]
#target variable
#target = 'score'
#print(numeric_features)
#print(all_features)

X=df['text'].values.astype('U')
y=df['score'].values


X_train, X_test, y_train, y_test = train_test_split(X ,y , test_size=0.25, random_state=42)



#X_train, X_test, y_train, y_test = train_test_split(df[all_features], df[target], test_size=0.25, random_state=42)
#print('X_train is :')
#print(X_train.head())



print(adhocstatspipe)
adhocstatspipe.fit(X_train,y_train)

sc = cross_val_score(adhocstatspipe, X, y, cv=5)

print(sc)
###################################################
#dictvectorizer
###################################################
#Before: CountVectorizer(analyzer ='word', ngram_range = (1,3)) 

#added pos_tagger
# Transform and classify the training data using only the 'text' column values
text_pipeline = Pipeline([
    ('selector', TextSelector(key='text')),
#    ('pos_tagger', pos_tag()),
    ('vec', DictVectorizer()), #count vectorizer to create a dtm. Tokenize on words
#    ('tfidf', TfidfTransformer()), # term frequency–inverse document frequency - not usefull for text?
#    ('clf', SGDClassifier(max_iter = 5)) #classifier THIS ISNT WORKING! because we use the classifier later (but use random forrest)
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
clf = GridSearchCV(pipeline, hyperparameters, cv=4)
 
# Fit and tune model
clf.fit(X_train, y_train)

print(clf.best_params_)





#refitting on entire training data using best settings only
clf.refit

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

print(np.mean(preds == y_test))

################
### PICKLING ###
################

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))




###############
### Testing ###
###############


#Testing the holdout set only
df_test = holdoutdf.reset_index(drop=True)

# load the pickled model from disk
loaded_model = pickle.load(open(filename, 'rb'))


#preprocessing
submission = processing(df_test)
predictions = loaded_model.predict_proba(submission) #was clf

preds = pd.DataFrame(data=predictions, columns = loaded_model.best_estimator_.named_steps['classifier'].classes_) #was clf
#valence_0 is the probability that the review is negative according to the model, valence_1 is the probability that the review is positive according to the model. 
preds.columns = ['valence_0', 'valence_1']
#print(preds.head())

#generating a submission file
result = pd.concat([submission[['text']], preds], axis=1)
#result.set_index('id', inplace = True)
#print(result.head(10))

#Thinking about log loss 
ellell = log_loss(df_test.iloc[:,1], result.iloc[:,2])
print('The log loss for the holdout set is ' + str(ellell))

#SOMEWHERE THERE IS A RANDOM STATE I HAVENT SET 