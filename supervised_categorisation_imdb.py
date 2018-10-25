#Is the movie review positive or negative? 

# Import standard packages 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

#import the csv file as a dataframe
#encoding option is essential
df = pd.read_csv('imdb.csv', encoding = "ISO-8859-1")

#review text - variable to be studied 
X = df['text'].values
#score (1 if posiyive 0 if negative) - variable we want to be able to determine 
y = df['score'].values

#For now we will create a train test split - later on we will use 5 fold cross validation 
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 123)



# Print the head of df
print(df.head())


# Transform the training data using only the 'text' column values: count_train
pipeline = Pipeline([
    ('vec', CountVectorizer(analyzer ='word', ngram_range = (1,3))),#count vectorizer to create a dtm. Tokenize on words
#    ('tfidf', TfidfTransformer()), # term frequency–inverse document frequency - not usefull for text 
    ('clf', SGDClassifier(max_iter = 5)), #classifier
])

#print what the pipeline is doing 
print(pipeline.fit(X_train, y_train))

#print an accuracy score - even though Theo wants me to 5-fold cross validate lets just do this to check 
print('pipeline score is: ' + str(pipeline.score(X_test, y_test)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(pipeline, X, y, cv = 5)

# Print the 5-fold cross-validation scores
print('All cross validation scores: '+ str(cv_scores))

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))