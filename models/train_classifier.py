import sys
# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine  
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.metrics import classification_report, confusion_matrix

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        tokens = tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        #print(pos_tags)
        try:
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
            return False
        except:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    
def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names 

def tokenize(text):
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    hashtag_regex = '#[a-zA-Z0-9_]+'
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Extract all the hashtags from the provided text 
    detected_tags = re.findall(hashtag_regex, text)
    
    url_place_holder_string = "urlplaceholder"
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)
    
    # Replace url with a url placeholder string
    for detected_tag in detected_tags:
        text = text.replace(detected_tag, detected_tag[1:])

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    clean_tokens = [ w for w in clean_tokens if re.search(r'[a-zA-Z]+', w)]
    return clean_tokens

def build_model():
    pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    'classifier__estimator__n_estimators': [100],
    'classifier__estimator__learning_rate': [0.3],
}

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=2)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_prediction_test = model.predict(X_test)

    print(classification_report(np.hstack(Y_test),np.hstack(y_prediction_test)))
    
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()