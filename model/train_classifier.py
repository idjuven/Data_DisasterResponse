import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import joblib
import string 

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB

#Loading data from file path
def load_data(database_filepath):
    """
    Args:
            database_filepath(str): file path of sqlite database
    Return:
            X(pandas dataframe): Features
            y(pandas dataframe): Targets/Labels
    categories (list): List of categorical columns
    """
    engine = create_engine('sqlite:///' + data_filepath)
    df = pd.read_sql_table('DS_message', engine)
    X = df['message']
    y = df[df.columns[5:]]
    categories = y.columns.tolist()
    
    return X, y, categories
    pass


def tokenize(text):
    """
    Write a tokenization function to process the text data
    
    Args:   
          text: text loaded above
    
    Returns:
          lemmed: tokened, stemmed and lemmed text
    """
    
    # normalized text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    
    #tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]

    #Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return lemmed

    pass


def build_model():
    
    """
    Build the machine learning model
    Args: none
    
    Returns: machine learning model
   
    """
    # machine learning pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),  
])
    
    #Grid Search parameters
    param_grid = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1,2)),
        'vect__max_features': (None, 5000,10000),
        'tfidf__use_idf': (True, False)
    }
    
    cv = GridSearchCV(pipeline, param_grid)
    
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Args:
        model: the machine learning model built above
        X_test: the X test data set
        Y_test: the y test classifications
        category_names: the category names
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    cv.fit(X_train, y_train)

    y_pred = cv.predict(X_test)

    print(classification_report(y_test, y_pred, target_names = categories))

    pass


def save_model(model, model_filepath):
    """
    save the model to file path
    args:
        model: the machine learning model
        model_filepath(str): filepath to save model
        
    Return: None
    """
    joblib.dump(model, model_filepath)
    pass


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
