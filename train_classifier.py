import sys
import nltk
import time
nltk.download(['punkt', 'wordnet','stopwords'])

import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    '''
    The load_data funcition load the table from the sqlite database
    processed in the process_data.py and return the necessary arrays for the
    classificatin algorithm

    INPUT:
    database_filepath = file path recived in the train_classifier.py program

    OUTPUT:
    X = numpy array with the features
    y = numpy array with the classifications labels
    targets = numpay array with the possible category names for classification

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MenssagesCategories',engine)

    X = df.message.values
    y = df.drop(['message','genre','related','id','original'],axis=1)

    return X, y.values, y.columns

def tokenize(text):
    '''
    The tokenize funcition recieve a text and process it by removing
    ponctuation/stopwords, sorting out tokens, normalizing it and lemmatizing.

    INPUT:
    text = string with text

    OUTPUT:
    clean_tokens = array of cleaned tokens

    '''

    #removing ponctuation
    text = re.sub(r'[^\w\s]',' ',text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    #this dict optimize the stopwords search by using it a hash O(1) look
    stop_words = stopwords.words('english')
    stopwords_dict = Counter(stop_words)

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords_dict:
            clean_tokens.append(clean_tok)
        else:
            pass
    return clean_tokens


def build_model():
    '''
    The build_model funcition creates a pipeline for the model and optimize it
    in a GridSearch funcition.

    INPUT:

    OUTPUT:
    model = pipeline object

    '''

    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
                ])

    parameters = {
    'vect__ngram_range': ((1, 1),(1,2)),
    'vect__max_df': (0.5, 0.75),
    #'vect__max_features': (None, 5000, 10000),
    #'clf__estimator__n_estimators': [50, 100, 200],
    #'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv =  GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The evaluate_model funcition uses a given model to predict a given array of
    features. It also show a report that summarizes the main metrics for the
    model.

    INPUT:
    model = pipeline object
    X_test = an array of features
    Y_test = an array of the correct labels classification
    category_names = the name of the labels that the report will show the
                     metrics

    OUTPUT:
    '''

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

    print('Best Parameters:{}'.format(cv.best_params_))


def save_model(model, model_filepath):
    '''
    The evaluate_model funcition uses a given model to predict a given array of
    features. It also show a report that summarizes the main metrics for the
    model.

    INPUT:
    model = pipeline object
    X_test = an array of features
    Y_test = an array of the correct labels classification
    category_names = the name of the labels that the report will show the
                     metrics

    OUTPUT:
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        start_time = time.time()
        print('Training model...')
        model.fit(X_train, Y_train)
        print(time.time() - start_time)
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
