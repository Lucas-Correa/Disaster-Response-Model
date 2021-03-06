import sys
import time
import pandas as pd
import re
import pickle

from joblib import parallel_backend

from sqlalchemy import create_engine

from collections import Counter

import nltk
nltk.download(['punkt', 'wordnet','stopwords','words','maxent_ne_chunker','averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import SelectPercentile,chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
#from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

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

    X = df[['message','genre']]
    y = df.iloc[:,4:]

    return X, y.values, y.columns

def ne_removal(text):
    '''
        The ne_removal funcition recieves a text corpus, tokenize it, set parts of
        speach and name entities. Then, remove the name entities from the
        tokenize text.

        INPUT:
        text = string with text

        OUTPUT:
        tokens_no_ne = array of tuple with token and part of speach
    '''
    raw_tokens = word_tokenize(text)

    chunked = ne_chunk(pos_tag(raw_tokens))

    tokens_no_ne = [leaf for leaf in chunked if (type(leaf) != nltk.Tree) ]

    return tokens_no_ne


def url_replacer(text):
    '''
        The url_replacer funcition recieve a text corpus, checks for existing
        urls and replace it in the text for a urlplaceholder

        INPUT:
        text = string with text

        OUTPUT:
        text = same string with url's replaced
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    return text


def tokenize(text):
    '''
        The tokenize funcition recieve a text and process it by removing
        ponctuation/stopwords, sorting out tokens, normalizing it and lemmatizing.
        Excluded tokens that were in the following parts of speach:
        CD: cardinal numbers
        SYM: symbols
        UH: interjection
        LS: list marker
        FW: foreign words

        INPUT:
        text = string with text

        OUTPUT:
        clean_tokens = array of cleaned tokens
    '''

    text_no_url = url_replacer(text)

    tokens = ne_removal(text_no_url)

    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words('english')
    stopwords_dict = Counter(stop_words)

    clean_tokens = []
    for idx, tok in enumerate(tokens):

        tok_no_ponct = re.sub(r'[^\w\s]','',tok[0])
        normalized_tok = lemmatizer.lemmatize(tok_no_ponct).lower().strip()

        if (normalized_tok not in stopwords_dict) & (tok[1] not in ['CD','SYM','UH','LS','FW']) & (normalized_tok != ''):
            clean_tokens.append(normalized_tok)
        else:
            pass

    return clean_tokens


def build_model():
    '''
        The build_model funcition creates a pipeline for the model and optimize
        it with GridSearch.

        Firt it is created a text(1) and category(2) pipeline which then are
        combined in the preprocessor(1+2) pipeline. Then the preprocessor
        pipeline is combined with the classifier step in the final pipeline.

        INPUT:

        OUTPUT:
        model = pipeline object
    '''
    #pipeline with the text processing
    text_transformer = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ])

    #pipeline with the categorial variable processing
    categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

    #preprocessor = text + categorical pipelines
    preprocessor = ColumnTransformer([
                ('text_features', text_transformer, 'message'),
                ('cat_features', categorical_transformer, ['genre'])
                ])

    #final pipeline = preprocessor pipeline + classification step
    pipeline = Pipeline([
                ('features', preprocessor),
                #('feature_selection',SelectPercentile(chi2)),
                ('clf', MultiOutputClassifier(AdaBoostClassifier()) )
                ])

    parameters = {
    #'features__text_features__vect__max_df': (0.5, 0.75,1),
    #'clf__estimator__n_estimators': [50, 200],
    #'feature_selection__percentile': [10, 100]
    #'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv =  GridSearchCV(pipeline, param_grid=parameters,verbose=10)#,n_jobs=-1)

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

    print(classification_report(Y_test, y_pred, target_names=category_names,zero_division=1))

    try:
        print('Best Parameters:{}'.format(model.best_params_))
    except:
        pass

def save_model(model, model_filepath):
    '''
        The save_model funcition saves the trained model in a peakcle file

        INPUT:
        model = pipeline object
        model_filepath = string with where the pickle file with the model should be
                        saved

        OUTPUT:
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                                random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start_time = time.time()
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
