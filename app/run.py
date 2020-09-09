import json
import plotly
import pandas as pd
import re

#import nltk
#nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):

    text = re.sub(r'[^\w\s]',' ',text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
        else:
            pass
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MenssagesCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# get words
vect = CountVectorizer(tokenizer=tokenize)
x = vect.fit_transform(df.message)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    most_needed_support_counts = df.iloc[:,4:].sum().sort_values(ascending=False).values
    most_needed_support_names = df.iloc[:,4:].sum().sort_values(ascending=False).index
    
 
    n_words = 10
    words_counts = dict(zip(vect.get_feature_names(), x.sum(0).flat))
    words = sorted(words_counts, key=words_counts.get, reverse=True)[:n_words]
    word_counts = []
    
    for w in words:
        word_counts.append(words_counts.get(w))
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Most needed support
        {
            'data': [
                Bar(
                    x=most_needed_support_names,
                    y=most_needed_support_counts
                )
            ],

            'layout': {
                'title': 'Most needed supports',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Support"
                }
            }
        },
        
        # Most appeared words
        {
            'data': [
                Bar(
                    x=words,
                    y=word_counts
                )
            ],

            'layout': {
                'title': 'Most appeared words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()