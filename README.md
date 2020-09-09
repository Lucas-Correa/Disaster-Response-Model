# Disaster-Response-Model

### 1. Installations
  To run this disaster response model you need to have a conda environment installed in you machine. The run.py file will only run in the Udacity provided workspace.
  
### 2. Project Motivation
  This project is motivated by the Data Engeniring module of Udacity Data Scientist course. For this project I needed to creat a disaster response model that reads direct, news and socia media messages and redirect it to the right non profit organization according to the model clasification label. There are 36 possible disaster categories in the dataset. Creating a Machine Learning model that could process text and redirect to the right organization is one of the many applications where Natural Language Processing could be helpfull. 
  
  In this project I used a ETL process to load the data and clean it, a NLP pipeline to build a classification model and report the results for the 36 categories and used Udacity's web based flask app model to add 2 more vizualizations of the data gether in a flask app.
  
### 3. File Descriptions
  The process_data.py runs the ETL pipeline that will clean the data and save a sqlite file with the dataset needed for the model script. The rain_classifier.py will run the Machine Learn pipeline that uses Natural Languague Process to classify the messages into each type of aid.
  
  For the text processing part I used the nltk data to tokineze the text variables, separate parts of speach and separate named entities. I choosed to remove the tokens which represent named entities because they are very specific of each message and do not provide a better adress of what type of aid that message should recive. In order to reduce futher clutter in the text features I removed url's and part of speach of cardinal numbers, symbols, interjection, list marker and foreign words.
  
 Besides the text features I wantted to include a genre of message feature (direct, news or social), so I used the [ColumnTransformer()](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) transformer that append differente types of features in the same set of features. In these case particula, category and text features. 
  
### 4. How to Interact with this project
  1. Run the following commands in the root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`
    
### 5. Licensing, Authors, Acknowledgements, etc.
  The dataset belongs to the [Figure Eight](https://appen.com/).
