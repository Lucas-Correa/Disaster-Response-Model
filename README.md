# Disaster-Response-Model

### 1. Installations
  To run this disaster response model you need to have a conda environment installed in you machine. The run.py file will only run in the Udacity workspace provided 
  in the lesson.
  
### 2. Project Motivation
  This project is motivated by the Data Engeniring module of Udacity Data Scientist course. For this project I needed to creat a disaster response model that reads 
  direct, news and socia media messages and redirect it to the right non profit organization according to the model clasification label. There are 36 possible 
  disaster categories in the dataset. Creating a Machine Learning model that could process text and redirect to the right organization is one of the many applications
  where Natural Language Processing could be helpfull. 
  
  In this project I used a ETL process to load the data and clean it, a NLP pipeline to build a classification model and report the results for the 36 categories and
  used Udacity's web based flask app model to add 2 more vizualizations of the data gether in a flask app.
  
### 3. File Descriptions
  The process_data.py runs the ETL pipeline that will clean the data and the rain_classifier.py will run the Machine Learn pipeline that uses Natural Languague Process 
  to classify the messages into each type of aid.
  
### 4. How to Interact with this project
  First run the the process_data.py scripts, which should be runed with additional arguments specifying filepath for the messages dataset, the categorical dataset and resulting sqlite base,
  respectively.
  
  Afterwards, run the train_classifier.py. Please add the filepath of the sqlite base to be used and the file path of the pickle file that will store the model.
  
### 5. Licensing, Authors, Acknowledgements, etc.
  This project was design entirly by me.
