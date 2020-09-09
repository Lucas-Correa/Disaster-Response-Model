import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        The load_data funcition recieve the filepath's of the messages
        and categories dataset, respectively, load the files and merge the in
        a single dataframe

        INPUT:
        messages_filepath = string with filepath of messages data
        categories_filepath = string with filepath of categories data

        OUTPUT:
        df = dataframe with merged datasets
    '''

    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = categories.merge(messages,left_on='id',right_on='id')

    return df


def clean_data(df):
    '''
        The clean_data funcition recieve a dataframe and expand the categorical
        columns into separeted columns, drop duplicates and drop rows that have
        nan for all categorical columns.

        INPUT:
        df = dataframe with merged datasets

        OUTPUT:
        df = cleaned dataframe
    '''

    categories = df.categories.str.split(';',expand=True)
    row = categories.loc[0]
    category_colnames = list(row.apply(lambda x : x[:-2]))
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)

    df.drop_duplicates(inplace=True)

    #rows that have nan for all categorical columns
    to_drop = df[df.iloc[: , 4:].isnull().all(axis=1)].index
    df.drop(to_drop,inplace=True)

    df.replace({'related': 2}, 0,inplace=True)
    df.drop('child_alone',axis=1,inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
        The save_data funcition saves the given dataframe in a sqlite base with
        a table called MenssagesCategories.

        INPUT:
        df = dataframe to be transformed in sqlite base
        database_filename = path with file name to be saved

        OUTPUT:
    '''

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MenssagesCategories', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
