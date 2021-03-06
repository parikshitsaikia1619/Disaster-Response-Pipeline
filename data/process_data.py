import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge
    df = pd.merge(messages, categories, on="id")
    
    # create a dataframe of the 36 individual category columns
    categories_new = categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories_new.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x[:-2])
    # rename the columns of `categories`
    categories_new.columns = category_colnames
    
    for column in categories_new:
        # set each value to be the last character of the string
        categories_new[column] = categories_new[column].str[-1]

        # convert column from string to numeric
        categories_new[column] = pd.to_numeric (categories_new[column], errors = 'coerce')
    categories = pd.concat([categories, categories_new], axis =1)
    categories.drop(columns=['categories'], inplace= True)
    df =  pd.merge(df, categories, on="id")
    
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)
    return df


def clean_data(df):
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  


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