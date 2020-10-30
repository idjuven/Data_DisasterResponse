import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath = 'disaster_messages.csv', categories_filepath = 'disaster_categories.csv'):
    """
    Load two data sets: messages and categories
    
    Args:
    message_filepath(string): the file path of message data
    categories_filepath(string): the file path of categories data
    
    Return:
    df: merged datafram from two data sets.
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on =['id'])

    
    pass


def clean_data(df):
    """
    This function clean the data
    """
    #split "categories" into sepaarate categories
    categories = df['categories'].str.split(';', expand = True)
    
    #select the first row of categories data
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    
    #rename the columns of 'categories'
    categories.columns = category_colnames
    
    #Convert category values to just number 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        #categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])
    
        # convert column from string to numeric
        #categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].astype(int)
        
        categories.loc[categories['related'] == 2,'related'] = 1
        
        #Replce "categories" in df with new column name
        df = df.drop(['categories'], axis =1, inplace = True)
        
        # concatenate the original dataframe with the new `categories` dataframe
        df = pd.concat([categories, df], axis =1)
        
        # Remove duplicates
        #sum(df.duplicated(df.columns))
        
        #drop duplicates
        df = df.drop_duplicates()
        
        return df
        
    
    pass


def save_data(df, database_filename):
    """
    Args:
    df: dataframe
    data_basefilename: the file path to save file .db
    
    Return:
    None
    
    """
    
    #save the clean dataset into an sqlite database
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DS_message', engine, index=False, if_exists = 'replace')
    pass  


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
