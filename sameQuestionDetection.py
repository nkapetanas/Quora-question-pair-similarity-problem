import string

import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/Quora-question-pair-similarity-problem/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/Quora-question-pair-similarity-problem/dataset/test_without_labels.csv"

QUESTION_1_COLUMN = "Question1"
QUESTION_2_COLUMN = "Question2"

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
stop = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def read_dataset(dataset):
    df = pd.read_csv(dataset, encoding='utf-8')
    return df


def clean_data(dataframe, column):
    dataframe[column] = dataframe[column].str.lower()
    dataframe[column] = dataframe[column].str.replace("[^\w\s]", "")
    dataframe[column] = dataframe[column].str.replace("\'ve", "have")
    dataframe[column] = dataframe[column].str.replace("can't", "cannot ")
    dataframe[column] = dataframe[column].str.replace("I'm", "I am ")
    dataframe[column] = dataframe[column].str.replace("\'d", " would ")
    dataframe[column] = dataframe[column].str.replace("\'ll", " will ")
    dataframe[column] = dataframe[column].str.replace("60k", " 60000 ")
    dataframe[column] = dataframe[column].str.replace("e-mail", " email ")
    dataframe[column] = dataframe[column].str.replace("quikly", " quickly ")
    dataframe[column] = dataframe[column].str.replace(" usa ", " America ")
    dataframe[column] = dataframe[column].str.replace(" USA ", " America ")
    dataframe[column] = dataframe[column].str.replace(" uk ", " England ")
    dataframe[column] = dataframe[column].str.replace(" UK ", " England ")
    dataframe[column] = dataframe[column].str.replace("intially", "initially")
    dataframe[column] = dataframe[column].str.replace(" dms ", "direct messages")
    dataframe[column] = dataframe[column].str.replace("demonitization", "demonetization")
    dataframe[column] = dataframe[column].str.replace(" cs ", " computer science ")
    dataframe[column] = dataframe[column].str.replace(" upvotes ", " up votes ")
    dataframe[column] = dataframe[column].str.replace("calender", "calendar")
    dataframe[column] = dataframe[column].str.replace("programing", "programming")
    dataframe[column] = dataframe[column].str.replace("bestfriend", "best friend")
    dataframe[column] = dataframe[column].str.replace("III", "3")
    dataframe[column] = dataframe[column].apply(
        lambda x: ' '.join([item for item in x.split() if item not in punctuation]))
    dataframe[column] = dataframe[column].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))
    dataframe[column] = dataframe[column].apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

def createCSV(prediction, csvName):
    np.savetxt(csvName,
               np.dstack((np.array(test_data["Id"].values), prediction))[0], "%d,%d",
               header="Id,Predicted")


def calculate_metrics(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted, average='macro')
    recall = recall_score(y_actual, y_predicted, average='macro')
    f1 = f1_score(y_actual, y_predicted, average='macro')

    return accuracy, precision, recall, f1


train_data = read_dataset(DATASET_PATH_TRAIN)
test_data = read_dataset(DATASET_PATH_TEST)

# Filling the null values with ' '
train_data = train_data.fillna('')
test_data = test_data.fillna('')
# cleaning of data
clean_data(train_data, QUESTION_1_COLUMN)
clean_data(train_data, QUESTION_2_COLUMN)
clean_data(test_data, QUESTION_1_COLUMN)
clean_data(test_data, QUESTION_2_COLUMN)