from string import punctuation

import numpy as np
import pandas as pd
import scipy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

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
    dataframe[column] = dataframe[column].str.replace("(?<=[0-9])\,(?=[0-9])", "")  # removes , between numbers
    dataframe[column] = dataframe[column].str.replace("\$", " dollar ")
    dataframe[column] = dataframe[column].str.replace("\%", " percent ")
    dataframe[column] = dataframe[column].str.replace("\&", " and ")
    dataframe[column] = dataframe[column].str.replace(" J K ", " JK ")
    dataframe[column] = dataframe[column].str.replace(" ios ", " operating system ")
    dataframe[column] = dataframe[column].str.replace(" kms ", " kilometers ")
    dataframe[column] = dataframe[column].str.replace("[0-9]+\.[0-9]+", " 50 ")
    dataframe[column] = dataframe[column].str.replace("\'re", " are ")

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

tfidf_ngram = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)

tfidf_ngram.fit(pd.concat((train_data[QUESTION_1_COLUMN], train_data[QUESTION_2_COLUMN])).unique())

trainq1_trans = tfidf_ngram.transform(train_data[QUESTION_1_COLUMN].values)
trainq2_trans = tfidf_ngram.transform(train_data[QUESTION_2_COLUMN].values)

y_train_data = train_data['IsDuplicate']
x_train_data = scipy.sparse.hstack((trainq1_trans, trainq2_trans))

params_xgb = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
              'gamma': np.linspace(.01, 1, 10, endpoint=True),
              'learning_rate': np.linspace(.01, 1, 10, endpoint=True),
              'reg_lambda': np.linspace(0.01, 10, 20, endpoint=True),
              'max_depth': np.linspace(1, 32, 32, endpoint=True, dtype=int)
              }

cv_xgb = RandomizedSearchCV(XGBClassifier(objective='binary:logistic', random_state=42),
                            param_distributions=params_xgb, cv=5, n_jobs=-1)

cv_xgb.fit(x_train_data, y_train_data)

xgb_model = XGBClassifier(random_state=42,
                          objective='binary:logistic',
                          n_estimators=cv_xgb.best_params_['n_estimators'],
                          gamma=cv_xgb.best_params_['gamma'],
                          learning_rate=cv_xgb.best_params_['learning_rate'],
                          reg_lambda=cv_xgb.best_params_['reg_lambda'],
                          max_depth=cv_xgb.best_params_['max_depth'],
                          n_jobs=-1)
xgb_model.fit(x_train_data, y_train_data)

testq1_trans = tfidf_ngram.transform(test_data['Question1'].values)
testq2_trans = tfidf_ngram.transform(test_data['Question2'].values)

X_test = scipy.sparse.hstack((testq1_trans, testq2_trans))
xgb_prediction_test = xgb_model.predict(X_test)

createCSV(xgb_prediction_test, "duplicate_predictions.csv")
