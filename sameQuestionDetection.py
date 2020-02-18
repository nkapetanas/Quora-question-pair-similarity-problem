import string

import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import scipy
from sklearn.metrics import f1_score, classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
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

scores_xgboost_accuracy = []
scores_xgboost_precision = []
scores_xgboost_recall = []
scores_xgboost_f1 = []

tfidf_ngram = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)

x_train_data = train_data()
# tfidf_ngram.fit(pd.concat((train_data[QUESTION_1_COLUMN], train_data[QUESTION_2_COLUMN])).unique())
#
# trainq1_trans = tfidf_ngram.transform(train_data[QUESTION_1_COLUMN].values)
# trainq2_trans = tfidf_ngram.transform(train_data[QUESTION_2_COLUMN].values)

y_train_data = train_data['IsDuplicate']
# X = scipy.sparse.hstack((trainq1_trans, trainq2_trans))

kfold = KFold(n_splits=5, random_state=42, shuffle=True)
xgb_model = XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1,
                          colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1,
                          subsample=0.8)
fold = 0

for train_index, test_index in kfold.split(x_train_data):
    fold += 1
    print("Fold: %s" % fold)

    x_train_k, x_test_k = x_train_data.iloc[train_index], x_train_data.iloc[test_index]
    y_train_k, y_test_k = y_train_data.iloc[train_index], y_train_data.iloc[test_index]

    tfidf_ngram.fit_transform(x_train_k)
    tfidf_ngram.fit_transform(x_test_k)

    xgb_model.fit(x_train_k, y_train_k)

    xgb_prediction = xgb_model.predict(x_test_k)

    accuracy, precision, recall, f1 = metrics = calculate_metrics(y_test_k, xgb_prediction)

    scores_xgboost_accuracy.append(accuracy)
    scores_xgboost_precision.append(precision)
    scores_xgboost_recall.append(recall)
    scores_xgboost_f1.append(f1)

    print(classification_report(x_test_k, xgb_prediction))

print("Xgboost metrics")
print("Accuracy:" + str(np.mean(scores_xgboost_accuracy)))
print("Precision:" + str(np.mean(scores_xgboost_precision)))
print("Recall:" + str(np.mean(scores_xgboost_recall)))
print("F1:" + str(np.mean(scores_xgboost_f1)))


testq1_trans = tfidf_ngram.transform(test_data['Question1'].values)
testq2_trans = tfidf_ngram.transform(test_data['Question2'].values)

X_test = scipy.sparse.hstack((testq1_trans, testq2_trans))
xgb_prediction_test = xgb_model.predict(X_test)

createCSV(xgb_prediction_test, "duplicate_predictions.csv")