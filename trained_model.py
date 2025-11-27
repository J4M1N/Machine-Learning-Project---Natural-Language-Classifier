from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
#import tensorflow as tf
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

RANDOM_SEED = 1276
def train_model():
    df = pd.read_json("new_dataset.json")
    X = np.asarray(df["text"].to_list())
    y = np.asarray(df["label"].to_list())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(X_train, X_test, y_train, y_test)
    vectorizer = CountVectorizer()# we need to vectorize x_train and x_test because they are text that need to be changed to numbers
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=1000) #initiate model - logistic regression from scikit learn
    clf.fit(X_train_vec, y_train) # fit model/ training 
    y_pred = clf.predict(X_test_vec) # make predictions with the model
    accuracy = accuracy_score(y_test, y_pred) #calculate the accuracy of these predictions
    print("\nAccuracy of the Decision Tree:", accuracy)
    precision = precision_score(y_test, y_pred, average='micro')
    print("\nPrecision of the Decision Tree:", precision )
    recall = recall_score(y_test, y_pred, average='micro')
    print("\nRecall of the Decision Tree:", recall)
    f1 = f1_score(y_test, y_pred, average='micro')
    print("\nF1 of the Decision Tree:", f1)
    
    # tree_rules = export_text(clf,)
    # print("\nDecision Tree Rules:")
    # print(tree_rules)
    
    with open('train_model.pkl', 'wb') as f: #save vectorizer to turn new entries to text/strings
        pickle.dump(clf, f)

    with open('vectorizer.pkl', 'wb') as f: #save vectorizer to turn new entries to text/strings
        pickle.dump(vectorizer, f)
    
train_model()