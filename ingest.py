from sklearn.model_selection import train_test_split
import os
import pandas as pd
import json
import logging
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import fastparquet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys


def download_dataset():
    df = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
    print(df)
    new_dataset = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.json"
    try:
        df.dropna(how='all', inplace=True) # Drop rows that are completely empty
        df.drop_duplicates(inplace=True) # Remove duplicate rows
        df.to_json(new_dataset, index=False) # Save cleaned dataset
        print(f"Cleaned dataset saved to {new_dataset}.")
    except Exception as e:
        print(f"Error cleaning dataset: {e}")

     # Save content to file
    df.to_json(new_dataset, orient='records', indent=4)
download_dataset()



# logging.basicConfig(
#     level=logging.INFO,  # Set minimum log level to INFO
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
#     handlers=[
#         logging.FileHandler("script_activity.log"),  # Log to a file
#         logging.StreamHandler()  # Log to console
#     ]
# )

def logging_data():
   logging.info("Script started")  # Replaces: print("Script started")
    
   try:
      logging.debug("Attempting some debug operation")  # Detailed debug info
      result = 10 / 2
      logging.info(f"Computation successful, result = {result}")  # Info level log
   except Exception as e:
      logging.error("An error occurred", exc_info=True)  # Logs error with traceback
    
      logging.warning("This is a warning message")  # Warning example
      logging.info("Script finished")  # Script completion message



class EmotionClassifier:
    def __init__(self, model_path: str, vectorizer_path: str):
        model_path = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/trained_model.pkl"
        # Load the trained model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Model file not found at: {model_path}")


    # Load the TF-IDF vectorizer
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Vectorizer file not found at: {vectorizer_path}")
        
        
    def predict(self, content: str):
        features = self.vectorizer.transform([content]) # Transform the content using the TF-IDF vectorizer
        prediction = self.model.predict(features)[0]  # Predict using the loaded model
        return prediction


    def prompt_user(self, emotion_input=str):
        print("Emotion Classifier is running. Enter your message below.")
        print("Type 'exit' to quit the program.")
        labels_and_emotions = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        for label, emotion in labels_and_emotions.items():
            if label.isdigit():
                label = int(label)
                return emotion
        while True:
        # Get user input
            emotion_input = input("Enter input here: ").strip()
        
        # Check if user wants to exit
            if emotion_input.lower() == 'exit':
                print("Exiting emotion Classifier. Goodbye!")
            break
        # Classify the emotion
        result = labels_and_emotions.get(emotion)
        # Display result
        print(f"Result: {result}")
        return result
    

    def chatbot_interface():

