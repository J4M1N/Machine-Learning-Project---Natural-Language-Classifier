from sklearn.model_selection import train_test_split
import os
import pandas as pd
import json
import logging
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#def download_dataset(self, data):
    # dataset_path = "/Downloads/sms+spam+collection/SMSSpamCollection"
    # cleaned_data_path = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.csv" 
    # new_dataset = new_dataset.csv
    
parq_df = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
print(parq_df)
new_dataset = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.csv"
X = text
y = label
def download_dataset(parq_df:str, new_dataset:str):
    try:
         # Ensure the directory exists
        os.makedirs(os.path.dirname(new_dataset), exist_ok=True)
        print(f"Downloading dataset from {parq_df}...")
        response = requests.get(parq_df)
        response.raise_for_status() # Raise HTTPError for bad responses

     # Save content to file
        parq_df.to_json(new_dataset, orient='records', lines=True)

        print(f"Dataset successfully downloaded and saved to {new_dataset}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
    except OSError as e:
        print(f"Error saving dataset: {e}")

    try:
        df = pd.read_json(new_dataset)
        df.dropna(how='all', inplace=True) # Drop rows that are completely empty
        df.drop_duplicates(inplace=True) # Remove duplicate rows
        df.to_json(new_dataset, index=False) # Save cleaned dataset
        print(f"Cleaned dataset saved to {new_dataset}.")
    except Exception as e:
        print(f"Error cleaning dataset: {e}")


logging.basicConfig(
    level=logging.INFO,  # Set minimum log level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("script_activity.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

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


# def train_model():
#     with open('new_dataset', 'r' encoding='utf-8') as f:
#         read_object = json.load(f)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

class SmsClassifier:
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
        
    def predict(self, message: str):
        features = self.vectorizer.transform([message]) # Transform the message using the TF-IDF vectorizer
        prediction = self.model.predict(features)[0]  # Predict using the loaded model
        return prediction
