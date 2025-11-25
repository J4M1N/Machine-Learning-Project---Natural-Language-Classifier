
import os
import pandas as pd
import json
import logging
import requests

#def download_dataset(self, data):
    # dataset_path = "/Downloads/sms+spam+collection/SMSSpamCollection"
    # cleaned_data_path = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.csv" 
    # new_dataset = new_dataset.csv
    
parq_df = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
print(parq_df)
new_dataset = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.csv"
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
