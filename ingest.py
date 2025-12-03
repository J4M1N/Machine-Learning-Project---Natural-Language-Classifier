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
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch
from peft import LoraConfig, get_peft_model
from trained_model import train_model
from tinyllama import TinyLlama
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def download_dataset(path):
    df = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")

    #new_dataset = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.json"
    
    try:
        df.dropna(how='all', inplace=True) # Drop rows that are completely empty
        df.drop_duplicates(inplace=True) # Remove duplicate rows
        df.to_json(path, index=False) # Save cleaned dataset
        print(f"Cleaned dataset saved to {path}.")
    except Exception as e:
        print(f"Error cleaning dataset: {e}")

     # Save content to file
     
    df.to_json(path, orient='records', indent=4)
if __name__ == "__main__":
    download_dataset("/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/new_dataset.json")



logging.basicConfig(
    level=logging.INFO,  # Set minimum log level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s - Chat History: [%(chat_history)s]',  # Log format
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
            label == int(label)
            emotion == labels_and_emotions.get(emotion)
        while True:
        # Get user input
            emotion_input = input("Enter input here: ").strip()
        
        # Check if user wants to exit
            if emotion_input.lower() == 'exit':
                print("Exiting emotion Classifier. Goodbye!")
            break
        # Classify the emotion
        result = emotion #labels_and_emotions.get
        # Display result
        print(f"Result: {result}")
        return result
    #prompt_user(prompt_user)    

    def chatbot_interface(model_name, response_mapping=None, device=None):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        system_prompt = """<|system|>\nYou are TinyLlama, a friendly assistant who responds based on the emotion results of a classification in natural language.
                            Respond according to the following detected emotion:
                            0: "sadness", Give an empathetic, gentle, supportive Response,
                            1: "joy", Give a cheerful, playful, positive response,
                            2: "love", Give a Romantic response,
                            3: "anger", Give an angry agressive response,
                            4: "fear", Give a nervous, Scared response,
                            5: "surprise", Give a Surprised Response<|end|>\n"""
        
        chat_history = system_prompt + """
            <|user|>
            Hello, how art thou today?<|end|>
            <|assistant|>
            Verily, I am well, kind soul. How fare thee?<|end|>
            <|user|>
            What thinkest thou of the weather?<|end|>
            <|assistant|>
            The heavens weep or smile, as doth the mood of fate. 'Tis fair today, by mine eye.<|end|>
        """

        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
            bias="none"
        )

        model = get_peft_model(model, lora_config)

        # Example training arguments
        training_args = TrainingArguments(
            output_dir="./lora_model",
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_model,  # your dataset here
            args=training_args,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading TinyLlama model '{model_name}' on device '{device}'...")
            model.to(device)
            model.eval()

        classifier_path = 'vectorizer.pkl'
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier file not found at {classifier_path}")
    
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
            #return classifier
        
        chatbot = {
            "model": model_name,
            "classifier": classifier
        }

        if response_mapping is None:
        # Default fallback responses if needed
            response_mapping = {
                'greeting': 'Hello! How can I help you today?',
                'farewell': 'Goodbye! Have a great day.',
                'thanks': 'You are welcome!',
                'fallback': "I'm not sure how to respond to that."
            }

            print("Chatbot is ready! Type 'exit' to quit.")

        model_name = chatbot["model"]
        classifier = chatbot["classifier"]
        #bot = EmotionClassifier()
        #predicted_emotion = self.prompt_user.emotion

        transparent_response = (
        f"I detected that your emotion might be '{classifier}'."
        f"Model evaluation metrics: "
        f"- Accuracy: {accuracy_score:.2f}"
        f"- Precision: {precision_score:.2f}"
        f"- Recall: {recall_score:.2f}"
        f"- F1 Score: {f1_score:.2f}"
    )
                
        # prompts = [
        #     "What is thy name??."
        #     "What does one think about AI?"
        # ]

        # for prompt in prompts:
        #print(system_prompt)
        while True:
            user_input = input("You: ").strip()
            #reply = outputs(system_prompt)
            print(transparent_response.system_prompt)
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break

        # Encode input and generate response from TinyLlama
            inputs = tokenizer.encode(user_input, return_tensors="pt").to(device)
            chat_history_ids = [] #history list
            if inputs.to(device) not in chat_history_ids:
                input_text = inputs.to(device)
            else:
        # First message: only include the current prompt
                input_text = " ".join(chat_history_ids) + " " + inputs.to(device)
       
        #user_tokens = self.tokenizer.encode(input_ids, return_tensors="pt")
            chat_history = chat_history_ids.append(inputs.to(device))
            with torch.no_grad():
                outputs = model.generate(inputs, max_length=150, do_sample=True, temperature=0.7)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    #return response
                print(response)
        
    chatbot_interface(chatbot_interface)
    #print(bot.system_prompt)

    def Rag_memory(question: str): # Function to ask the chatbot with memory
        model_path = "/home/allen11/Machine-Learning-Project---Natural-Language-Classifier/trained_model.pkl"
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        chatbot = TinyLlama(model_path)# Initialize your chatbot
        embeddings = SentenceTransformerEmbeddings(model_name) # Create embeddings for memory

        memory_store = FAISS(embedding_function=embeddings) # Initialize a FAISS vector store for storing conversation memory

        # Create a simple prompt template for RAG
        prompt_template = """
        You are a helpful assistant with memory. Given the following retrieved memory and the input question, provide a response.

        Memory: {context}
        Question: {question}
        Answer: {answer}
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Build the Retrieval QA chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=chatbot,
            retriever=memory_store.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        
        answer = rag_chain.run(question) # Retrieve answer with RAG
    
    # Store this Q&A in memory for future retrieval
        memory_store.add_texts([f"Q: {question}",
                                f"A:{answer}"]
                            )
        return answer
        