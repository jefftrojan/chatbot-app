# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize FastAPI app
app = FastAPI()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('crops.json') as json_file:
    intents = json.load(json_file)

# Load pre-trained model and supporting files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

class UserInput(BaseModel):
    question: str

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    return embeddings

def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    ERROR_THRESHOLD = 0.20
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if tag in i['tags']:
            return random.choice(i['responses']), intents_json['intents'].index(i), tag
    return "I do not know about it", -1, ""

@app.post("/predict/")
async def predict(user_input: UserInput):
    """
    Predicts the intent and generates a response for a given input message.
    """
    message = user_input.question.lower()
    intents_list = predict_class(message)
    if intents_list:
        response, index, intent = get_response(intents_list, intents)
        return {"response": response, "index": index, "intent": intent}
    raise HTTPException(status_code=404, detail="No response found")

## EDIT DISTANCE FUNCTION To Correct the misspelled words
def minDis(s1, s2, n, m, dp):
    if n == 0:
        return m
    if m == 0:
        return n
    if dp[n][m] != -1:
        return dp[n][m]
    if s1[n - 1] == s2[m - 1]:
        if dp[n - 1][m - 1] == -1:
            dp[n][m] = minDis(s1, s2, n - 1, m - 1, dp)
            return dp[n][m]
        else:
            dp[n][m] = dp[n - 1][m - 1]
            return dp[n][m]
    else:
        if dp[n - 1][m] != -1:
            m1 = dp[n - 1][m]
        else:
            m1 = minDis(s1, s2, n - 1, m, dp)
        if dp[n][m - 1] != -1:
            m2 = dp[n][m - 1]
        else:
            m2 = minDis(s1, s2, n, m - 1, dp)
        if dp[n - 1][m - 1] != -1:
            m3 = dp[n - 1][m - 1]
        else:
            m3 = minDis(s1, s2, n - 1, m - 1, dp)
        dp[n][m] = 1 + min(m1, min(m2, m3))
        return dp[n][m]

# Optional endpoint for edit distance (spell check)
@app.post("/spell_check/")
async def spell_check(word1: str, word2: str):
    n, m = len(word1), len(word2)
    dp = [[-1 for _ in range(m + 1)] for __ in range(n + 1)]
    distance = minDis(word1, word2, n, m, dp)
    return {"edit_distance": distance}
