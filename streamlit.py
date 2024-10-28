import nltk
import numpy as np
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from streamlit_chat import message
from datetime import datetime

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('crops.json') as json_file:
    intents = json.load(json_file)

# Load pre-trained model and supporting files
words = pickle.load(open('backend/words.pkl', 'rb'))
classes = pickle.load(open('backend/classes.pkl', 'rb'))
model = load_model('backend/chatbotmodel.h5')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

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
    return "I don't have information about that. Could you please rephrase or ask about a different crop disease?", -1, ""

def main_(message: str):
    ints = predict_class(message)
    if ints:
        return get_response(ints, intents)
    return "I don't have information about that. Could you please rephrase or ask about a different crop disease?", -1, ""

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

# Set page config
st.set_page_config(
    page_title="Crop Disease Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS with improved visibility
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7f9;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #218838;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .main-header {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .main-title {
        color: #28a745;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .description-text {
        color: #666;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .feature-list {
        list-style-type: none;
        padding-left: 0;
        margin-top: 1rem;
    }
    .feature-list li {
        padding: 0.5rem 0;
        color: #444;
    }
    .feature-list li:before {
        content: "🌿";
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🌱 Crop Disease Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="description-text">
    Welcome to the Crop Disease Assistant! I'm here to help you identify and manage various crop diseases.
    <ul class="feature-list">
        <li>Disease symptoms and identification</li>
        <li>Treatment recommendations</li>
        <li>Prevention strategies</li>
        <li>Best practices for crop health</li>
    </ul>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main Content Area
col1, col2 = st.columns([3, 2])

with col1:
    # Chat interface
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    def get_text():
        return st.text_input(
            "Ask your question here:",
            "",
            key="input",
            placeholder="e.g., What are the symptoms of tomato blight?"
        )

    # Create a container for the chat interface
    chat_container = st.container()

    with chat_container:
        user_input = get_text()
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            send_button = st.button("Send 📤")
        with col2:
            clear_button = st.button("Clear Chat 🗑️")

        if clear_button:
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['timestamp'] = []
            st.session_state.ind = -1
            st.session_state.question = ""

        if send_button and user_input:
            st.session_state.past.append(user_input)
            st.session_state['timestamp'].append(datetime.now().strftime("%H:%M"))
            
            msg = main_(user_input.lower())
            index = msg[1]
            question = msg[2]
            
            st.session_state.ind = index
            st.session_state.question = question
            st.session_state.generated.append(msg[0])

        # Display chat messages
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="bottts",
                    seed=123
                )
                message(
                    st.session_state['past'][i],
                    is_user=True,
                    key=str(i) + '_user',
                    avatar_style="avataaars",
                    seed=456
                )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    url = 'https://www.aci-bd.com/assets/images/rnd/2023/uai.jpg'
    st.image(url, caption="Fall Armyworm Disease", use_column_width=True)
    
    # Quick Links Section
    st.markdown("### Quick Links")
    st.markdown("""
    - [Common Crop Diseases Guide](#)
    - [Prevention Tips](#)
    - [Treatment Methods](#)
    - [Emergency Contacts](#)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Developed with ❤️ for farmers and agricultural professionals</p>
    <p>For emergencies, please consult with a local agricultural expert.</p>
</div>
""", unsafe_allow_html=True)