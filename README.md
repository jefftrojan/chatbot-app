# Crop Diseases Chat-Bot

This project is a conversational chatbot built to help farmers and agricultural advisors identify and manage crop diseases. Using deep learning and natural language processing, the bot leverages BERT embeddings to understand user queries about crop diseases and provides relevant guidance and information.

## Features
- **BERT-based Embeddings**: Utilizes BERT embeddings to interpret and respond accurately to complex queries.
- **Pre-trained and Fine-tuned Model**: The model is fine-tuned on a curated dataset of disease-related conversations, improving its understanding of agricultural terms and scenarios.
- **Spell Correction**: Includes an edit distance algorithm to handle minor spelling errors in user queries.
- **Streamlit Interface**: Provides an interactive, user-friendly interface for easy access to information on crop diseases.


## Dataset Creation

To train the model on understanding agricultural queries, create or collect a dataset of conversational pairs. Each pair should include a user input and a corresponding response. Organize these in a JSON format with various intents (e.g., crop disease names) and respective tags.

## Preprocessing
- Tokenization: Tokenize and lemmatize the input text to reduce vocabulary complexity.
- Normalization: Remove unnecessary characters, convert to lowercase, and handle spelling errors.
- Formatting: Ensure the dataset is in the correct format for training (e.g., in JSON format, with user intents and responses).



## Model Training

- Embedding Generation: Use BERT embeddings for sentence-level understanding.
- Fine-tuning the Model: Fine-tune a pre-trained Transformer model (GPT-2 or similar) on the dataset for better conversational response.
- Hyperparameter Tuning: Experiment with learning rates, batch size, and other hyperparameters to optimize performance.