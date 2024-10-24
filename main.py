import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm

class BERTChatbot:
    def __init__(self, model_name='bert-base-uncased'):
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = 128
        self.label_encoder = LabelEncoder()
        
        # Load responses
        with open('responses.json', 'r') as f:
            self.responses = json.load(f)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Encode labels
        self.label_encoder.fit(df['intent'])
        num_labels = len(self.label_encoder.classes_)
        
        # Initialize model after knowing number of labels
        self.model = TFBertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=num_labels
        )
        
        # Convert texts and labels
        texts = df['text'].values
        labels = self.label_encoder.transform(df['intent'])
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Convert to TensorFlow datasets
        train_dataset = self.create_tf_dataset(train_texts, train_labels)
        val_dataset = self.create_tf_dataset(val_texts, val_labels)
        
        return train_dataset, val_dataset
    
    def create_tf_dataset(self, texts, labels, batch_size=16):
        """Create TensorFlow dataset from texts and labels"""
        # Tokenize texts
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            },
            labels
        ))
        
        return dataset.shuffle(1000).batch(batch_size)
    
    def train(self, train_dataset, val_dataset, epochs=5, learning_rate=2e-5):
        """Train the model"""
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, text):
        """Predict intent for a given text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Get prediction
        outputs = self.model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_label = tf.argmax(predictions, axis=-1).numpy()[0]
        
        # Convert to intent label
        predicted_intent = self.label_encoder.inverse_transform([predicted_label])[0]
        confidence = float(predictions[0][predicted_label])
        
        return predicted_intent, confidence
    
    def get_response(self, text):
        """Get response for user input"""
        intent, confidence = self.predict(text)
        
        # Add confidence threshold
        if confidence < 0.5:
            return "I'm not quite sure what you're asking. Could you rephrase that?"
        
        return self.responses.get(intent, "I'm sorry, I don't understand. Could you rephrase that?")
    
    def evaluate_sample(self, text):
        """Evaluate a sample text with detailed prediction information"""
        intent, confidence = self.predict(text)
        print(f"Text: {text}")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Response: {self.get_response(text)}")

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('chatbot_data.csv')
    
    # Initialize chatbot
    chatbot = BERTChatbot()
    
    # Prepare data
    train_dataset, val_dataset = chatbot.prepare_data(df)
    
    # Train model
    history = chatbot.train(train_dataset, val_dataset)
    
    # Test some examples
    test_texts = [
        "How do I install Python?",
        "What is Python used for?",
        "How do I fix this error in my code?",
        "Which IDE should I use?",
    ]
    
    print("\nTesting the chatbot:")
    for text in test_texts:
        print("\n" + "="*50)
        chatbot.evaluate_sample(text)