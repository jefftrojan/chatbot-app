import pandas as pd

# Create training data with two columns: text and intent
data = {
    'text': [
        # Basic Python Questions
        "What is Python?",
        "How to start learning Python?",
        "Is Python good for beginners?",
        "What can I do with Python?",
        "Python vs other languages",
        
        # Installation Questions
        "How do I install Python?",
        "How to download Python?",
        "Python installation steps",
        "Install Python on Windows",
        "Python setup on Mac",
        
        # IDE Questions
        "Best Python IDE",
        "How to setup PyCharm",
        "VS Code for Python",
        "Python editor recommendation",
        "Jupyter notebook setup",
        
        # Syntax Questions
        "Python list syntax",
        "How to write functions",
        "Python class definition",
        "Dictionary syntax",
        "Loop examples in Python",
        
        # Error Messages
        "ImportError help",
        "Fix IndentationError",
        "TypeError solution",
        "ModuleNotFoundError",
        "Runtime error help",
        
        # Libraries Questions
        "How to install NumPy",
        "Using Pandas library",
        "TensorFlow installation",
        "Matplotlib tutorial",
        "scikit-learn guide"
    ],
    'intent': [
        # Matching intents for Basic Python
        "basic_python",
        "basic_python",
        "basic_python",
        "basic_python",
        "basic_python",
        
        # Matching intents for Installation
        "installation",
        "installation",
        "installation",
        "installation",
        "installation",
        
        # Matching intents for IDE
        "ide_setup",
        "ide_setup",
        "ide_setup",
        "ide_setup",
        "ide_setup",
        
        # Matching intents for Syntax
        "syntax",
        "syntax",
        "syntax",
        "syntax",
        "syntax",
        
        # Matching intents for Errors
        "error_handling",
        "error_handling",
        "error_handling",
        "error_handling",
        "error_handling",
        
        # Matching intents for Libraries
        "libraries",
        "libraries",
        "libraries",
        "libraries",
        "libraries"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('chatbot_data.csv', index=False)

# Display dataset statistics
print("Dataset Overview:")
print("-" * 50)
print(f"Total number of examples: {len(df)}")
print("\nIntent distribution:")
print(df['intent'].value_counts())
print("\nSample entries:")
print("\nFirst few examples from each intent:")
for intent in df['intent'].unique():
    print(f"\n{intent.upper()}:")
    print(df[df['intent'] == intent]['text'].head(2).to_string())

# Create response templates
responses = {
    "basic_python": "Python is a high-level programming language known for its simplicity and readability. It's great for beginners and widely used in web development, data science, AI, and automation.",
    
    "installation": "To install Python, visit python.org and download the latest version. For Windows, make sure to check 'Add Python to PATH' during installation. For Mac/Linux, Python might already be installed.",
    
    "ide_setup": "Popular Python IDEs include PyCharm (professional), VS Code (lightweight), and Jupyter Notebook (data science). For beginners, I recommend VS Code or PyCharm Community Edition.",
    
    "syntax": "I can help you with Python syntax. Could you specify which aspect you'd like to learn about (functions, classes, loops, etc.)?",
    
    "error_handling": "I see you're encountering an error. Could you share the exact error message and the code that's causing it? This will help me provide more specific guidance.",
    
    "libraries": "Python has many useful libraries. To install a library, use 'pip install library_name' in your terminal. Make sure to activate your virtual environment first if you're using one."
}

# Save responses to JSON
import json
with open('responses.json', 'w') as f:
    json.dump(responses, f, indent=4)

print("\nFiles created:")
print("1. chatbot_data.csv - Training data")
print("2. responses.json - Response templates")

# Show the structure of the CSV
print("\nCSV file structure:")
print(df.head())