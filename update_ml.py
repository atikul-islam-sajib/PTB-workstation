# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import gensim.downloader as api
import numpy as np

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('your_dataset.csv')

# Split the data into features (X) and labels (y)
X = data['Text']
y = data['Sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained Word2Vec embeddings (you can replace 'word2vec-google-news-300' with other embeddings)
word2vec_model = api.load('word2vec-google-news-300')

# Function to create document embeddings from Word2Vec
def doc_embedding_word2vec(text, model, size=300):
    words = text.split()
    words = [word for word in words if word in model.vocab]
    if len(words) > 0:
        return np.mean(model[words], axis=0)
    else:
        return np.zeros(size)

# Create document embeddings for training and test data
X_train_word2vec = np.array([doc_embedding_word2vec(text, word2vec_model) for text in X_train])
X_test_word2vec = np.array([doc_embedding_word2vec(text, word2vec_model) for text in X_test])

# Initialize classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Naive Bayes', MultinomialNB()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42))
]

# Train and evaluate each classifier
for clf_name, classifier in classifiers:
    print(f'\nClassifier: {clf_name}')
    
    # Train the classifier on the training data
    classifier.fit(X_train_word2vec, y_train)

    # Make predictions on the test data
    y_pred = classifier.predict(X_test_word2vec)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:\n', confusion)
    print('Classification Report:\n', report)
