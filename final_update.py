# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class TextClassifier:
    def __init__(self, classifier, name):
        self.classifier = classifier
        self.name = name

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f'Classifier: {self.name}')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print('Confusion Matrix:\n', confusion)
        print('Classification Report:\n', report)

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('your_dataset.csv')

# Split the data into features (X) and labels (y)
X = data['Text']
y = data['Sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features as needed

# Transform the text data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize classifiers
classifiers = [
    TextClassifier(MultinomialNB(), 'Naive Bayes'),
    TextClassifier(RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest'),
    TextClassifier(DecisionTreeClassifier(random_state=42), 'Decision Tree'),
    TextClassifier(XGBClassifier(random_state=42), 'XGBoost')
]

# Train and evaluate each classifier
for classifier in classifiers:
    classifier.train(X_train_tfidf, y_train)
    classifier.evaluate(X_test_tfidf, y_test)
