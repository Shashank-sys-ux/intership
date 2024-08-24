import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Define file paths
train_data_path = 'train_data.txt'
test_data_path = 'test_data.txt'
test_solution_path = 'test_data_solution.txt'

def load_data(file_path, has_plot=True):
    # Load data with ' ::: ' separator
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if has_plot and len(parts) == 4:
                data.append(parts)
            elif not has_plot and len(parts) == 3:
                data.append(parts)
            else:
                print(f"Warning: Skipping line due to incorrect format: {line}")
    columns = ['id', 'title', 'genre', 'plot'] if has_plot else ['id', 'title', 'genre']
    return pd.DataFrame(data, columns=columns)

# Load the data
try:
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path, has_plot=False)
    test_solution = load_data(test_solution_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise

# Check if data is loaded correctly
print("Train Data Sample:")
print(train_data.head())
print("\nTest Data Sample:")
print(test_data.head())
print("\nTest Solution Sample:")
print(test_solution.head())

# Data cleaning
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing
train_data['processed_plot'] = train_data['plot'].apply(preprocess_text)

# Check the processed data
print("\nProcessed Train Data Sample:")
print(train_data[['processed_plot']].head())

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(train_data['processed_plot'])
y_train = train_data['genre']

# Model Training with Multinomial Naive Bayes
nb_model = MultinomialNB()

# Fit the model
nb_model.fit(X_train_tfidf, y_train)

# Since we don't have plots in the test data, we'll use the model to predict on the training data for demonstration
# Normally, we would use unseen data
y_pred = nb_model.predict(X_train_tfidf)

# Evaluation
print("\nClassification Report on Training Data (for demonstration):")
print(classification_report(y_train, y_pred))

# In practice, you should use a validation set or cross-validation to evaluate the model properly
