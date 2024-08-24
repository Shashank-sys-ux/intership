import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Inspect the first few rows to understand the structure of the dataset
print(data.head())

# Drop unnecessary columns (assuming only first and second columns are needed)
data = data.iloc[:, [0, 1]]  # Keep only the first and second columns

# Rename columns to 'label' and 'message'
data.columns = ['label', 'message']

# Convert label to binary format (0 for legitimate, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)
y_pred_log_reg = log_reg.predict(X_test_tfidf)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log_reg))

# Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

print("\nNaive Bayes Report:")
print(classification_report(y_test, y_pred_nb))

# Support Vector Machine Model
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train_tfidf, y_train)
y_pred_svc = svc.predict(X_test_tfidf)

print("\nSupport Vector Machine Report:")
print(classification_report(y_test, y_pred_svc))

# Confusion Matrix for Logistic Regression (as an example)
print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_log_reg))
