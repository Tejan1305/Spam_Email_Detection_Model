import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the processed dataset
file_path = 'D:/Spam_Email_Project/processed_dataset.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Ensure the label column is in the correct format
df['label'] = df['label'].astype(int)

# Handle missing values in the 'text' column
df['text'] = df['text'].fillna('')  # Replace NaN values with an empty string

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a CountVectorizer and transform the training data
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train.values)

# Train a Multinomial Naive Bayes model
spam_detector = MultinomialNB()
spam_detector.fit(X_train_count, y_train)

# Save the trained model and vectorizer using pickle
with open('D:/Spam_Email_Project/spam_detector.pkl', 'wb') as model_file:
    pickle.dump(spam_detector, model_file)

with open('D:/Spam_Email_Project/count_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Make predictions on the test data
X_test_count = vectorizer.transform(X_test.fillna(''))  # Ensure no NaN values in test data
y_predict = spam_detector.predict(X_test_count)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)
class_report = classification_report(y_test, y_predict)

# Print the results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
