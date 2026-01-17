import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['labels', 'message']

df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})

x_train, x_test, y_train, y_test = train_test_split(
    df['message'], df['labels'], test_size=0.2, random_state=42, stratify=df['labels']
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))

sample_messages = [
    "Congratulations! You've won a free iPhone. Click now!",
    "Can we reschedule our meeting to tomorrow afternoon?"
]

predictions = model.predict(sample_messages)
for msg, predict in zip(sample_messages, predictions):
    print(f"Message: {msg}\nPrediction: {'Spam' if predict == 1 else 'Not Spam'}\n")

joblib.dump(model, "spam_model.pkl")
print("Model saved as spam_model.pkl")

