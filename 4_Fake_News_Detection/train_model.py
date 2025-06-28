import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = pd.read_csv("Fake.csv")  # contains 'title', 'text', 'label'

X = df['text']
y = df['label'].map({'REAL': 0, 'FAKE': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = PassiveAggressiveClassifier()
model.fit(X_train_tfidf, y_train)

joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'fake_news_vectorizer.pkl')
print("✅ Model saved")
