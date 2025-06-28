import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("train.txt", sep=";", names=["text", "label"])
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

joblib.dump(model, "emotion_model.pkl")
joblib.dump(vec, "emotion_vectorizer.pkl")
joblib.dump(le, "emotion_encoder.pkl")
