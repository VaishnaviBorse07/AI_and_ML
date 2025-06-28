import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df = pd.read_csv("ratings.csv")  # userId, movieId, rating
movies = pd.read_csv("movies.csv")

pivot = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
similarity = cosine_similarity(pivot)

joblib.dump(similarity, 'similarity.pkl')
joblib.dump(pivot, 'ratings_matrix.pkl')
joblib.dump(movies, 'movies.pkl')
print("✅ Similarity matrix and movie info saved")
