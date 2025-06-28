import streamlit as st
import joblib
import numpy as np

sim = joblib.load(similarity.pkl)
pivot = joblib.load(ratings_matrix.pkl)
movies = joblib.load(movies.pkl)

st.title(🎬 Movie Recommendation System)
user_id = st.number_input(Enter User ID, min_value=1, max_value=len(pivot))

if st.button(Recommend)
    similar_users = np.argsort(-sim[user_id - 1])[5]
    rec_movies = pivot.iloc[similar_users].mean().sort_values(ascending=False).head(5)
    rec_titles = movies[movies['movieId'].isin(rec_movies.index)]['title']
    st.write(Top Recommendations)
    st.write(rec_titles.tolist())
