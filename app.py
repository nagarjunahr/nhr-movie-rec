from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Load and prepare data
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])

# Create utility matrix
movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Compute similarity
movie_similarity = cosine_similarity(movie_matrix.T)
similarity_df = pd.DataFrame(movie_similarity, index=movie_matrix.columns, columns=movie_matrix.columns)

# Map movie_id to title
movie_id_to_title = dict(zip(movies['movie_id'], movies['title']))
title_to_movie_id = {v: k for k, v in movie_id_to_title.items()}

from rapidfuzz import process

def recommend_movies(input_title, top_n=5):
    match, score, _ = process.extractOne(input_title, title_to_movie_id.keys())

    if score < 60:
        return [f"Movie '{input_title}' not found. Try another one."]

    movie_id = title_to_movie_id[match]
    if movie_id not in similarity_df:
        return [f"No data for '{match}'. Try another."]

    similar_scores = similarity_df[movie_id].sort_values(ascending=False)[1:top_n+1]
    recommendations = [movie_id_to_title[mid] for mid in similar_scores.index]
    return recommendations


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie = data.get('movie')
    recommendations = recommend_movies(movie)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
