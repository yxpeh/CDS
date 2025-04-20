import torch
import numpy as np

def preprocess_genre_input(budget, runtime, viewCount, likeCount, favoriteCount, commentCount, releaseMonth, genre):
    genre_list = ['Drama', 'Romance', 'Horror', 'Thriller', 'Action', 'Adventure',
 'Science Fiction', 'Crime', 'Comedy', 'History', 'War',
 'Mystery', 'Fantasy', 'Family', 'Animation', 'Western', 'Music',
 'Documentary', 'TV Movie']
    
    # Numeric features
    numeric = np.array([
        budget,
        runtime,
        viewCount,
        likeCount,
        favoriteCount,
        commentCount,
        releaseMonth
    ], dtype=np.float32)

    # Genre one-hot
    genre_onehot = np.zeros(len(genre_list), dtype=np.float32)
    if genre in genre_list:
        genre_idx = genre_list.index(genre)
        genre_onehot[genre_idx] = 1.0

    # Concatenate
    combined = np.concatenate([numeric, genre_onehot]).reshape(1, -1)
    return combined

def preprocess_title_overview_input(title, overview):
    # TODO: Replace with your real preprocessing logic
    # For now, return a dummy 10D vector
    vector = np.random.rand(10).astype(np.float32)
    return torch.tensor([vector])

def preprocess_poster_input(poster_path):
    # TODO: Replace with your real preprocessing logic
    # For now, return a dummy 10D vector
    vector = np.random.rand(10).astype(np.float32)
    return torch.tensor([vector])

