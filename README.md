Movie Recommender System
1. Introduction

This project implements a Movie Recommendation System using Collaborative Filtering (Item-Item Similarity).
The goal is to recommend movies to users based on the similarity of movie rating patterns in the dataset.

Dataset used: MovieLens (ml-latest-small) – contains 100,836 ratings applied to 9,742 movies by 610 users.

2. Dataset Description

The dataset used in this project is the MovieLens ml-latest-small dataset provided by GroupLens Research
.
It covers user ratings, movie metadata, tags, and links to external databases.

Files in the dataset:

ratings.csv

Format: userId, movieId, rating, timestamp

Contains 100,836 ratings on a 0.5 to 5.0 scale.

This file is the primary input for building the User-Movie Matrix.

movies.csv

Format: movieId, title, genres

Provides movie titles and their associated genres.

Used in this project to display meaningful movie names in recommendations.

tags.csv (not directly used in this project)

Format: userId, movieId, tag, timestamp

Contains free-text user-generated tags (e.g., “thriller”, “Pixar”, “funny”).

Can be used for content-based recommendations in future work.

links.csv (not directly used in this project)

Format: movieId, imdbId, tmdbId

Maps MovieLens movie IDs to IMDb and TMDB for external references.

Useful for linking recommended movies to online databases.

The dataset spans from March 1996 to September 2018 and includes only users who rated at least 20 movies
.

3. Methodology
Step 1: Data Collection

ratings.csv → provides user ratings.

movies.csv → provides movie metadata.

Merged datasets on movieId.

Step 2: Data Preprocessing

Created a User-Movie Rating Matrix with users as rows, movies as columns, and ratings as values.

Missing ratings replaced with 0.

Step 3: Similarity Calculation

Transposed the User-Movie Matrix (rows → movies, columns → users).

Computed cosine similarity to build a Movie Similarity Matrix.

Step 4: Recommendation Function

For a given user:

Identify rated movies.

Predict ratings for unseen movies using weighted similarities.

Return Top-N (default = 10) recommendations.

Step 5: Model Evaluation

Train-test split: 80% train, 20% test.

Predicted ratings for test movies.

Evaluated accuracy using RMSE (Root Mean Squared Error).

4. Results

Successfully generated Top-N movie recommendations per user.

RMSE on a test sample: (value depends on your run, e.g., ~0.9–1.0)

Example Recommendation:

Input: User ID = 15

Output: ["Movie A", "Movie B", "Movie C", ...]

5. Deployment Notes

Save the Movie Similarity Matrix using pickle.

Build an API (Flask/FastAPI) to serve recommendations.

Periodically retrain/update similarity matrix as new ratings arrive.

6. Conclusion

This project demonstrates a basic collaborative filtering recommender system using the MovieLens dataset.
It shows how user preferences can be leveraged to recommend relevant movies.

Future improvements:

Use Matrix Factorization (SVD, ALS) for better accuracy.

Add Content-Based Filtering using tags.csv and genres.

Hybrid models (combine collaborative + content).

Deploy as a web or mobile app.
