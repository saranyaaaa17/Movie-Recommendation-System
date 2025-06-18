import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Configuration and Constants ---
# IMPORTANT: Replace 'ml-latest-small/' with the actual relative or absolute path
# where you extracted the MovieLens dataset on your system, if it's not directly
# in the same folder as this script.
# Example if it's in a subfolder: DATA_PATH = 'ml-latest-small/' (Current correct setup)
# Example if it's elsewhere (e.g., C:/datasets/movielens/): DATA_PATH = 'C:/datasets/movielens/ml-latest-small/'
DATA_PATH = 'ml-latest-small/' # This assumes ml-latest-small folder is in the same directory as this script
RATINGS_FILE = DATA_PATH + 'ratings.csv'
MOVIES_FILE = DATA_PATH + 'movies.csv'

# Number of recommendations to provide when calling get_recommendations
TOP_N_RECOMMENDATIONS = 10

# --- Step 1 & 2: Data Collection and Preprocessing ---

print("Step 1 & 2: Loading and Preprocessing Data...")

try:
    # Load ratings and movies data using pandas
    # ratings.csv contains userId, movieId, rating, timestamp
    # movies.csv contains movieId, title, genres
    ratings_df = pd.read_csv(RATINGS_FILE)
    movies_df = pd.read_csv(MOVIES_FILE)

    print(f"Loaded {len(ratings_df)} ratings from '{RATINGS_FILE}'")
    print(f"Loaded {len(movies_df)} movies from '{MOVIES_FILE}'")

    # Merge the two dataframes on the 'movieId' column
    # This combines user ratings with movie titles and genres
    movie_data = pd.merge(ratings_df, movies_df, on='movieId')
    print("Dataframes merged successfully. First 5 rows of merged data:")
    print(movie_data.head())

    # Create the User-Item Matrix
    # This matrix will have 'userId' as rows, 'title' as columns, and 'rating' as values.
    # fill_value=0 is used to replace NaN (missing) values with 0.
    # This implies that an unrated movie has a 'zero' rating, which simplifies similarity calculations
    # for a basic collaborative filtering approach.
    user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    print(f"\nUser-Movie Matrix created. Shape: {user_movie_matrix.shape}")
    print("First 5 rows and columns of User-Movie Matrix:")
    print(user_movie_matrix.iloc[:5, :5]) # Display a small part for inspection

except FileNotFoundError as e:
    print(f"Error: Could not find data files. Please ensure '{DATA_PATH}' exists and contains 'ratings.csv' and 'movies.csv'.")
    print(f"Details: {e}")
    print("Please download the ml-latest-small.zip dataset from https://grouplens.org/datasets/movielens/,")
    print("extract it, and place the 'ml-latest-small' folder in the same directory as this Python script,")
    print("or update the 'DATA_PATH' variable in the script to point to its correct location.")
    exit() # Exit the script if files are not found

# --- Step 3: Choosing a Suitable Algorithm (Item-Item Collaborative Filtering) ---

print("\nStep 3: Calculating Item Similarity (Cosine Similarity)...")

# Calculate the cosine similarity between movies
# We transpose the user_movie_matrix (.T) so that rows represent movies and columns represent users.
# This allows us to calculate the similarity between movie rating vectors.
# Cosine similarity measures the cosine of the angle between two vectors.
# A value closer to 1 indicates higher similarity (ratings patterns are similar).
movie_similarity = cosine_similarity(user_movie_matrix.T)

# Convert the numpy array (similarity matrix) into a pandas DataFrame.
# This makes it easier to work with, allowing us to use movie titles for indexing rows and columns.
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

print(f"Movie Similarity Matrix created. Shape: {movie_similarity_df.shape}")
print("First 5x5 similarities (e.g., how similar certain movies are to others based on user ratings):")
print(movie_similarity_df.iloc[:5, :5])

# --- Step 4: Model Training, Testing, and Recommendation ---

print("\nStep 4: Implementing Recommendation Function and Testing...")

def get_recommendations(user_id, user_movie_matrix_ref, movie_similarity_df_ref, n_recommendations=TOP_N_RECOMMENDATIONS):
    """
    Generates movie recommendations for a given user using Item-Item Collaborative Filtering.

    Args:
        user_id (int): The ID of the user for whom to generate recommendations.
        user_movie_matrix_ref (pd.DataFrame): The user-item matrix.
        movie_similarity_df_ref (pd.DataFrame): The item-item similarity matrix.
        n_recommendations (int): The number of top recommendations to return.

    Returns:
        pd.Series: A Series of recommended movie titles with their predicted ratings,
                   sorted in descending order.
    """
    print(f"\n--- Generating recommendations for User ID: {user_id} ---")

    # Check if the user exists in our user-movie matrix
    if user_id not in user_movie_matrix_ref.index:
        print(f"User ID {user_id} not found in the dataset. Cannot generate recommendations.")
        return pd.Series() # Return an empty Series if user not found

    # Get the row of ratings for the current user from the user-movie matrix.
    user_ratings = user_movie_matrix_ref.loc[user_id]
    
    # Identify movies that the user has already rated (rating > 0).
    # We don't want to recommend movies they've already seen or explicitly rated.
    rated_movies = user_ratings[user_ratings > 0].index.tolist()
    print(f"User {user_id} has rated {len(rated_movies)} movies.")
    # print(f"Rated movies by user {user_id}: {rated_movies[:5]}...") # Uncomment to see some rated movies

    # Initialize a Series to store predicted ratings for all movies.
    # All predicted ratings start at 0.0
    predicted_ratings = pd.Series(0.0, index=user_movie_matrix_ref.columns)
    
    # Initialize a Series to store the sum of similarities for normalization
    sum_of_similarities = pd.Series(0.0, index=user_movie_matrix_ref.columns)

    # Iterate through each movie the current user has rated
    for movie_title in rated_movies:
        # Get the actual rating the user gave to this particular movie
        rating = user_ratings[movie_title]

        # Get the similarity scores for the current movie with all other movies
        # .get() is used to safely retrieve a column; if movie_title is not found
        # in the similarity matrix's columns, it returns an empty Series of zeros.
        similar_movies = movie_similarity_df_ref.get(movie_title, pd.Series(0, index=movie_similarity_df_ref.columns))
        
        # Filter out self-similarity (similarity of a movie with itself is 1, but we don't want it to influence others)
        # Also, filter out movies that the user has already rated from the similar_movies list
        # We want to use the similarity of the rated movie to predict ratings for UNRATED movies.
        similar_movies_for_prediction = similar_movies.drop(rated_movies, errors='ignore')

        # Calculate the contribution of this rated movie to the predicted ratings of all other UNRATED movies.
        # Predicted_rating_for_unrated_movie = sum(Similarity(rated_movie, unrated_movie) * Rating(rated_movie)) / sum(abs(Similarity(rated_movie, unrated_movie)))
        # We sum up the product of similarity and actual rating for each rated movie.
        predicted_ratings += (similar_movies_for_prediction * rating).reindex(predicted_ratings.index, fill_value=0)
        
        # Sum up the absolute similarities for normalization (denominator of the weighted average)
        sum_of_similarities += abs(similar_movies_for_prediction).reindex(sum_of_similarities.index, fill_value=0)

    # Normalize the predicted ratings by dividing by the sum of absolute similarities.
    # This prevents movies with many similar but low-rated neighbors from getting artificially high scores.
    # Add a small epsilon to the denominator to prevent division by zero errors.
    predicted_ratings = predicted_ratings / (sum_of_similarities + 1e-6)

    # Filter out movies that the user has already rated from the final recommendation list
    # by setting their predicted rating to 0. This ensures they don't appear in the top N.
    predicted_ratings = predicted_ratings.drop(rated_movies, errors='ignore')

    # Sort the predicted ratings in descending order and get the top N recommendations
    top_recommendations = predicted_ratings.sort_values(ascending=False).head(n_recommendations)

    # Filter out any recommendations with predicted rating 0.0 (meaning no valid prediction could be made)
    top_recommendations = top_recommendations[top_recommendations > 0.0]

    return top_recommendations

# --- Example Recommendation ---
# Let's pick a random user from our dataset to get recommendations for
# user_id should be an integer present in the 'userId' column of ratings_df.
example_user_id = ratings_df['userId'].sample(1, random_state=42).iloc[0] # Get a random user ID for consistent example
# You can also manually set an example_user_id, e.g.:
# example_user_id = 15 # Example: User ID 15

print(f"\nAttempting to generate recommendations for a sample user (ID: {example_user_id})...")
recommended_movies = get_recommendations(example_user_id, user_movie_matrix, movie_similarity_df, TOP_N_RECOMMENDATIONS)

if not recommended_movies.empty:
    print(f"\nTop {TOP_N_RECOMMENDATIONS} recommendations for User ID {example_user_id}:")
    print(recommended_movies)
else:
    print(f"\nCould not generate recommendations for User ID {example_user_id}. This might happen if the user has not rated enough movies, or if no similar movies could be found.")


# --- Basic Model Evaluation (RMSE) ---

print("\n--- Performing Basic Model Evaluation (RMSE) ---")

# To evaluate the predictive accuracy, we split the original dataset into training and testing sets.
# The training set will be used to build the user-item matrix and similarity matrix.
# The testing set will be used to compare our predicted ratings against the actual ratings that the user gave.
# test_size=0.2 means 20% of the data will be used for testing, 80% for training.
# random_state ensures reproducibility of the split.
train_data, test_data = train_test_split(movie_data, test_size=0.2, random_state=42)

print(f"Split data into {len(train_data)} training ratings and {len(test_data)} testing ratings.")

# Create user-movie matrix and movie similarity matrix using ONLY the training data
train_user_movie_matrix = train_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
train_movie_similarity = cosine_similarity(train_user_movie_matrix.T)
train_movie_similarity_df = pd.DataFrame(train_movie_similarity, 
                                         index=train_user_movie_matrix.columns, 
                                         columns=train_user_movie_matrix.columns)

print(f"Training matrices built. Train user-movie matrix shape: {train_user_movie_matrix.shape}")
print(f"Train movie similarity matrix shape: {train_movie_similarity_df.shape}")


def predict_rating_for_evaluation(user_id, movie_title, user_movie_matrix_ref, movie_similarity_df_ref):
    """
    Predicts the rating for a specific movie by a specific user using the training data.
    This is a helper function specifically for RMSE evaluation.

    Args:
        user_id (int): The ID of the user.
        movie_title (str): The title of the movie.
        user_movie_matrix_ref (pd.DataFrame): The user-item matrix from training data.
        movie_similarity_df_ref (pd.DataFrame): The item-item similarity matrix from training data.

    Returns:
        float: The predicted rating, or np.nan if prediction cannot be made.
    """
    # Check if user or movie exists in the training data's matrix columns/index
    if user_id not in user_movie_matrix_ref.index or movie_title not in user_movie_matrix_ref.columns:
        return np.nan # Cannot predict if user or movie was not in training set

    user_ratings = user_movie_matrix_ref.loc[user_id]
    
    # Get similarity scores for the target movie with all other movies
    # If the target movie is not in the training similarity matrix (e.g., it's a new movie only in test set),
    # we cannot make a prediction based on item-item similarity.
    if movie_title not in movie_similarity_df_ref.columns:
        return np.nan

    target_movie_similarities = movie_similarity_df_ref[movie_title]

    # Filter out movies that the user rated 0 in the training data (i.e., didn't rate)
    # We only consider movies the user HAS rated in the training set to calculate the prediction.
    rated_movies_by_user = user_ratings[user_ratings > 0]

    numerator = 0.0
    denominator = 0.0
    
    # Iterate through movies rated by the user in the training set
    for rated_movie, rating in rated_movies_by_user.items():
        # Ensure the rated movie is in our similarity matrix (it should be if it's from training data)
        if rated_movie in target_movie_similarities.index:
            similarity = target_movie_similarities[rated_movie]
            # Accumulate weighted sum of ratings
            numerator += similarity * rating
            # Accumulate sum of absolute similarities for normalization
            denominator += abs(similarity)

    # Avoid division by zero: if no similar rated movies, or all similarities are zero, return nan
    if denominator == 0:
        return np.nan

    return numerator / denominator


# Collect actual and predicted ratings for RMSE calculation
actual_ratings = []
predicted_ratings = []

# Iterate over a sample of the test set to make predictions
print("Predicting ratings for a sample of the test set for RMSE calculation... (This might take some time)")

# We sample the test set to speed up evaluation for larger datasets.
# For a full evaluation, you would iterate through the entire `test_data`.
test_sample = test_data.sample(n=min(5000, len(test_data)), random_state=42) # Adjust sample size as needed

for index, row in test_sample.iterrows():
    user_id = row['userId']
    movie_title = row['title']
    actual_rating = row['rating']

    # Predict the rating using the training-derived matrices
    predicted_rating = predict_rating_for_evaluation(user_id, movie_title, train_user_movie_matrix, train_movie_similarity_df)

    # Only include pairs where a prediction could be successfully made
    if not np.isnan(predicted_rating):
        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)

if len(actual_ratings) > 0:
    # Calculate RMSE (Root Mean Squared Error)
    # RMSE measures the average magnitude of the errors between predicted and actual ratings.
    # A lower RMSE indicates better prediction accuracy.
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print(f"\nRoot Mean Squared Error (RMSE) on a sample of the test set: {rmse:.4f}")
    print("A lower RMSE generally indicates better prediction accuracy for ratings.")
else:
    print("\nNo valid predictions could be made for RMSE evaluation.")
    print("This might happen if the test data contains users/movies not present in the training set,")
    print("or if the sampling size was too small to find overlap.")

print("\n--- Conceptual Deployment Notes ---")
print("To 'deploy' this system, you would typically save the 'movie_similarity_df' and potentially the 'user_movie_matrix' (or parts of it) to disk.")
print("The 'pickle' library is commonly used for this in Python:")
print("\nExample of saving:")
print("import pickle")
print("with open('movie_similarity_df.pkl', 'wb') as f:")
print("    pickle.dump(movie_similarity_df, f)")
print("\nExample of loading:")
print("with open('movie_similarity_df.pkl', 'rb') as f:")
print("    loaded_similarity_df = pickle.load(f)")
print("\nIn a real-world application, this loaded data would be used by a web API (e.g., Flask/FastAPI) to serve recommendations.")
print("You would also need a system to regularly update these matrices as new users and ratings are added.")
