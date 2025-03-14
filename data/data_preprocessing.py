import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load IMDb dataset
df = pd.read_csv("imdb_top_1000.csv")  # Replace with your dataset path

# Clean data
df = df.dropna(subset=["Genre", "Overview", "IMDB_Rating"])
df["Genre"] = df["Genre"].str.split(", ")  # Split genres into lists
df["Genre"] = df["Genre"].apply(lambda x: " ".join(x))  # Convert to space-separated strings

# Feature Engineering
preprocessor = ColumnTransformer(
    transformers=[
        ("summary", TfidfVectorizer(max_features=500, stop_words="english"), "Overview"),
        ("genre", TfidfVectorizer(max_features=50), "Genre"),
        ("rating", StandardScaler(), ["IMDB_Rating"])
    ]
)

# Pipeline for clustering
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("cluster", KMeans(n_clusters=20, random_state=42))  # Adjust n_clusters as needed
])

# Fit the pipeline
df["cluster"] = pipeline.fit_predict(df)

# Generate synthetic user profiles
synthetic_profiles = []
for cluster_id in df["cluster"].unique():
    cluster_movies = df[df["cluster"] == cluster_id]
    if len(cluster_movies) >= 5:  # Ensure enough movies for input-output pairs
        movies = cluster_movies["Series_Title"].tolist()
        input_text = f"I liked movies like {', '.join(movies[:3])}."
        output_text = f"You might also like {', '.join(movies[3:5])}."
        synthetic_profiles.append({"input": input_text, "output": output_text})

# Save synthetic profiles to CSV
profiles_df = pd.DataFrame(synthetic_profiles)
profiles_df.to_csv("synthetic_profiles_clustered.csv", index=False)
