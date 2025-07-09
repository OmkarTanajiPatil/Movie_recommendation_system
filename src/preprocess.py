import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import logging
from scipy.sparse import lil_matrix, save_npz
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename="src/preprocess.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# Download nltk data
logging.info("📥 Downloading NLTK data...")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


def preprocess_text(text):
    logging.debug(f"🧹 Preprocessing text: {text[:30]}...")
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    token = word_tokenize(text)  # Tokenize the text
    text = " ".join([word for word in token if word not in stop_words])
    return text


logging.info("📖 Reading movies.csv...")
df = pd.read_csv("src/movies.csv")

required_columns = ["title", "overview", "genres", "keywords"]
logging.info(f"🔎 Filtering required columns: {required_columns}")
df = df[required_columns]

logging.info("🧹 Dropping rows with missing values...")
df.dropna(inplace=True)

logging.info("🧩 Combining overview, genres, and keywords into a single text column...")
df["combined"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

data = df[["title", "combined"]]


stop_words = set(stopwords.words("english"))


logging.info("🧼 Cleaning and preprocessing text data...")
data["cleaned_text"] = data["combined"].apply(preprocess_text)

# Vectorization with TF-IDF
logging.info("📊 Vectorizing text with TF-IDF...")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data["cleaned_text"])
# Calculate cosine similarity

logging.info("🤝 Calculating cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert to sparse: keep only top 20 similarities per movie
logging.info("🔎 Sparsifying cosine similarity matrix (top 20 per movie)...")
top_n = 20
n_movies = cosine_sim.shape[0]
sparse_cosine_sim = lil_matrix((n_movies, n_movies))
for i in range(n_movies):
    top_idx = np.argsort(cosine_sim[i])[::-1][1 : top_n + 1]  # Exclude self
    sparse_cosine_sim[i, top_idx] = cosine_sim[i, top_idx]

# Save everything
logging.info("💾 Saving preprocessed data and sparse cosine similarity matrix...")
joblib.dump(data, "src/df_cleaned.pkl")
save_npz("src/cosine_sim.npz", sparse_cosine_sim.tocsr())
logging.info(
    "✅ Preprocessing complete. Data and sparse cosine similarity matrix saved."
)

logging.info("📦 Completed preprocessing")
