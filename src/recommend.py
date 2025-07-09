import joblib
import logging
from scipy.sparse import load_npz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename="src/recommend.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logging.info("📥 Loading preprocessed data and model...")
# Load preprocessed data and model
try:
    df = joblib.load("src/df_cleaned.pkl")
    cosine_sim_sparse = load_npz("src/cosine_sim.npz")
    logging.info(
        "✅ Successfully loaded preprocessed data and sparse similarity matrix."
    )
except FileNotFoundError as e:
    logging.error(f"❌ Error loading data or model: {e}")
    raise


def recommend_movies(title, cosine_sim=cosine_sim_sparse, df=df, top_n=5):
    idx = df[df["title"].str.lower() == title.lower()].index
    if len(idx) == 0:
        return f"Movie '{title}' not found in the dataset."
    idx = idx[0]
    # Get the row as dense array for top similarities
    sim_scores = cosine_sim[idx].toarray().flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx and x[1] > 0]
    sim_scores = sim_scores[:top_n]

    movies_indices = [i[0] for i in sim_scores]
    logging.info(f"Top {top_n} recommendations for MOVIE: {title}")

    result_df = df[["title"]].iloc[movies_indices].reset_index(drop=True)
    result_df.index += 1  # Start index from 1 for user-friendliness
    result_df.index.name = "S.No."
    return result_df
