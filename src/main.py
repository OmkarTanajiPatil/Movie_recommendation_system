import streamlit as st
import json
from recommend import recommend_movies, df
from omdb_utils import getMovieDetails

config = json.load(open('src/config.json'))

# OMDB API key
OMDB_API_KEY = config.get('OMDB_API_KEY', 'your_api_key_here')

st.set_page_config(page_title="Movie Recommendation System", page_icon=":movie_camera:", layout="centered")

st.title(":movie_camera: Movie Recommendation System")

movie_list = sorted(df.title.dropna().unique())
st.markdown("### Select a movie")
selected_movie = st.selectbox("", movie_list)

if st.button("Recommend"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_movies(selected_movie)
        if recommendations is None or recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.success("Top similar movies:")
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = getMovieDetails(movie_title, OMDB_API_KEY)
                
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster != "N/A":
                            st.image(poster, width=100)
                        else:
                            st.image("https://via.placeholder.com/100", caption="No poster available")
                    with col2:
                        st.markdown(f"**{movie_title}**")
                        st.markdown(f"*{plot}*" if plot != "N/A" else "No plot available")
                        