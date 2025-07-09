import requests

def getMovieDetails(movie_name, api_key):
    """
    Fetch movie details from the OMDB API.

    Args:
        movie_name (str): The name of the movie to search for.
        api_key (str): Your OMDB API key.

    Returns:
        dict: A dictionary containing movie details or an error message.
    """
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api_key}"
    response = requests.get(url).json()
    print(response)  # Debugging line to check the response from OMDB API
    if response['Response'] == 'True':
        result = response.get("Plot", "N/A"), response.get("Poster", "N/A")
        plot = result[0]
        poster = result[1]
        return plot, poster
    
    return "N/A", "N/A"  # Return N/A if movie not found or error occurs

if __name__ == "__main__":
    # Example usage
    # Replace 'your_api_key_here' with your actual OMDB API key
    print(f"{getMovieDetails('Inception', '34cc6f04')}")