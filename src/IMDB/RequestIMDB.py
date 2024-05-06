import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

# Upload environment variables from .env file
load_dotenv()

url_omdb = os.getenv('OMDB_URL') + os.getenv('OMDB_API_KEY') + "&i="


def __film_search(imdb_id):
    """
    If the request is correct, this method returns a json containing all the information from IMDB.
    {
        "Title":
        "Year":
        "Rated":
        "Released":
        "Runtime":
        "Genre":
        "Director":
        "Writer":
        "Actors":
        "Plot":
        "Language":
        "Country":
        "Awards":
        "Poster":
        "Ratings": [
            {
                "Source":
                "Value":
            },
        ],
        "Metascore":
        "imdbRating":
        "imdbVotes":
        "imdbID":
        "Type":
        "DVD":
        "BoxOffice":
        "Production":
        "Website":
        "Response":
    }

    :param imdb_id: imdb_id
    """
    response = requests.get(url_omdb + imdb_id).json()
    response.raise_for_status()
    return response


imdb_csv_file_location = os.getenv("FILES_LOCATION") + 'CSV/IMDB_Record.csv'


def film_search(imdb_id):
    """
    Creates a dataframe with the information provided by the __film_search function
    and, if a .csv file exists in imdb_csv_file location, then it concatenates the values of the dataframes
    and updates the original dataframe with the previously request's information. If it does not,
    then creates a new csv in the given location with the information

    :param imdb_id: imdb_id
    :return: None
    """
    film_df = pd.DataFrame(__film_search(imdb_id))
    try:
        if Path(imdb_csv_file_location).exists():
            historic_films = pd.read_csv(imdb_csv_file_location)
            df = pd.concat([historic_films, film_df])
            df.to_csv(imdb_csv_file_location)
        else:
            film_df.to_csv(imdb_csv_file_location)
    except RequestException as e:
        print("Error in IMDB API response", e)
