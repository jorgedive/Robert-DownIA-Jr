import os
from pathlib import Path

from src.Exceptions.FormatException import FormatException
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

# Upload environment variables from .env file
load_dotenv()

url_omdb = os.getenv('OMDB_URL') + os.getenv('OMDB_API_KEY') + "&i="

request_key_format = ["Title", "Year", "Rated", "Released", "Runtime", "Genre", "Director", "Writer", "Actors", "Plot",
                      "Language", "Country", "Awards", "Poster", "Ratings", "Metascore", "imdbRating", "imdbVotes",
                      "imdbID", "Type", "DVD", "BoxOffice", "Production", "Website", "Response"]


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


def __film_update_csv(film_dict):
    """
    Creates a dataframe with the information provided by film_dict input parameter,
    and, if a .csv file exists in imdb_csv_file location, then it concatenates the values of the dataframes
    and updates the original dataframe with the given parameter's values, adding one extra row.

    If it does not, then creates a new .csv file in the given location with one row, that being the film_dict
    keys and values.

    :param film_dict:
    :return: None
    :raises FormatException: If the film_dict does not follow the existing csv file column pattern or
    the request_key_format pattern.
    """

    if Path(imdb_csv_file_location).exists():
        historic_films = pd.read_csv(imdb_csv_file_location)
        if list(film_dict.keys()) == list(historic_films.columns):
            df = pd.concat([historic_films, film_dict])
            df.to_csv(imdb_csv_file_location)
        else:
            raise FormatException(list(historic_films.columns))

    elif list(film_dict.keys()) == request_key_format:
        pd.DataFrame(film_dict).to_csv(imdb_csv_file_location)

    else:
        raise FormatException(request_key_format)


def get_film_info(imdb_id):
    """
    Given a imdb_id, first it looks in the local repository for a .csv file in imdb_csv_file_location. If the file does
    not exist, it creates it using __film_search and __film_update_csv functions.
    If the file does exit, it looks for rows sharing the same imdbID. If none exists, then it creates a new row
    in the .csv file using __film_update_csv function with the information received from __film_search function.
    If one row sharing the imdbID exists, it returns said row as a dictionary. If multiple rows sharing that imdbID
    exists, the user must input a text casteable to an integer, that will be reduced modulo the number of rows sharing
    the imdbID and will return that row as a dictionary.

    :param imdb_id: imdb_id
    :return: film dictionary associated to the imdb_id
    """

    try:
        with open(imdb_csv_file_location, 'r') as csv_file:
            historic_films = pd.read_csv(csv_file)

        film = historic_films[historic_films['imdbID'] == imdb_id]

        if film.empty:
            request = __film_search(imdb_id)
            __film_update_csv(request)
            return request

        elif len(film) == 1:
            return film.to_dict(orient='records')[0]

        else:
            for index, row in film.iterrows():
                print(f"index = {index}")
                print(row)
                print("\n")

            row_input = input(
                f"Input an integer from 0 to {len(film) - 1} to choose the row that will be returned: ")

            try:
                row_number = int(row_input)
                row_number %= len(film)
                row_dict = film.iloc[row_number].to_dict()
                return row_dict.to_dict(orient='records')[0]
            except ValueError:
                print(f"User input must casteable as an integer, instead it was {row_input}.")

    except FileNotFoundError:
        try:
            request = __film_search(imdb_id)
            __film_update_csv(request)
            return request

        except RequestException as request_exception:
            print("Invalid API Request. Unable to produce a response.", request_exception)

        except FormatException as format_exception:
            print(format_exception.mensaje)

    except pd.errors.ParserError:
        raise ValueError("Error de análisis al leer el archivo CSV: {}".format(imdb_csv_file_location))

    except RequestException as request_exception:
        print("Invalid API Request. Unable to produce a response.", request_exception)

    except FormatException as format_exception:
        print(format_exception.mensaje)
