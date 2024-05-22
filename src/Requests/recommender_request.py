from ast import literal_eval

import numpy as np
from pydantic import BaseModel, field_validator

from src.Models.embeddings_model import initialise_db, initialise_model, main


def are_strings(lst: list) -> bool:
    """ Checks if every item in a list is a string.

    Args:
        lst: A list containing (possibly) only strings.
    Returns:
        A boolean checking if every item in lst is a string.
    """
    return all(isinstance(arg, str) for arg in lst) if len(lst) != 0 else True


class RecommenderRequest(BaseModel):
    description: str
    genres: str

    @field_validator('genres')
    def __must_be_string_list(cls, genres_str) -> list:
        """ Checks the given input can be parsed to a list and if said list is empty and contains only strings.

        Args:
            cls: Instance of RecommenderRequest.
            genres_str: string passed from genres field.
        Returns:
            genres_lst: literal evaluation of genres_lst
        Raises:
            ValueError: if genres_str is not a list or it contains an item that isn't a string.
        """
        genres_lst = literal_eval(genres_str)
        if not isinstance(genres_lst, list) or not are_strings(genres_str):
            raise ValueError('The list must only contain strings.')
        return genres_lst


def get_embd_recommendation(recommendation_request: RecommenderRequest) -> str:
    """ Given a RecommenderRequest, it processes it, embeds it and queries the ChromaDB looking for the 10 closest films
    related with the RecommenderRequest fields embedded.

    Args:
        recommendation_request: a valid instance of RecommenderRequest
    Returns:
        A dictionary containing the 10 films with the most similar embeddings.
    """
    parsed_request = "Description: " + recommendation_request.description + " Genres: " + ", ".join(
        recommendation_request.genres)
    embd_request = initialise_model().embed_query(parsed_request)
    collection = initialise_db()

    if collection.count() == 0:
        main()

    result_embd = collection.query(query_embeddings=np.array([embd_request]))['ids'][0]
    return "\n".join([f"{result_embd.index(e) + 1}º Film: " + str(e) for e in result_embd])
