import os

import pandas as pd
from dotenv import load_dotenv

from src.Requests.request_open_ai import open_ai_api_request_movie_dict as open_request

load_dotenv()


def create_embedding(text, embedding_model):
    embedding_model.embed_query(text)


def create_embedding_model(model_name="llama3"):
    df = pd.read_csv(os.path.join(os.getenv("FILES_LOCATION"), "CSV", "cleaned_moves.csv"))[["imdb_id", "description"]]

