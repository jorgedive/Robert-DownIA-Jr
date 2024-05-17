import os

import chromadb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def create_embedding(text, embedding_model):
    embedding_model.embed_query(text)


def main():
    df = pd.read_csv(os.path.join(os.getenv("FILES_LOCATION"), "CSV", "cleaned_moves.csv"))[["imdb_id", "description"]]

    chroma_client = chromadb.Client()
