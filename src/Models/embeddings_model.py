import os
from ast import literal_eval

import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def initialise_db():
    chromadb_client = chromadb.PersistentClient(os.path.join(os.getenv("FILES_LOCATION"), 'ChromaDB'))
    return chromadb_client.get_or_create_collection("film_embeddings")


def main():
    csv_path = os.path.join(os.getenv("FILES_LOCATION"), "CSV")
    df_in_batches = pd.read_csv(os.path.join(csv_path, "cleaned_movies.csv"),
                                usecols=["title", "genres", "description"], low_memory=False)

    df_in_batches['description'] = "Description: " + df_in_batches['description'] + " Genres: " + df_in_batches[
        'genres'].apply(
        lambda x: ", ".join(literal_eval(x)))
    df_in_batches.drop(columns=['genres'], inplace=True)

    collection = initialise_db()
    model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = model.embed_documents(list(df_in_batches['description']))
    collection.add(ids=df_in_batches['title'].tolist(), embeddings=embeddings)


if __name__ == "__main__":
    main()
