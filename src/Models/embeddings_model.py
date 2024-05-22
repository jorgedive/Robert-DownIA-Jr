import os
from ast import literal_eval

import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def initialise_model():
    """ Initialises OpenAI Embedding with the provided embeddings.

    Returns:
       An OpenAIEmbeddings instance of the embedding model.
    """
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBD_MODEL"), openai_api_key=os.getenv("OPENAI_API_KEY"))


def initialise_db():
    """ Initialises the database and the collection with the given collection name
     in the provided files path. If the path does not exist, then it creates it.

     Returns:
         A ChromaDB collection instance with the cosine similarity function.
    """
    chroma_path = os.path.join(os.getenv("FILES_PATH"), 'ChromaDB')
    if not os.path.exists(chroma_path):
        os.mkdir(chroma_path)

    chromadb_client = chromadb.PersistentClient(os.path.join(os.getenv("FILES_PATH"), 'ChromaDB'))
    return chromadb_client.get_or_create_collection(os.getenv("COLLECTION_NAME"),
                                                    metadata={"hnsw:space": "cosine"})


def main():
    """ Creates all the embeddings from the cleaned movies df and upserts them in a chroma db instance.

    """
    csv_path = os.path.join(os.getenv("FILES_PATH"), "CSV")
    df = pd.read_csv(os.path.join(csv_path, "cleaned_movies.csv"),
                     usecols=["title", "genres", "description"], low_memory=False)

    df['description'] = "Description: " + df['description'] + " Genres: " + df[
        'genres'].apply(
        lambda x: ", ".join(literal_eval(x)))
    df.drop(columns=['genres'], inplace=True)

    collection = initialise_db()
    model = initialise_model()
    embeddings = model.embed_documents(list(df['description']))
    collection.upsert(ids=df['title'].tolist(), embeddings=embeddings)


if __name__ == "__main__":
    main()
