import json
import os

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer

from src.Requests.recommender_request import RecommenderRequest

load_dotenv()
rdIA_api = FastAPI()


@rdIA_api.post("/embd_recommender")
def question_recommender(recommender_request: RecommenderRequest, num: str = 5) -> str:
    recommender_query = "Description: " + recommender_request.description + "; Genres: " + ", ".join(
        recommender_request.genres)
    vectorizer = TfidfVectorizer()
    vector = vectorizer.transform(recommender_query)

    chroma_client = chromadb.PersistentClient(os.path.join(os.getenv("FILES_LOCATION"), "ChromaDB"))
    embd_collection = chroma_client.get_or_create_collection("recommender_embeddings",
                                                             metadata={"hnsw:space": "cosine"})
    list_titles = embd_collection.query(vector, n_results=int(num))['ids']
    return json.dumps({f"Title_{id_no}": list_titles[id_no] for id_no in range(len(list_titles))}, indent=4)
