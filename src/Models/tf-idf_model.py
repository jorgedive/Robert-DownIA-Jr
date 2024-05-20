import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
vector_len = 9823


def init():
    global tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=vector_len)


def process_batch(df):
    try:
        df['description'] = "Description: " + df['description'] + "; Genres: " + df['genres']
        df.drop(columns=['genres'], inplace=True)

        tfidf_embeddings_matrix = tfidf_vectorizer.fit_transform(df['description'])
        tfidf_padded_embeddings = np.zeros((tfidf_embeddings_matrix.shape[0], vector_len))
        tfidf_padded_embeddings[:, :tfidf_embeddings_matrix.shape[1]] = tfidf_embeddings_matrix.toarray()[:,
                                                                        :vector_len]

        batch_data = {'ids': df['title'].values, 'embeddings': tfidf_padded_embeddings,
                      'documents': df['description'].values}

        with open(os.path.join(os.getenv("MODELS_PATH"), 'recommender_embedding', 'embeddings.pickle'), 'ab') as file:
            pickle.dump(batch_data, file)
    except Exception as e:
        print(f"Error processing batch: {e}")


def main():
    csv_path = os.path.join(os.getenv("FILES_LOCATION"), "CSV")
    df_in_chunks = pd.read_csv(os.path.join(csv_path, "cleaned_movies.csv"),
                               usecols=["title", "genres", "description"],
                               chunksize=1000, low_memory=False)

    try:

        with Pool(processes=cpu_count(), initializer=init) as pool:
            pool.map(process_batch, df_in_chunks)

    except Exception as e:
        print(f"Error in main processing: {e}")


if __name__ == "__main__":
    main()
    with open(os.path.join(os.getenv("MODELS_PATH"), 'recommender_embedding', 'embeddings.pickle'), 'rb') as f:
        try:
            while True:
                data = pickle.load(f)

                ids = data['ids']
                embeddings = data['embeddings']
                documents = data['documents']

                print(f"Number of records in batch: {len(ids)}")
                print("First document:", documents[0])
                print("First embedding:", embeddings[0])
                print("\n")


        except EOFError:
            pass
