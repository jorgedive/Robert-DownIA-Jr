# Robert-DownIA-Jr

Film recommendation.

### Requirements

The .env file must contain:

- OMDB_API_KEY: OMDB API Key
- OMDB_URL: OMDB API URL
- From OpenAI the user must configure:
  1. OPENAI_API_KEY: key to connect to the OPENAI API.
  2. OPENAI_EMBD_MODEL. The models available in May 2023 are be:

  - text-embedding-3-large
  - text-embedding-3-small
  - text-embedding-ada-002
- FILES_PATH: The absolute path of a directory in which miscelaneous files will be stored and accessed locally.
- COLLECTION_NAME: The name given to the collection where the embeddings will be stored and accessed.
- KAGGLE_MOVIES: The Kaggle Movies Dataset can be found in
  here: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- KAGGLE_CHIHUAHUA: The Kaggle Chihuahuas vs Muffin Dataset can be found in
  here: https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification

Files directory must contain:

- A CSV directory storing all .csv files.
- A models directory that will contain all the models in local.
- A PNG directory containing all the images that will be downloaded.
- A TXT directory containing TXT files.
- A ChromaDB directory to contain the local ChromaDB to store the embeddings. 
