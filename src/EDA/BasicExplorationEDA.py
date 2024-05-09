import os
from ast import literal_eval

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

files_path = os.getenv("FILES_LOCATION")

# A Files/CSV/movies_metadata.csv file must exist with the Files folder.
movies_df = pd.read_csv(os.path.join(files_path, "CSV", "movies_metadata.csv"), low_memory=False)

print("df.head(3) = ")
print(movies_df.head(3))
print("df.info() = ")
print(movies_df.info())
print("df.isnull.sum() = ")
print(movies_df.isnull().sum())

# These characteristics hold no interest for this particular notebook, and will be dropped or processed accordingly,
# since the resulting .csv file will encompass the necessary information that needs to be used in recommendation
# systems.
movies_df = movies_df[movies_df["adult"] == "False"].drop(columns=["adult"])
movies_df = movies_df[movies_df["status"] != "Canceled"]
movies_df = movies_df[movies_df["overview"].notnull()]
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors="coerce")
movies_df["popularity"] = pd.to_numeric(movies_df["popularity"], errors="coerce")

movies_df["tagline"] = movies_df["tagline"].fillna("")
movies_df["description"] = movies_df["overview"] + movies_df["tagline"]
relevant_cols = ["genres", "id", "title", "overview", "description", "popularity", "vote_average", "vote_count"]
movies_df = movies_df[relevant_cols]

list_converter = lambda series: [genre["name"] for genre in literal_eval(series)] if isinstance(literal_eval(series),
                                                                                                list) else []

movies_df["genres"] = movies_df["genres"].fillna("[]").transform(func=list_converter)

movies_df.fillna({"popularity": movies_df["popularity"].median(), "vote_average": movies_df["vote_average"].median(),
                  "vote_count": movies_df["vote_count"].median()}, inplace=True)

movies_df = movies_df.drop_duplicates("id")
movies_df = movies_df.drop_duplicates("title").dropna()

# Writes the resulting file in
movies_df.to_csv(os.path.join(files_path, "CSV", "cleaned_movies.csv"), encoding="UTF-8", index=False)
