import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

files_path = os.getenv("FILES_LOCATION")
ratings_csv = os.path.join(files_path, "CSV", "ratings.csv")

df = pd.read_csv(ratings_csv, low_memory=False)


def filter_col(dataframe, col, threshold):
    filtered_col = (dataframe[col].value_counts() > threshold)
    return filtered_col.index.tolist()


mov_filtered = filter_col(df, col="movieId", threshold=1000)
user_filtered = filter_col(df, col="userId", threshold=200)

df_filtered = df[(df["movieId"].isin(mov_filtered)) & (df["userId"].isin(user_filtered))]
df_filtered.to_csv(os.path.join(files_path, "CSV", "cleaned_ratings.csv"), index=False)
