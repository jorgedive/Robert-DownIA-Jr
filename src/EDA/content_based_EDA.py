import os
from ast import literal_eval

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nltk.stem.snowball import SnowballStemmer

load_dotenv()

files_path = os.getenv("FILES_LOCATION")
cleaned_movies_csv = os.path.join(files_path, "CSV", "cleaned_movies.csv")
credits_csv = os.path.join(files_path, "CSV", "credits.csv")
keywords_csv = os.path.join(files_path, "CSV", "keywords.csv")

df = pd.read_csv(cleaned_movies_csv, low_memory=False)
credits = pd.read_csv(credits_csv, low_memory=False)
keyw = pd.read_csv(keywords_csv, low_memory=False)

print("df.head(3) = ")
print(df.head(3))
print("credits.head(3) = ")
print(credits.head(3))
print("keyw.head(3) = ")
print(keyw.head(3))

# Inner merge dataFrames on column = id.
cntn_based = df.merge(credits, on="id", how="inner")
cntn_based = cntn_based.merge(keyw, on="id", how="inner").drop_duplicates("id")

print("merged.head() = ")
print(cntn_based.head())

list_converter = lambda iterable: [element["name"] for element in literal_eval(iterable)] if isinstance(
    literal_eval(iterable), list) else []
get_director = lambda series: (element["name"] if element["job"] == "Director" else np.nan for element in
                               literal_eval(series))

cntn_based["director"] = cntn_based["crew"].apply(get_director)
cntn_based["cast"] = cntn_based["cast"].fillna("[]").apply(list_converter)
cntn_based["cast"] = cntn_based["cast"].apply(lambda x: x[:3] if len(x) >= 3 else x)
cntn_based["keywords"] = cntn_based["keywords"].apply(list_converter)
cntn_based = cntn_based.drop(columns=["crew"])


def kw_filter(list_kw, data_frame):
    if data_frame is None:
        raise ValueError("Content dataframe is not provided.")
    if not list_kw:
        return []
    all_keywords = data_frame['keywords'].explode()
    kw_count = all_keywords.value_counts()
    filtered = [word for word in list_kw if kw_count.get(word, 0) > 1]
    return filtered


cntn_based["keywords"] = cntn_based["keywords"].transform(func=lambda x: kw_filter(x, cntn_based))

stemmer = SnowballStemmer("english")
cntn_based["keywords"] = cntn_based["keywords"].transform(lambda kw_list: [stemmer.stem(kw) for kw in set(kw_list)])

cntn_based["director"] = cntn_based["director"].astype("str").transform(
    lambda director: [str.lower(director.replace(" ", ""))])
for col in ("keywords", "cast"):
    cntn_based[col] = cntn_based[col].transform(
        func=lambda iterable: [str.lower(word.replace(" ", "")) for word in iterable])

cntn_based["genres"] = cntn_based["genres"].transform(
    func=lambda iterable: [elem.lower().replace(" ", "") for elem in literal_eval(iterable)])

cntn_based["metadata"] = cntn_based["keywords"] + cntn_based["cast"] + cntn_based["director"] + cntn_based["genres"]
cntn_based["metadata"] = cntn_based["metadata"].transform(func=lambda x: " ".join(x))

cntn_based = cntn_based.drop_duplicates("id")
cntn_based = cntn_based.drop_duplicates("title")

cntn_based.to_csv(os.path.join(files_path, "CSV", "cleaned_content_based.csv"), index=False)
