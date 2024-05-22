import os
from ast import literal_eval

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from numpy import expand_dims
from tensorflow.keras.models import load_model

from Models.content_based import get_recommendation, improved_recommendations
from Models.demographic_filtering import get_demographic_recommendation
from Requests.recommender_request import RecommenderRequest, get_embd_recommendation

load_dotenv()
files_path = os.getenv("FILES_PATH")

df = pd.read_csv(os.path.join(files_path, "CSV", "cleaned_movies.csv"), usecols=["title", "genres"])


def content_recommender(Movie):
    return get_recommendation(Movie)


def demographic_recommender(Genre):
    return get_demographic_recommendation(Genre)


def improved_recommender(Movie):
    return improved_recommendations(Movie)


def get_chihuahua(Image):
    Image = expand_dims(Image, axis=0)

    model = load_model(os.path.join(files_path, "models", "muffin-chihuahua", "model_0058.keras"))
    pred = model.predict(Image)
    print(pred)
    if pred > 0.5:
        return "Muffin", os.path.join(files_path, "PNG", "annoyed_doggy.jpeg")
    else:
        return "Chihuahua", os.path.join(files_path, "PNG", "happy_doggy.jpeg")


def embedding_recommender(Description, Genres):
    genres_list = Genres.split(', ')
    recc_req = RecommenderRequest(description=Description, genres=str(genres_list))
    return get_embd_recommendation(recc_req)


content_based = gr.Interface(
    fn=content_recommender,
    inputs=[gr.Dropdown(choices=[title for title in df["title"].values])],
    outputs=[gr.Textbox(label="Recommendations:")],
    title="Your Trusty Recommender: Just Like Robert Downey Jr. Always Knows Best",
    description="Select a movie from the menu to receive recommendations based on its synopsis."
    )

demographic = gr.Interface(
    fn=demographic_recommender,
    inputs=[gr.Dropdown(choices=list(set([genre for val in df["genres"].values for genre in literal_eval(val)])))],
    outputs=[gr.Textbox(label="Recommendations:")],
    title="Your Trusty Recommender: Just Like Robert Downey Jr. Always Knows Best",
    description="Select a genre to receive its top-rated movies."
    )

hybrid = gr.Interface(
    fn=improved_recommender,
    inputs=[gr.Dropdown(choices=[title for title in df["title"].values])],
    outputs=[gr.Textbox(label="Recommendations:")],
    title="Your Trusty Recommender: Just Like Robert Downey Jr. Always Knows Best",
    description="Provide a title, and this assistant will recommend a similar movie with good ratings."
    )

muffin_chihuahua = gr.Interface(
    fn=get_chihuahua,
    inputs=gr.Image(),
    outputs=[gr.Textbox(label="¿Muffin or chihuahua?"), "image"],
    title="Robert-DownIA-Jr - Chihuahua... ¿Or Muffin?",
    description="Upload an image so this wonderful AI can help you distinguish whether it is a delicious muffin or an "
                "adorable chihuahua."
    )

embedding_recommender = gr.Interface(
    fn=embedding_recommender,
    inputs=[gr.Textbox(label="Film description:"), gr.Textbox(label="Prefered genres:")],
    outputs=[gr.Textbox(label="Recommendations:")],
    description="Provide a description of what you would like to see and the genres (separated by commas) "
                "and this assistant will recommend a similar movies."
    )

app = gr.TabbedInterface([demographic, content_based, hybrid, muffin_chihuahua, embedding_recommender],
                         ["Genre", "Synopsis", "Hybrid", "Chihuahuas", "Recommender"])

if __name__ == "__main__":
    app.launch(debug=True)
