import os
import gradio as gr
from Models.content_based import get_recommendation, improved_recommendations
from Models.demographic_filtering import get_demographic_recommendation
import pandas as pd
from ast import literal_eval
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Resizing, Rescaling, RandomRotation, RandomFlip
from tensorflow.keras import Sequential


default_files_path = os.path.join("../Files", "CSV")

df = pd.read_csv(os.path.join(default_files_path, "cleaned_movies.csv"), usecols=["title", "genres"])
unique_genres = set([genre for val in df["genres"].values for genre in literal_eval(val)])
model = load_model(os.path.join("../Files/models/muffin-chihuahua/checkpoint.model.keras"))


def content_recommender(Movie):
    return get_recommendation(Movie, file_path=os.path.join(default_files_path, "cleaned_content_based.csv"),
                              model_path=os.path.join("../Files", "models", "content_based", "tfidf_description"))

def demographic_recommender(Genre):
    return get_demographic_recommendation(Genre, path=os.path.join(default_files_path, "cleaned_movies.csv"))


def get_chihuahua(Image, model=model):

    resize_and_rescale = Sequential([Resizing(128, 128), Rescaling(1. / 255)])
    augment_image = Sequential([RandomFlip("horizontal"), RandomRotation(0.2)])

    model = Sequential([
        resize_and_rescale,
        augment_image,
        model
    ])

    pred = model.predict(Image)
    if pred == 1:
        return "Chihuahua"
    else:
        return "Muffin"


content_based = gr.Interface(
    fn=content_recommender,
    inputs=[gr.Dropdown(choices=[title for title in df["title"].values])],
    outputs=[gr.Textbox(label="Recomendaciones")],
    title="Robert-DownIA-Jr - Tu Recomendador de Confianza",
    description="Selecciona una película del menú para recibir recomendaciones en base a su sinopsis."
)

demographic = gr.Interface(
    fn=demographic_recommender,
    inputs=[gr.Dropdown(choices=list(unique_genres))],
    outputs=[gr.Textbox(label="Recomendaciones")],
    title="Robert-DownIA-Jr - Tu Recomendador de Confianza",
    description="Selecciona un género para recibir sus películas mejor valoradas."
)


muffin_chihuahua = gr.Interface(
    fn=get_chihuahua,
    inputs=gr.Image(),
    outputs=[gr.Textbox(label="¿Muffin o Chihuahua?")],
    title="Robert-DownIA-Jr - Chihuahua... ¿O Muffin?",
    description="Sube una imagen para que nuestra maravillosa IA te ayude a distinguir si es un muffin sabroso o"
                "un chihuahua precioso."
)

app = gr.TabbedInterface([demographic, content_based, muffin_chihuahua],
                         ["Género", "Sinopsis", "Chihuahuas"] )

app.launch(debug=True)
