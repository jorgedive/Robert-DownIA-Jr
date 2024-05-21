import os
import gradio as gr
from Models.content_based import get_recommendation, improved_recommendations
from Models.demographic_filtering import get_demographic_recommendation
import pandas as pd
from ast import literal_eval
from tensorflow.keras.models import load_model
from numpy import expand_dims


default_files_path = os.path.join("../Files", "CSV")

df = pd.read_csv(os.path.join(default_files_path, "cleaned_movies.csv"), usecols=["title", "genres"])
unique_genres = set([genre for val in df["genres"].values for genre in literal_eval(val)])
titles = [title for title in df["title"].values]
model = load_model(os.path.join("../Files/models/muffin-chihuahua/model_0058.keras"))


def content_recommender(Movie):
    return get_recommendation(Movie, file_path=os.path.join(default_files_path, "cleaned_content_based.csv"),
                              model_path=os.path.join("../Files", "models", "content_based",
                                                      "tfidf_description_matrix.pickle"))

def demographic_recommender(Genre):
    return get_demographic_recommendation(Genre, path=os.path.join(default_files_path, "cleaned_movies.csv"))


def improved_recommender(Movie):
    return improved_recommendations(Movie, file_path=os.path.join(default_files_path, "cleaned_content_based.csv"),
                              model_path=os.path.join("../Files", "models", "content_based",
                                                      "tfidf_description_matrix.pickle"))


def get_chihuahua(Image, model=model):
    Image = expand_dims(Image, axis=0)

    pred = model.predict(Image)
    print(pred)
    if pred > 0.5:
        return "Muffin", os.path.join("../Files", "PNG", "annoyed_doggy.jpeg")
    else:
        return "Chihuahua", os.path.join("../Files", "PNG", "happy_doggy.jpeg")


content_based = gr.Interface(
    fn=content_recommender,
    inputs=[gr.Dropdown(choices=titles)],
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


hybrid = gr.Interface(
    fn=improved_recommender,
    inputs=[gr.Dropdown(choices=titles)],
    outputs=[gr.Textbox(label="Recomendaciones")],
    title="Robert-DownIA-Jr - Tu Recomendador de Confianza",
    description="Indica un título y te recomendaré una película parecida con buenas valoraciones."
)


muffin_chihuahua = gr.Interface(
    fn=get_chihuahua,
    inputs=gr.Image(),
    outputs=[gr.Textbox(label="¿Muffin o Chihuahua?"), "image"],
    title="Robert-DownIA-Jr - Chihuahua... ¿O Muffin?",
    description="Sube una imagen para que nuestra maravillosa IA te ayude a distinguir si es un muffin sabroso o"
                "un chihuahua precioso."
)

app = gr.TabbedInterface([demographic, content_based, hybrid, muffin_chihuahua],
                         ["Género", "Sinopsis", "Híbrido", "Chihuahuas"] )

app.launch(debug=True)
