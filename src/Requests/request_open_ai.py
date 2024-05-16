import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from src.Requests.request_imdb import get_film_info

load_dotenv()


def open_ai_api_request_movie_dict(imdb_id):
    try:
        film_dict = get_film_info(imdb_id)
        movie_name = film_dict["Title"]

        system_template = SystemMessagePromptTemplate.from_template(
            "You are an avid film watcher with extensive knowledge in filmography. The user will input a film name and "
            "you have to return a summary of 100 words, the name of the director, the name of the main actors and the "
            "overall general opinion of the film by the public in under 25 words. "
            "You must return that using a json format with the aformentioned fields (summary, director, main actors, "
            "overall opinion).")
        user_template = HumanMessagePromptTemplate.from_template(movie_name)
        template = ChatPromptTemplate.from_messages([system_template, user_template])

        model = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
        chain = (template
                 | model
                 | JsonOutputParser()
                 )

        return chain.invoke({})
    except Exception as e:
        print(f"Could not retrieve id = {imdb_id}. Cause: {e}.")


print(open_ai_api_request_movie_dict("0114709"))
