from pydantic import BaseModel, Field, field_validator


class RecommenderRequest(BaseModel):
    description: str = Field(description="General description of what you want to watch.", default="")
    genres: list = Field(description="Genre", default=[])

    def __are_strings(*args):
        are_strings = True
        for arg in args:
            are_strings = are_strings and isinstance(arg, str)
        return are_strings

    @field_validator('genres')
    def must_be_string_list(cls, lst):
        if len(lst) == 0 or cls.__are_strings(lst):
            raise ValueError('The list must only contain strings.')
        return lst
