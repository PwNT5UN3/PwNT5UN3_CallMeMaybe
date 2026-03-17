from pydantic import BaseModel
from typing import Literal


class Parameter(BaseModel):
    type: Literal["string", "number"]


class ReturnType(BaseModel):
    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    name: str
    # description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class AllFunctions(BaseModel):
    funcs: list[FunctionDefinition]
