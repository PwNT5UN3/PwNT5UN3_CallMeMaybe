from pydantic import BaseModel
from typing import Literal


class Parameter(BaseModel):
    """Possible argument types"""
    type: Literal["string", "number"]


class ReturnType(BaseModel):
    """Possible return types"""
    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    """Function definiton structure"""
    name: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class AllFunctions(BaseModel):
    """Wrapper for function defs"""
    funcs: list[FunctionDefinition]
