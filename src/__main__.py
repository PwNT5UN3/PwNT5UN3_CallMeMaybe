# import llm_sdk
import json
import pydantic
import argparse
from pydantic import BaseModel
from typing import Literal
import sys


class Parameter(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class ReturnType(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class AllFunctions(BaseModel):
    funcs: list[FunctionDefinition]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--functions_definition', default="data/input/functions_definition.json")
    parser.add_argument('--input', default="data/input/function_calling_tests.json")
    parser.add_argument('--output', default="data/output/function_calling_results.json")
    args = parser.parse_args()
    try:
        with open(args.functions_definition, "r") as defs:
            funcs_json = json.load(defs)
        functions = AllFunctions.model_validate({"funcs": funcs_json})
    except Exception as e:
        print(e)
        sys.exit(1)
    try:
        with open(args.input, "r") as prompts_file:
            prompts = json.load(prompts_file)
            print(prompts)
            if not isinstance(prompts, list):
                raise ValueError("prompts must be passed as a list")
    except Exception as e:
        print(e)
        sys.exit(1)
    


if __name__ == "__main__":
    main()
