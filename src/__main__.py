import llm_sdk
import json
import argparse
from pydantic import BaseModel
from typing import Literal
import sys


class Parameter(BaseModel):
    type: Literal["string", "number"]


class ReturnType(BaseModel):
    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class AllFunctions(BaseModel):
    funcs: list[FunctionDefinition]


class State:
    def __init__(self) -> None:
        self.transitions: dict[int, list["State"]] = {}
        self.default_transitions: list[tuple["State", set[int]]] = []
        self.accept: bool = False
    
    def add_transition(self, byte_val: int, next_: "State") -> None:
        if byte_val not in self.transitions:
            self.transitions[byte_val] = []
        self.transitions[byte_val].append(next_)
    
    def add_default_transition(self, next_: "State", exclude_bytes: set[int] | None = None) -> None:
        exclude_set = set(exclude_bytes) if exclude_bytes else set()
        self.default_transitions.append((next_, exclude_set))


_CONSUME_CACHE: dict[tuple[frozenset["State"], int], frozenset["State"]] = {}


def consume_byte(states: frozenset["State"], byte: int) -> frozenset["State"]:
    key = (states, byte)
    if key in _CONSUME_CACHE:
        return _CONSUME_CACHE[key]
    
    next_ = set()
    for state in states:
        if byte in state.transitions:
            for nxt in state.transitions[byte]:
                next_.add(nxt)
        for nxt, excluded in state.default_transitions:
            if byte not in excluded:
                next_.add(nxt)
    result = frozenset(next_)
    _CONSUME_CACHE[key] = result
    return result

def consume_bytes(states: frozenset["State"], byte_seq: bytes) -> frozenset["State"]:
    current = states
    for byte in byte_seq:
        current = consume_byte(current, byte)
        if not current:
            break
    return current

def build_literal(state: "State", literal_bytes: bytes) -> "State":
    current = state
    for byte in literal_bytes:
        nxt = State()
        current.add_transition(byte, nxt)
        current = nxt
    return current

def build_number(state: "State", is_float: bool = False) -> "State":
    first_digit = State()
    state.add_transition(ord("-"), first_digit)
    digit_loop = State()
    for index in range(10):
        byte = ord("0") + index
        state.add_transition(byte, digit_loop)
        first_digit.add_transition(byte, digit_loop)
        digit_loop.add_transition(byte, digit_loop)
    if not is_float:
        return digit_loop
    dot = State()
    digit_loop.add_transition(ord("."), dot)
    float_loop = State()
    for index in range(10):
        byte = ord("0") + index
        dot.add_transition(byte, float_loop)
        float_loop.add_transition(byte, float_loop)
    return float_loop

def build_string(state: "State") -> "State":
    in_str = State()
    state.add_transition(ord('"'), in_str)
    end_state = State()
    in_str.add_transition(ord('"'), end_state)
    in_str.add_default_transition(in_str, {ord('"'), ord('\\')})
    escape_state = State()
    in_str.add_transition(ord("\\"), escape_state)
    valid_escapes = {ord('"'), ord("\\"), ord("/"), ord("b"), ord("f"), ord("n"), ord("r"), ord("t"), ord("u")}
    for escape in valid_escapes:
        escape_state.add_transition(escape, in_str)
    return end_state

class SchemaCompiler:
    def __init__(self, functions: AllFunctions):
        self.functions = functions.funcs

    def compile(self) -> State:
        start = State()
        prefix = build_literal(start, b'{"fn_name":"}')
        final_accept_states = []
        for fn in self.functions:
            fn_name = fn.name
            fn_state = build_literal(prefix, fn_name.encode("utf-8"))
            args_prefix = build_literal(fn_state, b'","args":{')
            current = [args_prefix]
            args_names = list(fn.parameters.keys())
            args_types = fn.parameters
            for index, arg_name in enumerate(args_names):
                if index > 0:
                    current = self._chain_literal(current, b",")
                arg_key = f'"{arg_name}":'.encode("utf-8")
                current = self._chain_literal(current, arg_key)
                arg_type = args_types.get(arg_name).type
                current = self._chain_type(current, arg_type)
            end_states = self._chain_literal(current, b"}}")
            final_accept_states.extend(end_states)
        for accept_state in final_accept_states:
            accept_state.accept = True
        return start
    
    def _chain_literal(self, states: list[State], literal_bytes: bytes) -> list[State]:
        new_states = []
        for s in states:
            new_states.append(build_literal(s, literal_bytes))
        return new_states
    
    def _chain_type(self, states: list[State], arg_type: str | None) -> list[State]:
        new_states = []
        for s in states:
            if arg_type == "number":
                new_states.append(build_number(s, is_float=True))
            elif arg_type == "string":
                new_states.append(build_string(s))
            else:
                raise ValueError(f"Type {arg_type} is not supported")
        return new_states


def get_vocab_strs(path: str) -> dict[int, bytes]:
    with open(path, "r") as vocab_file:
        vocab = json.load(vocab_file)
    raw_bytes = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    uni_chars = raw_bytes[:]
    num = 0
    for byte in range(2**8):
        if byte not in raw_bytes:
            raw_bytes.append(byte)
            uni_chars.append(byte + 256)
            num += 1
    unicode_to_byte = {chr(char): byte for char, byte in zip(uni_chars, raw_bytes)}
    token_to_byte = {}
    for token_str, token_id in vocab.items():
        token_bytes = bytes([unicode_to_byte.get(char, 0) for char in token_str])
        token_to_byte[token_id] = token_bytes
    return token_to_byte


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
    try:
        llm = llm_sdk.Small_LLM_Model(device="cpu")
        vocab_bytes = get_vocab_strs(llm.get_path_to_vocabulary_json())
    except Exception as e:
        print(e)
        sys.exit(1)
    # try:
    compiler = SchemaCompiler(functions)
    start = compiler.compile()
    # except Exception as e:
    #     print(e)
    #     sys.exit(1)
    


if __name__ == "__main__":
    main()
