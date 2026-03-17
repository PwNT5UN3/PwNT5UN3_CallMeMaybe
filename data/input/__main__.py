import llm_sdk
from llm_sdk import Small_LLM_Model
import json
import argparse
from pydantic import BaseModel
from typing import Literal
import sys
import numpy as np
import torch
from pathlib import Path

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


class ConstraintMask:
    def __init__(self, start: State, vocab: dict[int, bytes]) -> None:
        self.current: frozenset[State] = frozenset({start})
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.tokens_bytes: list[bytes] = [vocab.get(index, b"") for index in range(self.vocab_size)]
        self._valid_cache: dict[frozenset[State], list[int]] = {}
    
    def get_valid_tokens(self) -> list[int]:
        if self.current in self._valid_cache:
            return self._valid_cache[self.current]
        valid_ids = []
        for token_id, token_bytes in enumerate(self.tokens_bytes):
            if not token_bytes:
                continue
            next_ = consume_bytes(self.current, token_bytes)
            if next_:
                valid_ids.append(token_id)
        self._valid_cache[self.current] = valid_ids
        return valid_ids
    
    def advance(self, token_id: int) -> None:
        token_bytes = self.vocab.get(token_id, b"")
        if token_bytes:
            self.current = consume_bytes(self.current, token_bytes)
    
    def finished(self) -> bool:
        return any(state.accept for state in self.current)

def generate_constrained(llm: Small_LLM_Model, prompt_text: str, schema_start: State, vocab: dict[int, bytes], max_tokens: int = 100) -> str:
    input_ids = llm._encode(prompt_text).tolist()
    input_ids = [item for sublist in input_ids for item in sublist]
    mask = ConstraintMask(schema_start, vocab)
    generated_ids: list[int] = []
    for _ in range(max_tokens):
        if mask.finished():
            break
        current_input = input_ids + generated_ids
        logits = llm.get_logits_from_input_ids(current_input)
        next_token_logits = torch.tensor(logits)
        valid_tokens = mask.get_valid_tokens()
        if not valid_tokens:
            raise Exception("Something went wrong, stopping generation")
        token_mask = torch.full_like(next_token_logits, float("-inf"))
        token_mask[valid_tokens] = next_token_logits[valid_tokens]
        next_token_id = int(torch.argmax(token_mask).item())
        mask.advance(next_token_id)
        generated_ids.append(next_token_id)
    result = str(llm._decode(generated_ids))
    return result


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
    try:
        compiler = SchemaCompiler(functions)
        start = compiler.compile()
    except Exception as e:
        print(e)
        sys.exit(1)
    results = []
    for index, item in enumerate(prompts):
        if not isinstance(item, dict) or "prompt" not in item:
            print(f"Prompt '{item}' malformed, skipping...")
            continue
        prompt_text = item["prompt"]
        funcs_json_str = json.dumps(funcs_json, indent=2)
        formatted_prompt = (
            "<|im_start|>system\nYou are a reliable function-calling "
            "assistant. "
            f"You have access to the following functions:\n"
            f"{funcs_json_str}\n\n"
            "You must output ONLY a valid JSON object representing a "
            "function call. Start exactly with {\"fn_name\":...\n"
            "CRITICAL: If you need to write a regular expression,"
            " DO NOT use JavaScript "
            "regex delimiters like /.../g."
            " Just output the raw Python pattern. "
            "If your pattern requires backslashes"
            " (like \\w or \\b), you MUST double-escape "
            "them (e.g. \\\\w) because this is a JSON string.<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        try:
            response = generate_constrained(llm, formatted_prompt, start, vocab_bytes, max_tokens=150)
            parsed_json = json.loads(response)
            print(prompt_text)
            print(parsed_json)
            print()
            final_obj = {"prompt": prompt_text, "fn_name": parsed_json["fn_name"], "args": parsed_json["args"]}
            results.append(final_obj)
        except Exception as e:
            print(e)
            sys.exit(1)
    try:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as out_file:
            json.dump(results, out_file, indent=2)
    except Exception as e:
        print(e)
        sys.exit(1)
        
        

if __name__ == "__main__":
    main()
