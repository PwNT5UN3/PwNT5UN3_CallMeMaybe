import llm_sdk
from llm_sdk import Small_LLM_Model
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from src.funcdefs import AllFunctions
from src.schema_constraints import SchemaCompiler, ConstraintMask
from src.state_cache import State


def get_vocab_strs(path: str) -> dict[int, bytes]:
    """creates a token cache of significant tokens
    mapping to ascii and latin-1 characters"""
    with open(path, "r") as vocab_file:
        vocab = json.load(vocab_file)
    raw_bytes = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    uni_chars = raw_bytes.copy()
    num = 0
    for byte in range(2**8):
        if byte not in raw_bytes:
            raw_bytes.append(byte)
            uni_chars.append(byte + 256)
            num += 1
    unicode_to_byte = {
        chr(char): byte for char, byte in zip(uni_chars, raw_bytes)}
    token_to_byte = {}
    for token_str, token_id in vocab.items():
        token_bytes = bytes(
            [unicode_to_byte.get(char, 0) for char in token_str])
        token_to_byte[token_id] = token_bytes
    return token_to_byte


def generate_constrained(
    llm: Small_LLM_Model, prompt_text: str, schema_start: State,
        vocab: dict[int, bytes], max_tokens: int = 100) -> str:
    """
    Generates a response to the prompt while making use of the constrining mask
    """
    input_ids = llm._encode(prompt_text).tolist()
    input_ids = [item for sublist in input_ids for item in sublist]
    mask = ConstraintMask(schema_start, vocab)
    generated_ids: list[int] = []
    for _ in range(max_tokens):
        if mask.finished():
            break
        current_input = input_ids + generated_ids
        logits = llm.get_logits_from_input_ids(current_input)
        next_token_logits = np.array(logits)
        valid_tokens = mask.get_valid_tokens()
        if not valid_tokens:
            raise Exception("Something went wrong, stopping generation")
        token_mask = np.full_like(next_token_logits, float("-inf"))
        token_mask[valid_tokens] = next_token_logits[valid_tokens]
        next_token_id = int(np.argmax(token_mask).item())
        mask.advance(next_token_id)
        generated_ids.append(next_token_id)
    result = str(llm._decode(generated_ids))
    return result


def main() -> None:
    """The main Wrapper
    handling parsing, generation looping and output file creation"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--functions_definition',
        default="data/input/functions_definition.json")
    parser.add_argument(
        '--input', default="data/input/function_calling_tests.json")
    parser.add_argument(
        '--output', default="data/output/function_calling_results.json")
    args = parser.parse_args()
    try:
        with open(args.functions_definition, "r") as defs:
            funcs_json = json.load(defs)
        functions = AllFunctions.model_validate({"funcs": funcs_json})
        with open(args.input, "r") as prompts_file:
            prompts = json.load(prompts_file)
            if not isinstance(prompts, list):
                raise ValueError("prompts must be passed as a list")
        llm = llm_sdk.Small_LLM_Model()
        vocab_bytes = get_vocab_strs(llm.get_path_to_vocabulary_json())
        compiler = SchemaCompiler(functions)
        start = compiler.compile()
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
            response = generate_constrained(
                llm, formatted_prompt, start, vocab_bytes, max_tokens=150)
            parsed_response = json.loads(response)
            final_obj = {"prompt": prompt_text, "name": parsed_response[
                "fn_name"], "args": parsed_response["args"]}
            results.append(final_obj)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as out_file:
            json.dump(results, out_file, indent=2)
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
