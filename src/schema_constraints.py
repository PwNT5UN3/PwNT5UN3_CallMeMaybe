from src.funcdefs import AllFunctions
from src.state_cache import State, build_literal, build_number, build_string
from src.state_cache import consume_bytes


class SchemaCompiler:
    def __init__(self, functions: AllFunctions):
        self.functions = functions.funcs

    def compile(self) -> State:
        start = State()
        prefix = build_literal(start, b'{"fn_name":"')
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
                arg_type_raw = args_types.get(arg_name)
                if arg_type_raw is None:
                    raise ValueError("something unexpected happened")
                arg_type = arg_type_raw.type
                current = self._chain_type(current, arg_type)
            end_states = self._chain_literal(current, b"}}")
            final_accept_states.extend(end_states)
        for accept_state in final_accept_states:
            accept_state.accept = True
        return start

    def _chain_literal(
            self, states: list[State], literal_bytes: bytes) -> list[State]:
        new_states = []
        for s in states:
            new_states.append(build_literal(s, literal_bytes))
        return new_states

    def _chain_type(
            self, states: list[State], arg_type: str | None) -> list[State]:
        new_states = []
        for s in states:
            if arg_type == "number":
                new_states.append(build_number(s))
            elif arg_type == "string":
                new_states.append(build_string(s))
            else:
                raise ValueError(f"Type {arg_type} is not supported")
        return new_states


class ConstraintMask:
    def __init__(self, start: State, vocab: dict[int, bytes]) -> None:
        self.current: frozenset[State] = frozenset({start})
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.tokens_bytes: list[bytes] = [
            vocab.get(index, b"") for index in range(self.vocab_size)]
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
