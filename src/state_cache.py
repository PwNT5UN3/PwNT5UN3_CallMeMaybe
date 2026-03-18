class State:
    """Defines a single DFA state"""
    def __init__(self) -> None:
        """creates the state"""
        self.transitions: dict[int, list["State"]] = {}
        self.default_transitions: list[tuple["State", set[int]]] = []
        self.accept: bool = False

    def add_transition(self, byte_val: int, next_: "State") -> None:
        """Adds a valid follow-up state"""
        if byte_val not in self.transitions:
            self.transitions[byte_val] = []
        self.transitions[byte_val].append(next_)

    def add_default_transition(
            self, next_: "State", exclude_bytes: set[int] | None = None
            ) -> None:
        """
        Adds a collection of follow-up states for if no transition is given
        """
        exclude_set = set(exclude_bytes) if exclude_bytes else set()
        self.default_transitions.append((next_, exclude_set))


_CONSUME_CACHE: dict[tuple[frozenset["State"], int], frozenset["State"]] = {}


def consume_byte(states: frozenset["State"], byte: int) -> frozenset["State"]:
    """retrieves a state corresponding to a byte from the cache if it exists,
    otherwise it stores the state"""
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


def consume_bytes(
        states: frozenset["State"], byte_seq: bytes) -> frozenset["State"]:
    """automates consumption for multiple bytes"""
    current = states
    for byte in byte_seq:
        current = consume_byte(current, byte)
        if not current:
            break
    return current


def build_literal(state: "State", literal_bytes: bytes) -> "State":
    """Adds state chains for literals"""
    current = state
    for byte in literal_bytes:
        nxt = State()
        current.add_transition(byte, nxt)
        current = nxt
    return current


def build_number(state: "State") -> "State":
    """Adds state chains for numbers"""
    first_digit = State()
    state.add_transition(ord("-"), first_digit)
    digit_loop = State()
    for index in range(10):
        byte = ord("0") + index
        state.add_transition(byte, digit_loop)
        first_digit.add_transition(byte, digit_loop)
        digit_loop.add_transition(byte, digit_loop)
    dot = State()
    digit_loop.add_transition(ord("."), dot)
    float_loop = State()
    for index in range(10):
        byte = ord("0") + index
        dot.add_transition(byte, float_loop)
        float_loop.add_transition(byte, float_loop)
    return float_loop


def build_string(state: "State") -> "State":
    """Adds state chains for strings"""
    in_str = State()
    state.add_transition(ord('"'), in_str)
    end_state = State()
    in_str.add_transition(ord('"'), end_state)
    in_str.add_default_transition(in_str, {ord('"'), ord('\\')})
    escape_state = State()
    in_str.add_transition(ord("\\"), escape_state)
    valid_escapes = {
        ord('"'), ord("\\"), ord("/"), ord("b"), ord("f"),
        ord("n"), ord("r"), ord("t"), ord("u")}
    for escape in valid_escapes:
        escape_state.add_transition(escape, in_str)
    return end_state
