*This project has been created as part of the 42 curriculum by mawelsch*

## Description
This project focuses on utilizing **constrained decoding** to enable a small **LLM (Qwen3-0.6)** to generate responses formatted as **JSON** objects. This is intended for seamless integration with a function-calling system.

## Instructions
To execute the program, you can use either of the following commands:

- `python3 -m src`
- `make run`

### Optional Arguments
- `--functions_definition path/to/functions_definition.json` (defaults to `./data/input`)
- `--input path/to/function_calling_tests.json` (defaults to `./data/input`)
- `--output path/to/function_calling_results.json` (defaults to `./data/output`)

## Algorithm Explanation
The program employs a **state system** where each state corresponds to a token, and subsequent states are constrained based on the current tokenized state. 

- **Predefined States**: Key states such as the beginning and end are predefined to maintain structural cohesion. The LLM fills in the gaps within this structure, ensuring the output remains valid JSON regardless of the LLM's token choices.
  
- **State Functionality**: This state system acts as a mask, significantly narrowing down available tokens during the logit selection process. The use of a specialized prompt guides the LLM towards producing the desired output structure and valid function calls.

## Design Choices
Two crucial design decisions were made during the development of this project:

1. **State System**: Implementing a state system was an elegant approach to ensure the LLM adhered to the desired structure. Combined with a **state consumption cache**, this provided enhanced performance and reliability.

2. **Specialized Vocabulary Dictionary**: A dictionary mapping all printable ASCII codes and portions of the Latin-1 code set to their token representations was created. This optimization reduces the number of calls necessary to the tokenizer's decoding function, speeding up the process.

## Challenges Faced
The journey to implement constrained decoding was filled with challenges:

- I had no prior knowledge of constrained decoding, requiring extensive reading, much of which was less than helpful. Fortunately, a peer who had previously completed the project was invaluable in guiding my understanding.

- I encountered difficulties with the provided PCs due to heavy dependencies, which led me to primarily work on my personal laptop. This allowed me to utilize **NVIDIA CUDA** for GPU acceleration, significantly streamlining the debugging process.

## Testing Strategy
Testing was approached systematically:

1. A set of hypothetical functions and corresponding prompts were developed.
2. Successful function calls were validated, after which additional sets were constructed.
3. If issues arose in function output, portions of the codebase were refined based on assumptions regarding the fault until the desired outputs were achieved.

## Example Usage
To effectively utilize the program, two essential components are required: function definitions and corresponding prompts.

### Example Function
```json
{
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
        "a": {
            "type": "number"
        },
        "b": {
            "type": "number"
        }
    },
    "returns": {
        "type": "number"
    }
}
```

### Example Prompt
**Prompt**: What is the sum of 2 and 3?

### Expected Output
```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "args": {
        "a": 2.0,
        "b": 3.0
    }
}
```

## Resources
- **Ayoub Ben Cadi**: [GitHub](https://github.com/ExceptedPrism3) - Instrumental in providing support and a foundational codebase that facilitated my progress with the project.
- **HuggingFace**: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) - Valuable resource for understanding the LLM utilized.
- **GeeksForGeeks**: [Introduction to Finite Automata](https://www.geeksforgeeks.org/theory-of-computation/introduction-of-finite-automata/) - Essential in grasping the concept of finite automata used within the state system.

### AI Utilization
Throughout the project, AI facilitated several tasks including:
- Topic clarification
- Code clarification
- Design brainstorming
- Documentation creation