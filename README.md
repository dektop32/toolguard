# ToolGuard

A small reliability layer that makes tool-calling work better with small,
local open-weight models (Gemma 4, Qwen) running through Ollama.

## The problem
Small local models often call tools wrongly: made-up arguments, wrong tool
names, or a tool call when none is needed (for example, for "Tell me a joke").

## What ToolGuard does
1. Validates every tool call against its JSON schema (`jsonschema`).
2. Checks that text arguments really appear in the user's request.
3. Gives the model a `no_tool_needed` option, so it has a valid way to say
   "no tool applies".
4. If a call is invalid, retries up to 2 times, sending back a precise
   error message.

## Results
Same model, same 8 test tasks (weather, add, and a no-tool case):

| Model        | Raw  | With ToolGuard | Retries used |
|--------------|------|----------------|--------------|
| qwen2.5:3b   | 7/8  | 8/8            | FILL-IN      |

## Run it
Needs Python 3.12 and [Ollama](https://ollama.com).

    ollama pull qwen2.5:3b
    pip install ollama pydantic jsonschema pytest
    python baseline.py     # raw model score
    python compare.py      # raw vs ToolGuard

## Files
- `toolguard.py`: the validation and retry layer
- `baseline.py`: tool definitions and test tasks
- `compare.py`: before/after benchmark

## Models and dependencies
Open-weight models via Ollama (qwen2.5:3b, Gemma 4), `ollama` Python
client, `jsonschema`.

## License
MIT