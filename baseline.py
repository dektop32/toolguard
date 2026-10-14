import ollama

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "add", "description": "Add two numbers",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "number"},
                                      "b": {"type": "number"}},
                       "required": ["a", "b"]}}},
]

# (prompt, expected tool name or None, expected arguments or None)
TASKS = [
    ("What's the weather in Bengaluru?", "get_weather", {"city": "bengaluru"}),
    ("Add 17 and 25", "add", {"a": 17, "b": 25}),
    ("Is it raining in Mysuru? Check please.", "get_weather", {"city": "mysuru"}),
    ("I have 3.5 rupees and found 4 more. Total?", "add", {"a": 3.5, "b": 4}),
    ("Weather for Hubballi-Dharwad", "get_weather", {"city": "hubballi-dharwad"}),
    ("sum of -8 and 12.25", "add", {"a": -8, "b": 12.25}),
    ("Tell me a joke", None, None),
    ("Add 5 and seven", "add", {"a": 5, "b": 7}),
]


def norm(args):
    return {k: v.lower() if isinstance(v, str) else v for k, v in args.items()}


def run(model):
    passed = 0
    for prompt, tool, expected in TASKS:
        try:
            r = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOLS,
            )
        except Exception as e:
            print(f"ERROR | {prompt} | {e}")
            continue

        calls = r.message.tool_calls or []
        if tool is None:
            ok = not calls
        else:
            ok = (
                bool(calls)
                and calls[0].function.name == tool
                and norm(dict(calls[0].function.arguments)) == expected
            )

        passed += ok
        print("PASS" if ok else "FAIL", "|", prompt)
        if not ok and calls:
            print("   got:", calls[0].function.name,
                  dict(calls[0].function.arguments))

    print(f"\n{model}: {passed}/{len(TASKS)}\n")

if __name__ == "__main__":
    for m in ["qwen2.5:3b"]:
        run(m)