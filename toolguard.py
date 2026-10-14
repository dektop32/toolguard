import ollama
from jsonschema import validate, ValidationError

NO_TOOL = {"type": "function", "function": {
	"name": "no_tool_needed",
	"description": "Use when the request needs none of the other tools "
				   "(chit-chat, jokes, general questions).",
	"parameters": {"type": "object", "properties": {}, "required": []}}}


def check_call(call, tools, prompt):
	"""Return an error message if the tool call is invalid, else None."""
	schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
	name = call.function.name
	if name not in schemas:
		return f"Unknown tool '{name}'. Valid tools: {list(schemas)}"
	args = dict(call.function.arguments)
	try:
		validate(args, schemas[name])
	except ValidationError as e:
		return f"Bad arguments for '{name}': {e.message}"
	for k, v in args.items():
		if isinstance(v, str) and v.lower() not in prompt.lower():
			return f"Argument {k}='{v}' does not appear in the user's request"
	return None


def guarded_chat(model, messages, tools, max_retries=2):
	"""Returns (validated tool call or None, number of retries used)."""
	prompt = messages[-1]["content"]
	all_tools = tools + [NO_TOOL]
	msgs = list(messages)
	call = None
	for attempt in range(max_retries + 1):
		r = ollama.chat(model=model, messages=msgs, tools=all_tools)
		calls = r.message.tool_calls or []
		if not calls:
			return None, attempt
		call = calls[0]
		if call.function.name == "no_tool_needed":
			return None, attempt
		error = check_call(call, tools, prompt)
		if error is None:
			return call, attempt
		msgs.append({"role": "assistant", "content": str(call.function)})
		msgs.append({"role": "user", "content":
					 f"Your tool call was invalid: {error}. Try again. "
					 "If no tool is needed, use no_tool_needed."})
	return call, attempt
