import ollama
from baseline import TOOLS, TASKS, norm
from toolguard import guarded_chat


def is_ok(call, tool, expected):
	if tool is None:
		return call is None
	return (call is not None and call.function.name == tool
			and norm(dict(call.function.arguments)) == expected)


def raw_call(model, prompt):
	r = ollama.chat(model=model,
					messages=[{"role": "user", "content": prompt}],
					tools=TOOLS)
	calls = r.message.tool_calls or []
	return calls[0] if calls else None


def main(models):
	n = len(TASKS)
	print(f"{'model':<16}{'raw':>8}{'guarded':>10}{'retries':>9}")
	for m in models:
		raw = grd = retries = 0
		for prompt, tool, expected in TASKS:
			raw += is_ok(raw_call(m, prompt), tool, expected)
			call, used = guarded_chat(
				m, [{"role": "user", "content": prompt}], TOOLS)
			grd += is_ok(call, tool, expected)
			retries += used
		print(f"{m:<16}{raw:>5}/{n}{grd:>7}/{n}{retries:>9}")


if __name__ == "__main__":
	main(["qwen2.5:3b"])
