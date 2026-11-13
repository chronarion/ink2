"""Tool-call round trip through the OpenAI-compatible API (the server's tool parser): the model must call the tool
with valid JSON arguments, then answer from the tool result. Thinking on and off.
   python3 -I tools_check.py [URL]"""
import json, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111/v1"
model = json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}]


def chat(messages, thinking):
    body = dict(model=model, messages=messages, tools=TOOLS, max_tokens=2048, temperature=0,
                reasoning_effort="medium" if thinking else "none")
    req = urllib.request.Request(f"{URL}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]


ok = 0
for thinking in (False, True):
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    m = chat(msgs, thinking)
    calls = m.get("tool_calls") or []
    good = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    args = {}
    if good:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
            good = str(args.get("city", "")).lower().startswith("paris")
        except Exception:
            good = False
    print(f"thinking={thinking}: tool call {'OK' if good else 'FAIL'} {calls[:1]}")
    if not good:
        continue
    msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp_c": 17, "sky": "light rain"})}]
    m2 = chat(msgs, thinking)
    ans = (m2.get("content") or "")
    good2 = "17" in ans and not m2.get("tool_calls")
    print(f"thinking={thinking}: final answer {'OK' if good2 else 'FAIL'}: {ans[:160]!r}")
    ok += good and good2
print(f"TOOLS {ok}/2")
