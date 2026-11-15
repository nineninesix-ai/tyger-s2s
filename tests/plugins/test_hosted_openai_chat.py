"""The `openai-chat` LLM plugin without a network (no GPU, no key, no cost): the history conversion, the request it
sends (max_completion_tokens, never max_tokens; reasoning_effort, extra_body), the stream parser on synthetic chunk
sequences (text, reasoning dropped, tool calls in pieces, a call cut off, refusals, usage counts), and the plugin
driven by a stub client (stop closes the stream, errors become LLMError, warmup's free request).
Run: .venv/bin/python tests/plugins/test_hosted_openai_chat.py"""
import asyncio, os, sys, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.plugins.base import LLMError, PluginContext  # noqa: E402
from app.plugins.llm.hosted.openai_chat import ChatStreamParser, OpenAIChatLLM, to_chat_messages  # noqa: E402
from app.runtime.event_loop import EngineLoop  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_current_time", "description": "Get the current time in a timezone or city.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA tz"}},
                   "required": ["timezone"]}}}]


def chunk(finish=None, usage=None, **delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage}


def piece(index, name=None, args=""):
    fn = {"arguments": args} if name is None else {"name": name, "arguments": args}
    return chunk(tool_calls=[{"index": index, "function": fn}])


def usage(prompt, cached, completion):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                     "prompt_tokens_details": {"cached_tokens": cached}}}


STOP = chunk(finish="stop")


def parse(chunks, tools=TOOLS):
    p = ChatStreamParser(tools); out = []
    for c in chunks:
        out += p.feed(c)
    return out + p.finish()


# ---- conversion -------------------------------------------------------------------------------------------
def test_history_to_chat_messages():
    neutral = [{"role": "system", "content": "sys"}, {"role": "user", "content": "time in Oslo and Rome?"},
               {"role": "assistant", "content": "", "native": {"x": 1},
                "tool_calls": [{"id": "call_a", "name": "get_current_time", "arguments": {"timezone": "Europe/Oslo"}},
                               {"id": "call_b", "name": "get_current_time", "arguments": {"timezone": "Europe/Rome"}}]},
               {"role": "tool", "tool_call_id": "call_a", "content": "{}"},
               {"role": "tool", "tool_call_id": "call_b", "content": "{}"},
               {"role": "assistant", "content": "Ten past nine in both. —"}]
    out = to_chat_messages(neutral)
    assert out[:2] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "time in Oslo and Rome?"}]
    assert out[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_a", "type": "function",
         "function": {"name": "get_current_time", "arguments": '{"timezone": "Europe/Oslo"}'}},
        {"id": "call_b", "type": "function",
         "function": {"name": "get_current_time", "arguments": '{"timezone": "Europe/Rome"}'}}]}
    assert out[3] == {"role": "tool", "tool_call_id": "call_a", "content": "{}"} and out[4]["tool_call_id"] == "call_b"
    assert out[5] == {"role": "assistant", "content": "Ten past nine in both. —"}
    assert all("native" not in m for m in out)


# ---- the stream parser --------------------------------------------------------------------------------------
def test_text_comes_through_and_reasoning_is_dropped():
    evs = parse([chunk(role="assistant"), chunk(reasoning="The user wants"), chunk(reasoning_content=" a joke."),
                 chunk(content="Sure, "), chunk(reasoning="hmm", content="here goes."), STOP])
    assert evs == [("text", "Sure, "), ("text", "here goes.")]


def test_tool_call_in_pieces():
    evs = parse([chunk(content="One moment. "), piece(0, "get_current_time"), piece(0, args='{"time'),
                 piece(0, args='zone": "Asia/'), piece(0, args='Tokyo"}'), chunk(finish="tool_calls")])
    assert evs[:2] == [("text", "One moment. "), ("tool_start", {"name": "get_current_time"})]
    kind, calls, native = evs[2]
    assert kind == "tool_calls" and native is None and len(calls) == 1 and len(evs) == 3
    assert calls[0]["name"] == "get_current_time" and calls[0]["arguments"] == {"timezone": "Asia/Tokyo"}
    assert calls[0]["id"].startswith("call_") and "error" not in calls[0]


def test_two_calls_by_index_and_text_after_a_call_not_spoken():
    evs = parse([piece(0, "get_current_time", '{"timezone": "Europe/Oslo"}'), chunk(content="leaked"),
                 piece(1, "get_current_time", '{"timezone": '), piece(1, args='"Europe/Rome"}'),
                 chunk(finish="tool_calls")])
    assert [e[0] for e in evs] == ["tool_start", "tool_calls"]
    calls = evs[1][1]
    assert [c["arguments"]["timezone"] for c in calls] == ["Europe/Oslo", "Europe/Rome"]
    assert calls[0]["id"] != calls[1]["id"]


def test_invalid_arguments_are_flagged_never_run():
    for args in ('{"tz": "Europe/Oslo"}', '{"timezone": 5}', '{"timezone": "Eu', '[1, 2]'):
        evs = parse([piece(0, "get_current_time", args), chunk(finish="tool_calls")])
        call = evs[1][1][0]
        assert "error" in call and isinstance(call["arguments"], dict), (args, call)
    evs = parse([piece(0, "no_such_tool", "{}"), chunk(finish="tool_calls")])
    assert "unknown tool" in evs[1][1][0]["error"]
    evs = parse([piece(0, "get_current_time", ""), chunk(finish="tool_calls")])
    assert "missing timezone" in evs[1][1][0]["error"]          # no arguments: {} -> its required key is missing


def test_a_call_cut_off_at_the_limit_is_not_run():
    evs = parse([piece(0, "get_current_time", '{"timezone": "Asia/To'), chunk(finish="length")])
    assert evs == [("tool_start", {"name": "get_current_time"}), ("tool_calls", [], None)]
    evs = parse([chunk(content="A long answer that"), chunk(finish="length")])
    assert evs == [("text", "A long answer that")]               # cut-off text is still spoken


def test_refusal():
    assert parse([chunk(refusal="I can't help with that."), STOP]) == [("refusal", "refusal")]
    assert parse([chunk(content="Well"), chunk(finish="content_filter")]) == [("text", "Well"), ("refusal", "refusal")]
    evs = parse([piece(0, "get_current_time", '{"timezone": "UTC"}'), chunk(finish="content_filter")])
    assert evs[1][0] == "tool_calls"                             # a call that came whole is a call


def test_usage_counts():
    p = ChatStreamParser(TOOLS)
    for c in [chunk(content="Hi."), STOP, usage(3500, 3072, 9)]:
        p.feed(c)
    assert p.usage == {"prompt": 3500, "cached": 3072, "completion": 9}
    p = ChatStreamParser(TOOLS)
    p.feed({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 3}})   # no cache details
    assert p.usage == {"prompt": 40, "cached": 0, "completion": 3}


# ---- the plugin on a stub client ----------------------------------------------------------------------------
class StubStream:
    def __init__(self, chunks, delay=0.0, fail=None):
        self.chunks, self.delay, self.fail, self.closed = list(chunks), delay, fail, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            if self.fail is not None:
                raise self.fail
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        return self.chunks.pop(0)

    async def close(self):
        self.closed = True


def make_plugin(**conf):
    os.environ["TYGER_TEST_KEY"] = "test-not-a-key"
    return OpenAIChatLLM({"model": "stub", "base_url": "http://127.0.0.1:9", "api_key_env": "TYGER_TEST_KEY",
                          **conf}, PluginContext(EngineLoop(), {}, "stub"))


def stub_create(p, make_stream, record=None):
    async def create(**kw):
        if record is not None:
            record.append(kw)
        return make_stream()
    p.client.chat.completions.create = create


def test_request_keywords():
    p = make_plugin(max_tokens=24, temperature=0.7, reasoning_effort="none",
                    extra_body={"include_reasoning": False})
    kw = p.request([{"role": "user", "content": "hi"}], tools=TOOLS, max_new=192)
    assert kw["model"] == "stub" and kw["stream"] is True and kw["stream_options"] == {"include_usage": True}
    assert kw["max_completion_tokens"] == 24 and "max_tokens" not in kw
    assert kw["reasoning_effort"] == "none" and kw["temperature"] == 0.7 and kw["tools"] == TOOLS
    assert kw["extra_body"] == {"include_reasoning": False}
    kw = make_plugin().request([{"role": "user", "content": "hi"}], max_new=12)
    assert kw["max_completion_tokens"] == 12 and "max_tokens" not in kw
    for absent in ("tools", "temperature", "reasoning_effort", "extra_body"):
        assert absent not in kw, absent


def test_plugin_streams_and_counts_usage():
    p = make_plugin(); sent = []
    stub_create(p, lambda: StubStream([chunk(reasoning="thinking"), chunk(content="Sunny."), STOP,
                                       usage(100, 64, 2)]), sent)
    evs = list(p.stream([{"role": "user", "content": "weather?"}], max_new=12))
    assert evs == [("text", "Sunny.")] and sent[0]["max_completion_tokens"] == 12
    assert p.usage == {"requests": 1, "prompt": 100, "cached": 64, "completion": 2}


def test_plugin_stop_closes_stream():
    p = make_plugin(max_concurrency=1, queue_timeout_s=0.5); streams = []

    def mk():
        streams.append(StubStream([chunk(content=f"word{i} ") for i in range(200)], delay=0.01))
        return streams[-1]
    stub_create(p, mk)
    got = []
    for ev in p.stream([{"role": "user", "content": "hi"}], stop_flag=lambda: len(got) >= 3):
        got.append(ev)
    time.sleep(0.1)                                  # the close runs on the engine loop
    assert len(got) <= 4 and streams[0].closed and len(streams[0].chunks) > 150
    again = []
    for ev in p.stream([{"role": "user", "content": "again"}], stop_flag=lambda: bool(again)):
        again.append(ev)
    assert again and len(streams) == 2               # the stopped request gave its one slot back


def raises_llmerror(p, kind):
    try:
        list(p.stream([{"role": "user", "content": "hi"}]))
    except LLMError as e:
        assert e.kind == kind, (e.kind, kind)
        return
    raise AssertionError(f"expected LLMError({kind})")


def test_errors_become_llmerror():
    import httpx, openai
    p = make_plugin()
    req = httpx.Request("POST", "http://x")
    for status, kind in ((500, "server"), (400, "request"), (429, "rate_limit")):
        cls = openai.RateLimitError if status == 429 else openai.APIStatusError

        async def create(cls=cls, status=status, **kw):
            raise cls("boom", response=httpx.Response(status, request=req), body=None)
        p.client.chat.completions.create = create
        raises_llmerror(p, kind)

    async def create(**kw):
        raise openai.APIConnectionError(request=req)
    p.client.chat.completions.create = create
    raises_llmerror(p, "connection")
    # once the stream has started: an error event, a dropped connection, a read timeout
    for fail, kind in ((openai.APIError("overloaded", req, body=None), "server"),
                       (httpx.RemoteProtocolError("peer closed"), "connection"),
                       (httpx.ReadTimeout("slow"), "timeout")):
        stub_create(p, lambda fail=fail: StubStream([chunk(content="Well, ")], fail=fail))
        raises_llmerror(p, kind)


def test_warmup_lists_models_and_a_rejected_key_stops_startup():
    import httpx, openai
    p = make_plugin(); listed = []

    async def ok():
        listed.append(1)
    p.client.models.list = ok
    p.warmup()
    assert listed == [1] and p.last_used > 0

    async def rejected():
        raise openai.AuthenticationError("no", response=httpx.Response(401, request=httpx.Request("GET", "http://x")),
                                         body=None)
    p.client.models.list = rejected
    try:
        p.warmup()
        raise AssertionError("a rejected key must stop start-up")
    except RuntimeError as e:
        assert "TYGER_TEST_KEY" in str(e)

    async def down():
        raise openai.APIConnectionError(request=httpx.Request("GET", "http://x"))
    p.client.models.list = down
    p.warmup()                                       # only logged: the first turn opens the connection


def test_pings_only_while_in_use():
    p = make_plugin(ping_s=0.05, ping_window_s=0.4); pings = []

    async def ping():
        pings.append(time.monotonic())
    p.client.models.list = ping
    time.sleep(0.2)
    assert pings == []                               # never used: no pings
    stub_create(p, lambda: StubStream([chunk(content="ok"), STOP]))
    list(p.stream([{"role": "user", "content": "hi"}]))
    time.sleep(0.3)
    assert len(pings) >= 2
    time.sleep(0.4)
    n = len(pings); time.sleep(0.2)
    assert len(pings) == n                           # the window has passed: no more
    p.shutdown()


def test_idle_connections_kept_open():
    pool = make_plugin().client._client._transport._pool         # httpx's own pool (its default expiry: 5 s)
    assert pool._keepalive_expiry == 120.0 and pool._max_keepalive_connections == 8
    assert make_plugin(keepalive_s=30).client._client._transport._pool._keepalive_expiry == 30.0


def test_registered():
    from app.plugins import plugin_class
    assert plugin_class("llm", "openai-chat") is OpenAIChatLLM and make_plugin().remote


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
