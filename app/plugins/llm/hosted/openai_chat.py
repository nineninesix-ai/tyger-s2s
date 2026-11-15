"""openai-chat: any OpenAI-compatible Chat Completions endpoint, for the `reply` and `filler` roles on a hosted
provider (Cerebras, Groq, ...). Nothing here names a provider: `base_url`, `model`, `api_key_env` and `extra_body`
in the config say which one.

HOSTED: transcripts (and, in a tool turn, tool results) go to the endpoint in `base_url`. That is allowed only
because the config names this plugin explicitly; the startup log says which roles are remote. Request and response
bodies are never logged; keep OPENAI_LOG unset (the SDK's debug logging prints bodies).

  * Client: AsyncOpenAI on MOUTH's EngineLoop, one per instance; chat.completions.create(stream=True). An idle
    connection stays open `keepalive_s` (httpx's default is 5 s: most turns would pay a new TLS handshake), and
    warmup() opens one with a free request (GET /models). With `ping_s` set, an idle connection gets another free
    request every `ping_s`, while the instance had a real request in the last `ping_window_s`.
  * Streaming: delta.content -> text. delta.reasoning (and reasoning_content) is never read: a reasoning model's
    thinking is never spoken or logged. The first delta.tool_calls piece -> tool_start, with the name (it comes in
    that piece); the pieces' arguments add up per index and are parsed at the end -> tool_calls, checked with
    calls.flag_invalid, IDs from calls.new_call_id. Text after the first call piece isn't spoken. A response cut
    off at the token limit (finish_reason "length") never runs a call cut off with it. delta.refusal, or
    finish_reason "content_filter", -> ("refusal", "refusal").
  * Request: max_completion_tokens, never max_tokens (one provider takes max_tokens as an alias: never send both);
    temperature and reasoning_effort only if set; extra_body passed through, for a provider's own parameters. The
    usage chunk is asked for (stream_options.include_usage) and kept as counts only: prompt, cached, completion.
  * History: neutral messages -> chat messages: an assistant turn's tool_calls with JSON-string arguments, a tool
    result with its tool_call_id; `native` dropped (this plugin writes none).
  * Key, timeouts, concurrency, errors -> LLMError, shutdown: HostedLLM (hosted/base.py). Here also: an error event
    inside the stream, and a connection dropped or timed out mid-stream, are LLMError.
  * Abort: when stop_flag turns true the stream is closed (tokens already generated are still billed).
"""
from __future__ import annotations
import asyncio, json, logging, time
from dataclasses import dataclass

from app.plugins.base import LLMError
from app.plugins.llm.hosted.base import HostedLLM
from app.plugins.llm.calls import flag_invalid, new_call_id

log = logging.getLogger("llm")


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def to_chat_messages(messages):
    """Neutral history -> chat-completions messages: tool calls in the nested shape with JSON-string arguments, a
    tool result with its tool_call_id; `native` dropped."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"],
                                                     "arguments": json.dumps(c.get("arguments", {}),
                                                                             ensure_ascii=False)}}
                                       for c in m["tool_calls"]]})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


class ChatStreamParser:
    """Chat-completions stream chunks -> neutral LLM events. Pure (no I/O), so it is tested on synthetic chunk
    sequences. feed(chunk) and finish() return lists of events; `usage` holds the usage chunk's counts."""
    def __init__(self, tools=None):
        self.tools = tools
        self.started = False; self.calls = {}           # index -> {"name", "args"}
        self.finish_reason = None; self.refused = False; self.usage = None

    def feed(self, chunk):
        out = []
        usage = _get(chunk, "usage")
        if usage:
            details = _get(usage, "prompt_tokens_details")
            self.usage = {"prompt": _get(usage, "prompt_tokens") or 0,
                          "cached": (_get(details, "cached_tokens") if details else 0) or 0,
                          "completion": _get(usage, "completion_tokens") or 0}
        for choice in _get(chunk, "choices") or []:
            delta = _get(choice, "delta") or {}
            if _get(choice, "finish_reason"):
                self.finish_reason = _get(choice, "finish_reason")
            if _get(delta, "refusal"):
                self.refused = True
            for tc in _get(delta, "tool_calls") or []:
                fn = _get(tc, "function") or {}
                acc = self.calls.setdefault(_get(tc, "index") or 0, {"name": None, "args": ""})
                acc["name"] = acc["name"] or _get(fn, "name")
                acc["args"] += _get(fn, "arguments") or ""
                if not self.started:
                    self.started = True; out.append(("tool_start", {"name": acc["name"]}))
            content = _get(delta, "content")
            if content and not self.started:
                out.append(("text", content))
        return out

    def finish(self):
        if (self.refused or self.finish_reason == "content_filter") and not self.calls:
            return [("refusal", "refusal")]
        if not self.calls:
            return []
        if self.finish_reason == "length":              # cut off mid-call: never run a truncated call
            log.warning("tool call cut off at max_completion_tokens; not run")
            return [("tool_calls", [], None)]
        calls = []
        for i in sorted(self.calls):
            acc = self.calls[i]
            try:
                args = json.loads(acc["args"]) if acc["args"].strip() else {}
            except ValueError:
                args = None
            calls.append(flag_invalid({"id": new_call_id(), "name": acc["name"] or "", "arguments": args},
                                      self.tools))
        return [("tool_calls", calls, None)]


class OpenAIChatLLM(HostedLLM):
    SDK = "openai"

    @dataclass(frozen=True, kw_only=True)
    class Options(HostedLLM.Options):
        temperature: float | None = None       # sent only if set
        reasoning_effort: str | None = None    # sent only if set ("none": the model doesn't reason)
        max_tokens: int | None = None          # if set, every request gets it instead of MOUTH's max_new
        extra_body: dict | None = None         # passed through: a provider's own parameters
        keepalive_s: float = 120.0             # an idle connection stays open this long (httpx's default: 5 s)
        ping_s: float | None = None            # a free request (GET /models) when idle this long; None: never
        ping_window_s: float = 300.0           # ...only while there was a real request in this window

    def __init__(self, conf: dict, ctx):
        super().__init__(conf, ctx)
        self.max_tokens = self.opts.max_tokens
        self.last_used = 0.0                   # monotonic: the end of the last request (pings too)
        self.last_request = 0.0                # monotonic: the start of the last real request
        self.usage = {"requests": 0, "prompt": 0, "cached": 0, "completion": 0}   # counts only
        self.pinger = self.eloop.run(self._start_pings()) if self.opts.ping_s else None

    def make_client(self, key):
        import httpx
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient
        o = self.opts
        timeout = httpx.Timeout(o.read_timeout_s, connect=o.connect_timeout_s)
        limits = httpx.Limits(max_connections=o.max_concurrency + 2, max_keepalive_connections=o.max_concurrency,
                              keepalive_expiry=o.keepalive_s)
        return AsyncOpenAI(api_key=key, base_url=o.base_url, max_retries=o.max_retries, timeout=timeout,
                           http_client=DefaultAsyncHttpxClient(timeout=timeout, limits=limits))

    # ---- the warm connection ----
    async def _ping(self):
        """A free request (GET /models): opens a connection, or keeps one open."""
        await self.client.models.list()
        self.last_used = time.monotonic()

    def warmup(self):
        """Open the connection before the first turn. A rejected key stops start-up; anything else is logged and
        left to the first turn."""
        t0 = time.monotonic()
        try:
            self.eloop.run(self._ping(), timeout=self.opts.connect_timeout_s + self.opts.read_timeout_s + 1)
            log.info("llm %s: connection open (GET /models %.0f ms)", self.name, (time.monotonic() - t0) * 1e3)
        except Exception as e:
            err = self.llm_error(e)
            if err is not None and err.kind == "request":
                raise RuntimeError(f"llm {self.name}: the provider rejected the request (check "
                                   f"{self.opts.api_key_env} and base_url)") from None
            log.warning("llm %s: warmup failed (%s); the first turn opens the connection", self.name,
                        err.kind if err else type(e).__name__)

    async def _start_pings(self):
        return asyncio.ensure_future(self._pings())

    async def _pings(self):
        o = self.opts
        while True:
            await asyncio.sleep(min(o.ping_s, 5.0))
            now = time.monotonic()
            if now - self.last_request < o.ping_window_s and now - self.last_used >= o.ping_s:
                try:
                    await self._ping()
                except Exception as e:                   # not a turn's: the next turn reports its own
                    log.debug("llm %s: ping failed (%s)", self.name, type(e).__name__)
                    self.last_used = now

    # ---- requests ----
    def request(self, messages, tools=None, max_new=192) -> dict:
        """The chat.completions.create keyword arguments for one request (no I/O: tested as is)."""
        o = self.opts
        kw = dict(model=self.model, messages=to_chat_messages(messages), stream=True,
                  stream_options={"include_usage": True}, max_completion_tokens=int(self.max_tokens or max_new))
        if tools:
            kw["tools"] = tools
        if o.temperature is not None:
            kw["temperature"] = float(o.temperature)
        if o.reasoning_effort:
            kw["reasoning_effort"] = o.reasoning_effort
        if o.extra_body:
            kw["extra_body"] = dict(o.extra_body)
        return kw

    def llm_error(self, e):
        """HostedLLM's mapping, plus what a stream raises once it has started: an error event in the stream
        (openai.APIError) and a connection dropped or timed out (httpx's own errors)."""
        err = super().llm_error(e)
        if err is not None:
            return err
        import httpx, openai
        if isinstance(e, httpx.TimeoutException):
            return LLMError("timeout")
        if isinstance(e, httpx.TransportError):
            return LLMError("connection")
        if isinstance(e, openai.APIError):
            log.warning("llm %s: error in the stream", self.name)
            return LLMError("server")
        return None

    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        kw = self.request(messages, tools, max_new)
        self.last_request = time.monotonic()

        async def chunks():
            s = await self.client.chat.completions.create(**kw)
            try:
                async for c in s:
                    yield c
            finally:
                self.last_used = time.monotonic()
                await s.close()                          # also on cancel (barge-in): closes the HTTP stream

        p = ChatStreamParser(tools)
        try:
            for c in self.eloop.iterate(lambda: self.guarded(chunks), stop_flag=stop_flag):
                yield from p.feed(c)
        finally:
            self._count(p.usage)
        if stop_flag and stop_flag():
            return
        yield from p.finish()

    def _count(self, usage):
        self.usage["requests"] += 1
        for k, v in (usage or {}).items():
            self.usage[k] += v

    def shutdown(self):
        if self.pinger is not None:
            self.eloop.loop.call_soon_threadsafe(self.pinger.cancel)
        super().shutdown()
