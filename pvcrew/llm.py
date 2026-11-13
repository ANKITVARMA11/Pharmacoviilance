"""LLM clients. Everything else in the project talks to one tiny interface:

    client.chat(model=..., messages=[...], format=<json schema or None>, options={"temperature": ...})
        -> {"message": {"content": "<text>"}}

`NimClient` adapts an OpenAI-compatible NVIDIA NIM endpoint to that interface (hosted at
build.nvidia.com or a self-hosted NIM container). Ollama's client already matches it.

List the models your key can use:   python -m pvcrew.llm [filter]
"""
import json
import math
import re
import sys
import time

from . import config

RETRY_STATUS = {429}  # plus any 5xx, see _retryable
RETRY_NAMES = {"APIConnectionError", "APITimeoutError"}  # openai exceptions without a status code


class ModelUnavailableError(RuntimeError):
    """The model is retired (410) or the ID is unknown (404). Retrying will not help."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _status(exc: Exception):
    return getattr(exc, "status_code", None)


def _retryable(exc: Exception) -> bool:
    s = _status(exc)
    return (s in RETRY_STATUS) or (s is not None and s >= 500) or type(exc).__name__ in RETRY_NAMES


def _with_schema(messages: list[dict], schema: dict) -> list[dict]:
    """Copy of messages with the JSON schema spelled out in the system prompt (NIM docs recommend this)."""
    note = "Respond with a single JSON object that matches this JSON schema:\n" + json.dumps(schema)
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = msgs[0]["content"] + "\n\n" + note
    else:
        msgs.insert(0, {"role": "system", "content": note})
    return msgs


class NimClient:
    def __init__(self, openai_client, min_interval: float = config.NIM_MIN_INTERVAL, max_retries: int = 4,
                 backoff: float = 2.0, sleep=time.sleep, clock=time.monotonic):
        self._client = openai_client
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep
        self._clock = clock
        self._last_call = None
        self._guided = True  # flips to False if the server rejects nvext.guided_json

    def _pace(self) -> None:
        if self._last_call is not None:
            wait = self._min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _create(self, kwargs: dict, retries: int | None = None) -> str:
        limit = self._max_retries if retries is None else retries
        attempt = 0
        timeouts = 0
        while True:
            self._pace()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - classified below
                if _status(e) in (404, 410):
                    raise ModelUnavailableError(
                        f"Model {kwargs.get('model')!r} is not available (HTTP {_status(e)}): it was probably "
                        "retired, or the ID is wrong. List current models with `python -m pvcrew.llm`, "
                        "then set PV_MODEL to one of them.",
                        status=_status(e),
                    ) from e
                if _status(e) == 400 and "extra_body" in kwargs:
                    # Server does not accept guided_json for this model: fall back to prompt-only JSON.
                    # extract_case still validates and retries, so correctness is unchanged.
                    self._guided = False
                    kwargs = {k: v for k, v in kwargs.items() if k != "extra_body"}
                    continue
                if type(e).__name__ == "APITimeoutError":
                    # A stalled model is not worth minutes of retries. First, drop guided_json (constrained
                    # decoding is the usual cause of a hang) and try once more; a second timeout gives up.
                    timeouts += 1
                    if timeouts > 1:
                        raise
                    if "extra_body" in kwargs:
                        self._guided = False
                        kwargs = {k: v for k, v in kwargs.items() if k != "extra_body"}
                    continue
                if _retryable(e) and attempt < limit:
                    self._sleep(min(self._backoff * 2**attempt, 30))
                    attempt += 1
                    continue
                raise

    def chat(self, model: str, messages: list[dict], format: dict | None = None,
             options: dict | None = None, **_ignored) -> dict:
        opts = options or {}
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": opts.get("temperature", 0),
        }
        if "max_tokens" in opts:
            kwargs["max_tokens"] = opts["max_tokens"]
        if "timeout" in opts:  # seconds; lets a quick probe give up on a stalled model
            kwargs["timeout"] = opts["timeout"]
        if format is not None:
            kwargs["messages"] = _with_schema(messages, format)
            if self._guided:
                kwargs["extra_body"] = {"nvext": {"guided_json": format}}
        return {"message": {"content": self._create(kwargs, opts.get("retries"))}}

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list())


def get_client(provider: str | None = None):
    provider = (provider or config.PROVIDER).lower()
    if provider == "nim":
        key = config.NVIDIA_API_KEY
        if not key and config.NIM_BASE_URL == config.NIM_HOSTED_URL:
            where = (f"Open {config.ENV_FILE} and put your key after NVIDIA_API_KEY="
                     if config.ENV_FILE.exists()
                     else f"Create {config.ENV_FILE} (copy .env.example) and put your key after NVIDIA_API_KEY=")
            raise SystemExit(
                f"NVIDIA_API_KEY is not set. {where}. "
                "Get a free key at https://build.nvidia.com (open any model, click 'Get API Key')."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("The 'openai' package is missing. Run: pip install -r requirements.txt") from e
        # max_retries=0: NimClient does its own pacing and retrying
        return NimClient(OpenAI(base_url=config.NIM_BASE_URL, api_key=key or "not-used", timeout=60, max_retries=0))
    if provider == "ollama":
        try:
            import ollama
        except ImportError as e:
            raise SystemExit("The 'ollama' package is missing. Run: pip install ollama") from e
        return ollama.Client(host=config.OLLAMA_HOST)
    raise ValueError(f"unknown provider {provider!r}; use 'nim' or 'ollama'")


PROBE = [{"role": "user", "content": "Reply with the single word OK."}]
PROBE_OPTIONS = {"temperature": 0, "max_tokens": 5, "timeout": 20, "retries": 1}
SKIP_STATUS = {400, 403, 404, 410, 422}  # about this model only. 401 (bad key) and 429 (throttled) are not.

# Name fragments of models that are not general chat/instruct models (embedding, vision, safety, code, ...).
_NOT_CHAT = ("embed", "rerank", "reward", "guard", "safety", "topic-control", "vision", "-vl", "vlm", "parse",
             "code", "coder", "starcoder", "deplot", "fuyu", "kosmos", "neva", "vila", "clip", "riva", "translate",
             "retriev", "cosmos", "video", "detector", "calibration", "diffusion", "reasoning", "-r1", "thinking")


def _short(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _is_chat_like(model_id: str) -> bool:
    return not any(word in _short(model_id) for word in _NOT_CHAT)


def _size_b(model_id: str):
    m = re.search(r"(\d+(?:\.\d+)?)b(?![a-z])", _short(model_id))
    return float(m.group(1)) if m else None


def _closeness(model_id: str) -> float:
    """Prefer models near 30B: strong enough for JSON extraction and quick on the free tier. Unknown size goes last."""
    size = _size_b(model_id)
    return 99.0 if size is None else abs(math.log(size / 30.0))


def rank_models(available, candidates) -> list[str]:
    """The order to try models in: preferred candidates that are listed, then every other chat-like model,
    with plainly instruction-tuned ones ('instruct' / '-it') ahead of the rest."""
    listed = set(available)
    first = [m for m in candidates if m in listed]

    def tier(m):
        n = _short(m)
        return 0 if ("instruct" in n or n.endswith("-it")) else 1

    rest = sorted((m for m in available if m not in first and _is_chat_like(m)),
                  key=lambda m: (tier(m), _closeness(m), m))
    return first + rest


def _skip_reason(exc: Exception):
    """Why a model should be skipped, or None if the error is bigger than this one model."""
    status = _status(exc)
    if isinstance(exc, ModelUnavailableError):
        return f"retired or unavailable (HTTP {exc.status})"
    if status in SKIP_STATUS:
        return f"rejected the request (HTTP {status}), probably not a chat model"
    if (status is not None and status >= 500) or type(exc).__name__ in RETRY_NAMES:
        return "too slow or erroring (timeout or server error)"
    return None


def resolve_model(client, explicit: str | None = None, candidates=None, accept=None, max_tries: int | None = None,
                  say=None, discover: bool = True, exclude=()) -> str:
    """Pick the model to use. An explicit name (argument or PV_MODEL) always wins untouched.

    Otherwise try models in `rank_models` order until one answers a 5-token test call (and, if given,
    `accept(model)` returns True). `say` receives progress lines. Never trusts the model list alone: NVIDIA's
    list still contains retired models.
    """
    explicit = explicit or config.env("PV_MODEL")
    if explicit:
        return explicit
    if not hasattr(client, "list_models"):  # Ollama
        return config.DEFAULT_MODELS["ollama"]
    say = say or (lambda *_: None)
    available = client.list_models()
    candidates = config.NIM_CANDIDATES if candidates is None else candidates
    order = rank_models(available, candidates) if discover else [m for m in candidates if m in set(available)]
    order = [m for m in order if m not in set(exclude)]
    limit = max_tries or config.NIM_MAX_MODEL_TRIES
    skipped = []
    for model in order[:limit]:
        say(f"   trying {model} ...")
        try:
            client.chat(model=model, messages=PROBE, options=PROBE_OPTIONS)
            passed = accept is None or accept(model)
        except Exception as e:  # noqa: BLE001 - classified by _skip_reason
            reason = _skip_reason(e)
            if reason is None:
                raise
            skipped.append((model, reason))
            say(f"      skipped: {reason}")
            continue
        if not passed:
            skipped.append((model, "answered, but failed the extraction test"))
            say("      skipped: answered, but failed the extraction test")
            continue
        say("      ok")
        return model
    lines = [f"  - {m}: {why}" for m, why in skipped] or ["  (no chat models found on your key)"]
    untried = order[limit:]
    more = (f"\n{len(untried)} more were not tried (e.g. {', '.join(untried[:3])}). "
            "Raise NIM_MAX_MODEL_TRIES in .env to try more." if untried else "")
    raise ModelUnavailableError(
        f"None of the {len(skipped)} models tried worked on your key:\n" + "\n".join(lines) + more +
        "\nOr pick one from `python -m pvcrew.llm` and set PV_MODEL in .env."
    )


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    client = get_client()
    if not hasattr(client, "list_models"):
        raise SystemExit("Listing models is only supported for the nim provider.")
    needle = argv[0].lower() if argv else ""
    for model_id in client.list_models():
        if needle in model_id.lower():
            print(model_id)


if __name__ == "__main__":
    main()
