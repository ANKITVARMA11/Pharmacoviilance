"""Preflight check:  python -m pvcrew.check

Confirms, in order: the .env file, your key, the models your key can use, which model gets picked, and
one real structured extraction. Costs about 2 API calls. Prints only a masked form of your key.
"""
import sys

from . import config
from .extract import ExtractionError, extract_case, extraction_test
from .llm import ModelUnavailableError, get_client, resolve_model

EXAMPLE = (
    "Hi, my mum (67) started taking Amoxicillin last week and was rushed to hospital "
    "with a rash and trouble breathing. She's recovering now."
)


def run(client=None, say=print) -> bool:
    say(f"1. .env file : {config.ENV_FILE} ({'found' if config.ENV_FILE.exists() else 'NOT found'})")
    say(f"2. provider  : {config.PROVIDER}")
    if config.PROVIDER == "nim":
        say(f"   API key   : {config.mask(config.NVIDIA_API_KEY)}")
    try:
        client = client or get_client()
    except SystemExit as e:
        say(f"FAIL: {e}")
        return False
    try:
        if hasattr(client, "list_models"):
            say(f"3. models on your key: {len(client.list_models())}")
        say("4. finding a model that works (testing real extraction):")
        works, cases = extraction_test(client)
        model = resolve_model(client, accept=works, say=say)
        say(f"   model chosen: {model}")
        case = cases.get(model) or extract_case(EXAMPLE, client, model)
    except ModelUnavailableError as e:
        say(f"FAIL: {e}")
        return False
    except ExtractionError as e:
        say(f"FAIL: the model answered but never produced valid JSON ({e}). Try another model via PV_MODEL in .env.")
        return False
    except Exception as e:  # noqa: BLE001 - turn the common failures into plain advice
        status = getattr(e, "status_code", None)
        if status in (401, 403):
            say(f"FAIL: NVIDIA rejected the key (HTTP {status}). Check NVIDIA_API_KEY in {config.ENV_FILE}: "
                "paste only the key (it starts with nvapi-), no quotes or spaces.")
        elif status == 429:
            say("FAIL: rate limited (HTTP 429) even after retries. Wait a minute and run again.")
        else:
            say(f"FAIL: {type(e).__name__}: {e}")
        return False
    say(f"5. test extraction: {case.model_dump()}")
    say("OK: .env, key, model and structured output all work.")
    if not config.env("PV_MODEL") and hasattr(client, "list_models"):
        say(f"Tip: add this line to {config.ENV_FILE} to skip the model search next time:  PV_MODEL={model}")
    return True


def main() -> None:
    sys.exit(0 if run() else 1)


if __name__ == "__main__":
    main()
