import json
import re


def extract_json_object(text):
    """
    Pull a JSON object out of an LLM response.

    It first tries to parse the whole text. If that fails it falls back
    to a regex for a brace span, and that regex is greedy: it matches
    from the first `{` to the LAST `}`, not the first complete object.
    So a response holding two separate objects matches both of them plus
    whatever sits in between, which is not valid JSON, and the result is
    an empty dictionary.

    Returns the parsed dictionary. Returns an empty dictionary when there
    is no brace span at all, or when the whole span is not one valid JSON
    object.
    """
    teks = str(text or "").strip()

    if not teks:
        return {}

    try:
        payload = json.loads(teks)

        if isinstance(payload, dict):
            return payload

    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", teks, flags=re.DOTALL)

    if not match:
        return {}

    try:
        payload = json.loads(match.group(0))

        if isinstance(payload, dict):
            return payload

    except json.JSONDecodeError:
        pass

    return {}


__all__ = [
    "extract_json_object",
]