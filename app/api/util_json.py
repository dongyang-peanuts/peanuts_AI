import json


def extract_outer_json(s: str) -> str:
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("No JSON object found in the response")
    return s[i:j+1]


def parse_json_or_raise(s: str) -> dict:
    return json.loads(extract_outer_json(s))