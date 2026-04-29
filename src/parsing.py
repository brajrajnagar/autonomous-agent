"""JSON-extraction helpers and action hashing for loop detection.

LLMs frequently wrap JSON in prose or markdown fences. These helpers try
multiple strategies to recover a usable dict from the raw text.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional


def parse_response(response: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Tries balanced-brace extraction, then a coarse first-`{` to last-`}`
    fallback, then several common-prefix heuristics. Returns the response
    wrapped as `{"thought": ...}` if no JSON object can be parsed.
    """
    response = response.strip()

    # Strategy 1: balanced-brace scan, return the first complete JSON object.
    try:
        brace_count = 0
        start = -1
        for i, char in enumerate(response):
            if char == "{":
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    return json.loads(response[start:i + 1])
    except json.JSONDecodeError:
        pass

    # Strategy 2: outermost { ... } fallback.
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    # Strategy 3: skip past common LLM prefixes.
    for prefix in ("Here's the result:", "Sure!", "OK", "Certainly", "Response:"):
        idx = response.find(prefix)
        if idx != -1:
            try:
                remaining = response[idx:]
                start = remaining.find("{")
                end = remaining.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(remaining[start:end])
            except json.JSONDecodeError:
                pass

    return {"thought": response}


def safe_json_parse(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Strict JSON extraction used by planning methods.

    Strips markdown fences, then returns the first balanced-brace JSON
    object found. Returns None on any parse failure (caller decides
    fallback behavior).
    """
    if text is None:
        return None
    text = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if text.startswith(fence):
            text = text[len(fence):].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()

    brace_count = 0
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start != -1:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def hash_action(tool_name: str, parameters: Dict[str, Any]) -> str:
    """Stable hash of a tool call (for repeated-action loop detection)."""
    action_str = f"{tool_name}:{str(sorted(parameters.items()))}"
    return hashlib.md5(action_str.encode()).hexdigest()


# Qwen-style tool calls emitted as XML inside `content` when the provider
# doesn't translate them to OpenAI's structured `tool_calls` field. Format:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>VALUE</parameter>
#   ...
#   </function>
#   </tool_call>
_QWEN_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN_FUNCTION = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_QWEN_PARAMETER = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def parse_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse Qwen-style `<tool_call>...<function=NAME>...</function></tool_call>` blocks.

    Returns a list of `{"name": str, "arguments": dict}` for each tool call
    found, or an empty list when none are present. Used as a fallback when
    a provider accepts the OpenAI `tools=...` parameter but emits its
    tool calls as text inside `content` (e.g. Qwen3.5 via Clarifai).
    """
    if not text:
        return []
    out: List[Dict[str, Any]] = []
    for tc_match in _QWEN_TOOL_CALL.finditer(text):
        body = tc_match.group(1)
        fn_match = _QWEN_FUNCTION.search(body)
        if not fn_match:
            continue
        name = fn_match.group(1).strip()
        params: Dict[str, Any] = {}
        for p_match in _QWEN_PARAMETER.finditer(fn_match.group(2)):
            key = p_match.group(1).strip()
            value = p_match.group(2).strip()
            params[key] = value
        out.append({"name": name, "arguments": params})
    return out
