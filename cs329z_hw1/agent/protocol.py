"""Problem 6 (tool_protocol): how the model writes a tool call, as plain text.

The LM wrapper has no tool-calling feature, so the protocol is text on both
sides of the call:

    <optional prose>
    <tool_call>
    {"name": "add", "args": {"a": 17, "b": 25}}
    </tool_call>

* ``format_tool_call``  writes a reply in this format (used by the tests' ScriptedLM).
* ``parse_response``    reads a reply: prose, at most one call, or an error.
* ``render_tools``      the system-prompt text that teaches a model the format.

Why this format
* The call is a JSON object, which models write reliably, with a real parser
  behind it, so argument types (1, 1.0, true, "1") survive exactly.
* The JSON is read with ``JSONDecoder.raw_decode``, which finds where the object
  ends by JSON syntax, not by searching for the closing tag. So a string
  argument may contain ``</tool_call>``, or a whole call in this format, without
  confusing the parser. The formatter also writes ``<`` as ``\\u003c`` inside
  the JSON, so a formatted reply contains the opening tag exactly once.
* A cut-off reply cannot parse as a different call: a prefix of a JSON object
  is not a complete object until its final ``}``. A cut inside the arguments is
  an error (the reply is malformed, not a final answer); a cut that removes
  only the closing tag still gives the complete, original call.
* Two calls in one reply is an error: one action per step keeps approval
  (Problem 13) and the transcript simple.
"""

from __future__ import annotations

import json
import re

from cs329z_hw1.types import ParsedResponse, ToolCall

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
_MIN_PARTIAL_TAG = 5  # "<tool" at the end of a reply means the reply was cut inside the tag

_DECODER = json.JSONDecoder()
_BARE_START = re.compile(r'\{\s*"name"\s*:')


# ------------------------------------------------------------- formatting --


def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps({"name": call.name, "args": call.args}, ensure_ascii=False)
    payload = payload.replace("<", "\\u003c")  # only occurs inside strings; same value when decoded
    block = f"{OPEN}\n{payload}\n{CLOSE}"
    prose = (text or "").strip()
    return f"{prose}\n{block}" if prose else block


# ---------------------------------------------------------------- parsing --


def _error(prose: str, message: str) -> ParsedResponse:
    return ParsedResponse(text=prose.strip(), tool_call=None, error=message)


def parse_response(text: str) -> ParsedResponse:
    """Never raises: anything unexpected becomes ``error``."""
    try:
        return _parse(text if isinstance(text, str) else "")
    except Exception as exc:  # RecursionError on absurdly nested JSON, and the like
        prose = text.strip() if isinstance(text, str) else ""
        return ParsedResponse(text=prose, tool_call=None, error=f"Could not read the reply: {exc}")


def _bare_calls(text: str) -> list[tuple[int, int, ToolCall]]:
    """Complete ``{"name": ..., "args": {...}}`` objects written without the tags.

    The grading model sometimes writes a correct call but leaves out the
    tags. Only an object whose keys are exactly a string "name" plus an
    "args" object counts, so ordinary prose with braces stays prose.
    """
    found = []
    pos = text.find("{")
    while pos != -1:
        match = _BARE_START.match(text, pos)
        if match:
            try:
                obj, end = _DECODER.raw_decode(text, pos)
            except ValueError:
                obj, end = None, pos + 1
            if (
                isinstance(obj, dict)
                and set(obj) == {"name", "args"}
                and isinstance(obj["name"], str)
                and obj["name"].strip()
                and isinstance(obj["args"], dict)
            ):
                found.append((pos, end, ToolCall(name=obj["name"].strip(), args=obj["args"])))
                pos = text.find("{", end)
                continue
        pos = text.find("{", pos + 1)
    return found


def _parse(text: str) -> ParsedResponse:
    start = text.find(OPEN)
    if start == -1:
        bare = _bare_calls(text)
        if len(bare) > 1:
            return _error(
                text,
                f"The reply contains more than one tool call. Write exactly one, inside {OPEN} ... {CLOSE}.",
            )
        if bare:
            begin, end, call = bare[0]
            return ParsedResponse(text=_join(text[:begin], text[end:]), tool_call=call, error=None)
        prose = text.strip()
        for n in range(len(OPEN) - 1, _MIN_PARTIAL_TAG - 1, -1):
            if prose.endswith(OPEN[:n]):
                return _error(
                    prose,
                    f"The reply ends in the middle of a {OPEN} tag, so it looks cut off. "
                    f"Write the whole call: {OPEN}, a JSON object with \"name\" and \"args\", then {CLOSE}.",
                )
        return ParsedResponse(text=prose, tool_call=None, error=None)

    before = text[:start]
    pos = start + len(OPEN)
    pos = _skip_space(text, pos)
    fenced = text.startswith("```", pos)  # tolerate a markdown fence inside the tags
    if fenced:
        newline = text.find("\n", pos)
        if newline == -1:
            return _error(before, _incomplete())
        pos = _skip_space(text, newline + 1)

    try:
        obj, end = _DECODER.raw_decode(text, pos)
    except ValueError:
        return _error(before, _incomplete())

    end = _skip_space(text, end)
    if fenced and text.startswith("```", end):
        end = _skip_space(text, end + 3)
    if text.startswith(CLOSE, end):
        end += len(CLOSE)  # a missing closing tag is tolerated: the JSON object was complete

    after = text[end:]
    if OPEN in after:
        return _error(
            before,
            f"The reply contains more than one {OPEN}. Write exactly one tool call per reply, "
            "then wait for its result.",
        )

    call, problem = _to_call(obj)
    prose = _join(before, after)
    if call is None:
        return _error(prose, problem)
    return ParsedResponse(text=prose, tool_call=call, error=None)


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _incomplete() -> str:
    return (
        f"The tool call is incomplete or is not valid JSON. Write {OPEN}, then one JSON object "
        f'like {{"name": "tool_name", "args": {{...}}}}, then {CLOSE}. '
        "If the reply was cut off, make the call shorter."
    )


def _to_call(obj) -> tuple[ToolCall | None, str]:
    if not isinstance(obj, dict):
        return None, 'The tool call must be a JSON object with a "name" and an "args" object.'
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, 'The tool call needs a "name": the name of the tool, as a string.'
    args = obj.get("args", obj.get("arguments", obj.get("parameters")))
    if args is None:
        args = {}
    if isinstance(args, str):  # some models write the arguments as a JSON string
        try:
            args = json.loads(args)
        except ValueError:
            pass
    if not isinstance(args, dict):
        return None, 'The "args" of a tool call must be a JSON object, for example {"query": "budget"}.'
    return ToolCall(name=name.strip(), args=args), ""


def _join(before: str, after: str) -> str:
    before, after = before.strip(), after.strip()
    return f"{before}\n{after}" if before and after else before or after


# --------------------------------------------------------------- rendering --


def render_tools(schemas: list[dict]) -> str:
    """Text for the system prompt: the tools, and how to call one."""
    lines = [
        "# Tools",
        "",
        "This conversation has no built-in function calling: no functions are attached to it, and "
        "there is no special channel for calls. You use a tool by writing the call as ordinary text "
        "in your visible reply. The system reads that text, runs the tool, and sends you the result.",
        "",
        "To call a tool, end your reply with exactly one call written in this format:",
        "",
        OPEN,
        '{"name": "<tool name>", "args": {<arguments>}}',
        CLOSE,
        "",
        "For example, to call a hypothetical tool named get_weather with the argument city set to Paris, "
        "the whole call is:",
        "",
        OPEN,
        '{"name": "get_weather", "args": {"city": "Paris"}}',
        CLOSE,
        "",
        f"Rules for tool calls:",
        f"- The tags are always literally {OPEN} and {CLOSE}, whatever the tool. The tool's name goes in "
        '"name" and its arguments in "args"; never use the tool\'s name as a tag.',
        '- "args" is a JSON object that follows the tool\'s schema: use the exact argument names, '
        'give numbers as numbers (17, not "17"), and include every required argument.',
        "- Make at most one tool call per reply, then stop. The result comes back in the next message, "
        "labelled with the tool's name and status. Never write a result yourself.",
        "- Copy text arguments exactly as the user gave them, using standard JSON string escaping once: "
        'each backslash becomes \\\\ and each double quote becomes \\". For example, the path '
        'C:\\Users\\ann\\a.txt is written "C:\\\\Users\\\\ann\\\\a.txt". Never escape twice.',
        "- Do not put the call in a markdown code block. A short sentence before the call is fine.",
        "- When a tool can do what the user asks, call it instead of answering yourself, even for simple "
        "arithmetic or text you could write directly. Never say you will call a tool without writing the "
        "call in the same reply.",
        "- To answer the user, or to ask them something, reply in plain text with no tool call. "
        "A reply that contains no call is treated as your final answer.",
        "",
        "Available tools:",
    ]
    for schema in schemas:
        lines += [
            "",
            f"## {schema.get('name', '')}",
            str(schema.get("description", "")).strip(),
            "Arguments (JSON Schema): "
            + json.dumps(schema.get("parameters", {}), ensure_ascii=False, separators=(",", ":")),
        ]
    return "\n".join(lines)
