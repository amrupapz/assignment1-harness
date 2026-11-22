"""Problem 5 (tool_registry): the agent's toolbox.

The registry keeps the tools in registration order and offers three operations:

* ``schemas()``   what the model is told about each tool (name, description,
                  parameters, and nothing else);
* ``validate()``  checks a call without running it: unknown tool, or arguments
                  that fail the tool's JSON Schema;
* ``execute()``   validates, then runs the tool, and always returns a ToolResult.

Validation is separate from execution because the interactive layer
(Problem 13) must reject a bad call *before* asking the user to approve it.
Neither method raises: whatever the model asks for and whatever a tool does,
the caller gets a ToolResult it can show the model.
"""

from __future__ import annotations

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec

MAX_ERRORS_SHOWN = 3  # a long list of schema errors would only distract the model


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for tool in tools:
            self._tools[tool.name] = tool
            self._validators[tool.name] = Draft202012Validator(tool.parameters)

    # -- what the model sees -------------------------------------------------

    def schemas(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    # -- lookups for the agent loop ------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        """The ToolSpec for ``name``, or None (used to read ``read_only``)."""
        return self._tools.get(name) if isinstance(name, str) else None

    def names(self) -> list[str]:
        return list(self._tools)

    # -- checking and running a call -----------------------------------------

    def validate(self, name: str, args: dict) -> ToolResult | None:
        """None if the call could run; otherwise the result ``execute`` would
        return for it. Never calls the tool and never raises."""
        try:
            return self._validate(name, args)
        except Exception as exc:  # a malformed schema or an odd argument must not escape
            return ToolResult("invalid_args", f"Could not check the arguments of {name!r}: {exc}")

    def _validate(self, name: str, args: dict) -> ToolResult | None:
        if not isinstance(name, str) or name not in self._tools:
            available = ", ".join(self._tools) or "(none)"
            return ToolResult("unknown_tool", f"Unknown tool {name!r}. Available tools: {available}.")
        if not isinstance(args, dict):
            return ToolResult(
                "invalid_args",
                f"Arguments for {name!r} must be a JSON object, got {type(args).__name__}.",
            )
        errors = sorted(self._validators[name].iter_errors(args), key=lambda e: list(map(str, e.path)))
        if not errors:
            return None
        lines = []
        for err in errors[:MAX_ERRORS_SHOWN]:
            where = "/".join(str(p) for p in err.path)
            lines.append(f"{err.message}" + (f" (at '{where}')" if where else ""))
        more = f" (and {len(errors) - MAX_ERRORS_SHOWN} more)" if len(errors) > MAX_ERRORS_SHOWN else ""
        return ToolResult("invalid_args", f"Invalid arguments for {name!r}: " + "; ".join(lines) + more)

    def execute(self, name: str, args: dict) -> ToolResult:
        """Validate, then run the tool once as ``fn(**args)``. Never raises."""
        rejected = self.validate(name, args)
        if rejected is not None:
            return rejected
        try:
            return ToolResult("ok", str(self._tools[name].fn(**args)))
        except Exception as exc:
            return ToolResult("tool_error", f"{type(exc).__name__}: {exc}")
