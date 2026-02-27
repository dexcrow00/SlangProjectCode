"""Prompt template definitions and loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PromptTemplate:
    id: str
    system: str
    user: str
    variables: dict = field(default_factory=dict)

    def render(self, **kwargs) -> tuple[str, str]:
        """Return (system, user) with placeholders filled.

        Merges template-level default variables with call-site overrides.
        """
        ctx = {**self.variables, **kwargs}
        return self.system.format_map(ctx), self.user.format_map(ctx)


def load_prompts(path: str | Path) -> list[PromptTemplate]:
    """Load prompt templates from a JSONL file.

    Supports both compact (one object per line) and pretty-printed (multi-line)
    formats. Each record must have at least 'id', 'system', 'user'.
    An optional 'variables' key holds default interpolation values.
    """
    templates: list[PromptTemplate] = []
    decoder = json.JSONDecoder()
    text = Path(path).read_text(encoding="utf-8")
    pos = 0
    while pos < len(text):
        # Skip whitespace (including blank lines between records).
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at position {pos}: {exc}") from exc
        templates.append(
            PromptTemplate(
                id=obj["id"],
                system=obj["system"],
                user=obj["user"],
                variables=obj.get("variables", {}),
            )
        )
        pos = end
    return templates
