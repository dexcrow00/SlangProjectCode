"""Orchestrates model × prompt combinations with retry logic."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .client import TogetherClient
from .collector import ResponseCollector
from .prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Retry on Together rate-limit (429) or server errors (5xx).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status in _RETRYABLE_STATUS


class Runner:
    def __init__(
        self,
        client: TogetherClient,
        collector: ResponseCollector,
        models: list[str],
        gen_kwargs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ):
        self.client = client
        self.collector = collector
        self.models = models
        self.gen_kwargs = gen_kwargs or {}
        self.run_id = run_id or uuid.uuid4().hex

    def run(self, prompts: list[PromptTemplate]) -> None:
        """Iterate over every model × prompt combination and collect responses."""
        combos = [(model, prompt) for model in self.models for prompt in prompts]
        logger.info(
            "Starting run %s — %d model(s) × %d prompt(s) = %d requests",
            self.run_id,
            len(self.models),
            len(prompts),
            len(combos),
        )

        for model, prompt in tqdm(combos, desc="Prompting", unit="req"):
            self._process(model, prompt)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call(self, model: str, messages: list[dict]) -> dict:
        return self.client.complete(model, messages, **self.gen_kwargs)

    def _process(self, model: str, prompt: PromptTemplate) -> None:
        system_text, user_text = prompt.render()
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        try:
            result = self._call(model, messages)
        except Exception as exc:
            logger.error("Failed model=%s prompt=%s: %s", model, prompt.id, exc)
            return

        record = {
            "run_id": self.run_id,
            "model": model,
            "prompt_id": prompt.id,
            "prompt_text": user_text,
            "system_text": system_text,
            "response": result["text"],
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.collector.save(record)
