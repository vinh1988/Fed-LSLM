"""Qwen2.5-0.5B-Instruct answer generator for the mRAG pipeline.

Uses retrieved PathVQA training examples as context.
Always returns both ``raw_output`` and ``final_prediction`` for debugging.

Prompt strategy:
- closed questions  → constrained to "yes" / "no"
- open questions    → short medical phrase, no explanation
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Default model to load
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Regex for extracting yes/no from generation output
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _build_prompt(
    question: str,
    question_type: str,
    retrieved_evidence: list[dict],
    max_examples: int = 5,
) -> str:
    """Build the chat-style user prompt.

    Args:
        question: The query question.
        question_type: ``"closed"`` or ``"open"``.
        retrieved_evidence: List of retrieved dicts (rank, score, question, answer, ...).
        max_examples: Maximum number of retrieved examples to include.

    Returns:
        Formatted prompt string (user turn only — system prompt is separate).
    """
    examples_text = ""
    for i, ev in enumerate(retrieved_evidence[:max_examples], start=1):
        q = ev.get("question", "").strip()
        a = ev.get("answer", "").strip()
        examples_text += f"{i}. Q: {q}\n   A: {a}\n"

    if question_type == "closed":
        instruction = (
            "Answer ONLY with 'yes' or 'no'. "
            "Do NOT include any other words or explanation."
        )
    else:
        instruction = (
            "Answer with a short medical phrase (1–5 words). "
            "Do NOT explain or elaborate."
        )

    prompt = (
        f"Question type: {question_type}\n"
        f"Question: {question}\n\n"
        f"Retrieved examples:\n{examples_text}\n"
        f"Instruction: {instruction}\n\n"
        f"Final answer:"
    )
    return prompt


def _postprocess(raw: str, question_type: str) -> str:
    """Extract the final answer from raw model output.

    For closed questions: search for 'yes'/'no', fallback to 'no'.
    For open questions: take first non-empty line after 'Final answer:'.

    Args:
        raw: Full decoded output from the model.
        question_type: ``"closed"`` or ``"open"``.

    Returns:
        Cleaned final answer string.
    """
    # Strip leading/trailing whitespace
    text = raw.strip()

    # Remove any echo of the prompt prefix (models sometimes repeat it)
    if "Final answer:" in text:
        text = text.split("Final answer:")[-1].strip()

    if question_type == "closed":
        match = _YES_NO_RE.search(text)
        if match:
            return match.group(0).lower()
        logger.debug(
            "Could not extract yes/no from output: %r — defaulting to 'no'", raw
        )
        return "no"
    else:
        # Take first non-empty line, strip punctuation artifacts
        for line in text.splitlines():
            line = line.strip().strip(".").strip(",").strip()
            if line:
                # If model output multiple words separated by \n, keep first line
                return line
        # If no lines at all, return raw stripped text (truncated)
        return text[:100]


class QwenAnswerGenerator:
    """Zero-shot answer generator using Qwen2.5-0.5B-Instruct.

    Args:
        model_name: HuggingFace model identifier.
        device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Attributes:
        model_name: Model identifier actually loaded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        device: str = "cpu",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for QwenAnswerGenerator."
            ) from exc

        import torch as _torch

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Resolve device
        if device == "auto":
            device = (
                "cuda" if _torch.cuda.is_available()
                else "mps" if _torch.backends.mps.is_available()
                else "cpu"
            )
        self._device = device

        logger.info("Loading Qwen tokenizer: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        logger.info("Loading Qwen model: %s on %s", model_name, device)
        dtype = _torch.float16 if device in ("cuda", "mps") else _torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

        logger.info("Qwen model loaded successfully | device=%s | dtype=%s", device, dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        question_type: str,
        retrieved_evidence: list[dict],
    ) -> dict[str, str]:
        """Generate an answer for a single question.

        Args:
            question: The query question text.
            question_type: ``"closed"`` or ``"open"``.
            retrieved_evidence: Retrieved training examples from FAISSRetriever.

        Returns:
            Dict with keys:
            - ``raw_output``: Full decoded model generation.
            - ``final_prediction``: Post-processed answer string.
        """
        import torch

        system_prompt = (
            "You are a medical visual question answering assistant. "
            "Answer concisely based on the retrieved examples. "
            "Do not explain your reasoning."
        )
        user_prompt = _build_prompt(question, question_type, retrieved_evidence)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if self.temperature == 0.0:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][input_len:]
        raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        final_prediction = _postprocess(raw_output, question_type)

        logger.debug(
            "question_type=%s | raw=%r | pred=%r",
            question_type,
            raw_output,
            final_prediction,
        )

        return {
            "raw_output": raw_output,
            "final_prediction": final_prediction,
        }

    def batch_generate(
        self,
        questions: list[str],
        question_types: list[str],
        retrieved_evidences: list[list[dict]],
    ) -> list[dict[str, str]]:
        """Generate answers for a list of questions (sequential, not batched).

        Each item is processed individually because prompt lengths vary
        significantly between samples. True batching can be added later.

        Returns:
            List of dicts with ``raw_output`` and ``final_prediction``.
        """
        results = []
        for q, qt, ev in zip(questions, question_types, retrieved_evidences):
            results.append(self.generate_answer(q, qt, ev))
        return results
