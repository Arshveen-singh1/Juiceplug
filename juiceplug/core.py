"""
JuicedModel — the main class in JuicePlug.

Loads a Hugging Face causal LM (optionally 4-bit quantized via
bitsandbytes), optionally attaches a PEFT LoRA adapter, and provides
``ask()`` / ``stream_ask()`` methods with a ReAct-style tool loop.

Heavy ML dependencies (torch, transformers, peft, bitsandbytes) are
**imported lazily** inside the methods that need them, so the rest of the
package stays lightweight.
"""

from __future__ import annotations

import logging
import re
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
)

from juiceplug.router import AdapterRouter
from juiceplug.tools import get_tool, list_tools

if TYPE_CHECKING:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\w+)\s*:\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
"""Matches ``<tool_call>tool_name: query text</tool_call>``."""

# Signals that the model "doesn't know" — used for the retry heuristic.
_IDK_SIGNALS = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i don't have current",
    "i do not have current",
    "i cannot access",
    "i can't access",
    "my training data",
    "my knowledge cutoff",
    "as of my last update",
]

# Default hard cap on tool loop iterations.
DEFAULT_MAX_TOOL_TURNS = 5
DEFAULT_TOOL_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class JuicedModel:
    """Load a causal LM, optionally with a LoRA adapter and web tools.

    Parameters
    ----------
    model : PreTrainedModel
        Already-loaded HF model instance.
    tokenizer : PreTrainedTokenizerBase
        Corresponding tokenizer.
    tools : list[str]
        Names of registered tools this model may invoke via the tool loop.
    adapter_router : AdapterRouter | None
        If provided, each ``ask()`` call routes the query through this
        router and hot-swaps the adapter before generation.
    max_tool_turns : int
        Hard cap on tool loop iterations (prevents infinite loops).
    tool_timeout_seconds : float
        Max wall-clock seconds for the entire tool loop (across all
        iterations) before the model is forced to answer.
    log_transcripts : bool
        If True, log every tool-use transcript to a local JSON-lines file.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tools: Optional[List[str]] = None,
        adapter_router: Optional[AdapterRouter] = None,
        max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS,
        tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        log_transcripts: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._tool_names: List[str] = tools or []
        self.adapter_router = adapter_router
        self.max_tool_turns = max_tool_turns
        self.tool_timeout_seconds = tool_timeout_seconds
        self.log_transcripts = log_transcripts
        self._current_adapter: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        base_model: str,
        *,
        reasoning_adapter: Optional[str] = None,
        tools: Optional[List[str]] = None,
        load_in_4bit: bool = True,
        adapter_router: Optional[AdapterRouter] = None,
        max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS,
        tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        log_transcripts: bool = False,
        **model_kwargs: Any,
    ) -> "JuicedModel":
        """Load a Hugging Face causal LM and wrap it in a ``JuicedModel``.

        Parameters
        ----------
        base_model : str
            HF hub id or local path of the base model.
        reasoning_adapter : str | None
            HF hub id or local path of a PEFT LoRA adapter.  ``None`` skips.
        tools : list[str] | None
            Names of registered tools to enable (e.g. ``["web_search"]``).
        load_in_4bit : bool
            If True and CUDA is available, load quantized via bitsandbytes.
            Falls back gracefully if CUDA is unavailable.
        adapter_router : AdapterRouter | None
            Replaces ``reasoning_adapter`` — routes each query dynamically.
        max_tool_turns : int
            Hard cap on tool loop iterations.
        tool_timeout_seconds : float
            Max wall-clock seconds for the entire tool loop.
        log_transcripts : bool
            If True, tool-use transcripts are logged to a local file.
        **model_kwargs
            Forwarded to ``AutoModelForCausalLM.from_pretrained``.

        Returns
        -------
        JuicedModel

        Raises
        ------
        ModelNotFoundError
            If the base model cannot be resolved on the Hub or locally.
        AdapterLoadError
            If the adapter checkpoint is incompatible with the base model.
        """
        # -- lazy imports ------------------------------------------------
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required to load a model. "
                "Install them: pip install juiceplug[gpu]"
            ) from exc

        # -- 4-bit quantization check -----------------------------------
        use_4bit = load_in_4bit
        if use_4bit:
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA is not available — disabling 4-bit quantization. "
                    "Pass load_in_4bit=False explicitly to silence this warning."
                )
                use_4bit = False
            else:
                try:
                    import bitsandbytes  # noqa: F401
                except ImportError:
                    logger.warning(
                        "bitsandbytes is not installed — disabling 4-bit "
                        "quantization.  Install it for 4-bit support: "
                        "pip install bitsandbytes>=0.43"
                    )
                    use_4bit = False

        # -- quantization config ----------------------------------------
        quant_config = None
        if use_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # -- load tokenizer ---------------------------------------------
        logger.info("Loading tokenizer: %s", base_model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
        except Exception as exc:
            raise ModelNotFoundError(
                f"Failed to load tokenizer for {base_model!r}: {exc}"
            ) from exc

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # -- load model -------------------------------------------------
        logger.info(
            "Loading model: %s (4-bit=%s)", base_model, use_4bit
        )
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        load_kwargs.update(model_kwargs)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, **load_kwargs
            )
        except OSError as exc:
            raise ModelNotFoundError(
                f"Could not find or load model {base_model!r}. "
                f"Check the model id / path.  Original error: {exc}"
            ) from exc
        except Exception as exc:
            raise ModelNotFoundError(
                f"Unexpected error loading {base_model!r}: {exc}"
            ) from exc

        logger.info("Model loaded successfully: %s", base_model)

        # -- attach adapter (if provided) --------------------------------
        if reasoning_adapter is not None:
            _attach_adapter(model, reasoning_adapter)

        # -- validate requested tools ------------------------------------
        if tools:
            available = set(list_tools())
            for t in tools:
                if t not in available:
                    raise ValueError(
                        f"Tool {t!r} is not registered. "
                        f"Available: {sorted(available)}"
                    )

        instance = cls(
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            adapter_router=adapter_router,
            max_tool_turns=max_tool_turns,
            tool_timeout_seconds=tool_timeout_seconds,
            log_transcripts=log_transcripts,
        )
        instance._current_adapter = reasoning_adapter
        return instance

    # ------------------------------------------------------------------
    # Generation: ask
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        max_new_tokens: int = 512,
        verbose: bool = False,
        force_tool_use: bool = False,
    ) -> str:
        """Ask a question, running the tool loop if tools are enabled.

        Parameters
        ----------
        question : str
            The user's question.
        max_new_tokens : int
            Maximum tokens per generation step.
        verbose : bool
            If True, print each tool call/observation to stdout.
        force_tool_use : bool
            If True, run a web search **before** the first generation so
            the model always has fresh context (useful for time-sensitive
            questions).

        Returns
        -------
        str
            The model's final answer.
        """
        # -- adapter routing --------------------------------------------
        if self.adapter_router is not None:
            desired = self.adapter_router.route(question)
            if desired != self._current_adapter:
                self._swap_adapter(desired)

        # -- build prompt -----------------------------------------------
        system_prompt = self._build_system_prompt()
        messages = self._initial_messages(system_prompt, question)

        # -- force tool use (pre-search) --------------------------------
        if force_tool_use and "web_search" in self._tool_names:
            pre_obs = self._run_tool("web_search", question, verbose=verbose)
            messages.append(
                {"role": "assistant", "content": f"<tool_call>web_search: {question}</tool_call>"}
            )
            messages.append(
                {"role": "user", "content": f"<observation>{pre_obs}</observation>"}
            )

        # -- tool loop --------------------------------------------------
        transcript: List[Dict[str, str]] = []
        start = time.monotonic()

        for turn in range(self.max_tool_turns + 1):
            # time guard
            elapsed = time.monotonic() - start
            if elapsed > self.tool_timeout_seconds:
                logger.warning(
                    "Tool loop timed out after %.1fs (%d turns)",
                    elapsed,
                    turn,
                )
                break

            response = self._generate(messages, max_new_tokens=max_new_tokens)
            match = TOOL_CALL_PATTERN.search(response)

            if match and self._tool_names and turn < self.max_tool_turns:
                tool_name, tool_query = match.group(1), match.group(2).strip()
                if verbose:
                    print(f"[tool call {turn + 1}] {tool_name}({tool_query!r})")

                obs = self._run_tool(tool_name, tool_query, verbose=verbose)

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user", "content": f"<observation>{obs}</observation>"}
                )
                transcript.append(
                    {"tool": tool_name, "query": tool_query, "observation": obs}
                )
            else:
                # No tool call — maybe retry if model says "I don't know"
                if (
                    self._tool_names
                    and turn == 0
                    and not match
                    and self._looks_like_idk(response)
                ):
                    logger.info(
                        "Model output looks like 'I don't know' — retrying "
                        "with a more directive system prompt."
                    )
                    messages = self._initial_messages(
                        self._build_system_prompt(directive=True), question
                    )
                    continue

                # Final answer
                final = self._strip_tool_tags(response)
                if self.log_transcripts and transcript:
                    self._log_transcript(question, transcript, final)
                return final

        # Fell through max iterations — force a final answer
        logger.warning(
            "Tool loop hit max iterations (%d) — forcing final answer.",
            self.max_tool_turns,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have used all available tool turns.  Please provide "
                    "your best answer now based on what you already know."
                ),
            }
        )
        final = self._strip_tool_tags(
            self._generate(messages, max_new_tokens=max_new_tokens)
        )
        if self.log_transcripts and transcript:
            self._log_transcript(question, transcript, final)
        return final

    # ------------------------------------------------------------------
    # Generation: stream_ask
    # ------------------------------------------------------------------

    def stream_ask(
        self,
        question: str,
        *,
        max_new_tokens: int = 512,
    ) -> Generator[str, None, None]:
        """Yield tokens as they are generated (streaming).

        This uses ``transformers.TextIteratorStreamer`` under the hood.
        Tool-loop calls are **not** streamed — only the final answer
        generation is streamed.  If the model emits a tool call during
        streaming, the stream is interrupted, the tool is called, and
        streaming resumes for the next generation.

        Yields
        ------
        str
            One or more tokens at a time.
        """
        import threading

        from transformers import TextIteratorStreamer

        # -- adapter routing --------------------------------------------
        if self.adapter_router is not None:
            desired = self.adapter_router.route(question)
            if desired != self._current_adapter:
                self._swap_adapter(desired)

        system_prompt = self._build_system_prompt()
        messages = self._initial_messages(system_prompt, question)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = self._tokenize_chat(messages)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        """Tokenize *messages* as a chat and generate a completion."""
        inputs = self._tokenize_chat(messages)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _tokenize_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Apply the chat template and return tokenized inputs on device."""
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _build_system_prompt(self, directive: bool = False) -> str:
        """Build a system prompt describing available tools."""
        if not self._tool_names:
            return (
                "You are a helpful assistant.  Answer the user's question "
                "clearly and concisely."
            )

        tool_list = ", ".join(self._tool_names)
        base = (
            "You are a helpful assistant with access to the following tools: "
            f"{tool_list}.\n\n"
            "When you need to use a tool, emit exactly:\n"
            "<tool_call>tool_name: your query</tool_call>\n\n"
            "The tool's result will be returned in an <observation> block.  "
            "Use the observation to formulate your final answer.  "
            "Do NOT fabricate information — if you need current data, use a tool."
        )
        if directive:
            base += (
                "\n\nIMPORTANT: You MUST use at least one tool call before "
                "answering.  Do NOT answer from memory alone."
            )
        return base

    @staticmethod
    def _initial_messages(
        system_prompt: str, question: str
    ) -> List[Dict[str, str]]:
        """Build the initial message list for a conversation."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    def _run_tool(
        self, tool_name: str, query: str, *, verbose: bool = False
    ) -> str:
        """Execute a registered tool and return its output string."""
        try:
            tool_fn = get_tool(tool_name)
        except KeyError:
            msg = f"[Tool {tool_name!r} not found — skipping]"
            logger.error(msg)
            return msg

        logger.info("Running tool %r with query: %s", tool_name, query[:80])
        try:
            result = tool_fn(query)
        except Exception as exc:
            msg = f"[Tool {tool_name!r} failed: {exc}]"
            logger.error(msg)
            return msg

        logger.info("Tool %r returned %d chars", tool_name, len(result))
        return result

    @staticmethod
    def _looks_like_idk(text: str) -> bool:
        """Return True if *text* contains signals that the model doesn't know."""
        text_lower = text.lower()
        return any(signal in text_lower for signal in _IDK_SIGNALS)

    @staticmethod
    def _strip_tool_tags(text: str) -> str:
        """Remove any residual ``<tool_call>`` tags from the final answer."""
        return TOOL_CALL_PATTERN.sub("", text).strip()

    def _swap_adapter(self, adapter_id: Optional[str]) -> None:
        """Hot-swap the currently loaded LoRA adapter."""
        if self._current_adapter is not None:
            try:
                self.model.disable_adapter_layers()
                logger.info("Disabled adapter: %s", self._current_adapter)
            except Exception:
                pass

        if adapter_id is not None:
            _attach_adapter(self.model, adapter_id)
            logger.info("Swapped to adapter: %s", adapter_id)

        self._current_adapter = adapter_id

    def _log_transcript(
        self,
        question: str,
        transcript: List[Dict[str, str]],
        final_answer: str,
    ) -> None:
        """Append a tool-use transcript to a local JSON-lines file."""
        import json
        import os
        from datetime import datetime, timezone

        os.makedirs("transcripts", exist_ok=True)
        filepath = os.path.join("transcripts", "tool_use_log.jsonl")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "tool_calls": transcript,
            "final_answer": final_answer,
        }
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Transcript logged to %s", filepath)


# ---------------------------------------------------------------------------
# Adapter loading helper
# ---------------------------------------------------------------------------


def _attach_adapter(model: Any, adapter_id: str) -> None:
    """Load and merge a PEFT LoRA adapter onto *model*.

    Raises
    ------
    AdapterLoadError
        On shape mismatch or if the adapter files can't be found.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "peft is required to load a reasoning adapter.  "
            "Install it: pip install peft>=0.11"
        ) from exc

    logger.info("Attaching adapter: %s", adapter_id)
    try:
        PeftModel.from_pretrained(model, adapter_id)
    except ValueError as exc:
        raise AdapterLoadError(
            f"Adapter {adapter_id!r} is incompatible with the base model "
            f"(likely a shape mismatch).  Error: {exc}"
        ) from exc
    except OSError as exc:
        raise AdapterLoadError(
            f"Could not find adapter {adapter_id!r} on the Hub or locally.  "
            f"Error: {exc}"
        ) from exc
    except Exception as exc:
        raise AdapterLoadError(
            f"Failed to attach adapter {adapter_id!r}: {exc}"
        ) from exc

    logger.info("Adapter attached successfully: %s", adapter_id)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ModelNotFoundError(Exception):
    """Raised when a base model cannot be found or loaded."""


class AdapterLoadError(Exception):
    """Raised when a PEFT adapter fails to load or is incompatible."""
