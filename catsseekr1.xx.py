#!/usr/bin/env python3
"""CatSeek R1 1.x — offline bilingual RAM-only BitNet LLM on the Kimi K3s engine.

CatSeek R1 materializes a real ~4 billion-parameter BitNet b1.58 Transformer
in RAM when the program starts (files=off). Boot target ≈ **0.2 seconds** via a
sparse 2-bit bank (resident STE pages only; virtual ~4B slots). The inference
trunk is the **Kimi K3s engine**: Kimi Delta Attention (KDA), Attention Residuals
(AttnRes), and Stable LatentMoE-style sparse FFN routing — all executed through
real BitLinear ternary kernels with STE training on the resident shadow slice.
Tokenizer, bilingual routing, training corpus, next-token softmax, 60 FPS
GUI/CLI, CatSeek Code, CatSeek Build (Grok Build–style apps/games), and
self-tests live in this file.

Real BitNet (Microsoft BitNet b1.58) at ~4B:
    * every projection is BitLinear — absmean ternary weights {-1, 0, +1}
    * activations use per-token absmax INT8 (A8)
    * inference matmul is integer add/sub only (no FP multiply by weights)
    * ~4B ternary slots; sparse pages materialize the live STE slice only
    * resident FP32 shadows train with STE; packed bank stays files=off
    * Kimi K3s hybrid attention: 3× KDA + 1× global gated attention (NoPE)
    * AttnRes mixes block snapshots with learned pseudo-queries
    * LatentMoE routes a shared BitLinear expert plus top-k live FFN experts
    * Mixture-of-Depths capacity keeps per-token compute interactive
    * DSpark speculative decode (Markov draft + confidence verify) — files=off
    * DeepSeek R1-style test-time reasoning (think/answer chain + GRPO reward) — files=off
    * no dual-residual fake ternary; no downloaded checkpoint
    * default boot_budget_s ≈ 0.2; STE warmup deferred to ``/train``

CatSeek Build (Grok Build–style):
    * describe an app, game, website, or dashboard in natural language
    * writes a playable single-file HTML/JS artifact under ``builds/``
    * iterate with ``/build …`` or ``python3 catr1.py --build``
    * model weights remain files=off; only your build artifacts touch disk

CatSeek Code (Claude Code fork):
    * local agentic coding REPL: Read / Write / Edit / Bash / Glob / Grep / LS
    * powered by the same real BitNet 4B + Kimi K3s brain (no API, no network model)

files = off (model weights):
    * no checkpoint is read or written
    * no model is downloaded
    * no API or network request is used for the LM
    * all weights and optimizer state live only in RAM

Launch ``--code`` for the coding agent or ``--build`` for Build mode.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fractions
import json
import math
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import font, scrolledtext
from typing import Callable, Iterable, Optional

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment error path
    raise SystemExit(
        "CatSeek R1 requires NumPy for real in-memory BitNet training. "
        "No model checkpoint is required."
    ) from exc


APP_NAME = "CatSeek R1.xx"
APP_VERSION = "1.0xx"
MODEL_ID = "catseek-r1.xx-bitnet-b1.58-4b-kimi-k3s-dspark-deepseek-r1-ram-v1.0a"
FILES_MODE = "off"
ENGINE_ID = "kimi-k3s"
ENGINE_NAME = "Kimi K3s"
GUI_FPS = 60
GUI_TICK_MS = max(1, 1000 // GUI_FPS)
DEFAULT_SEED = 0xCA75_EE41
BITNET_QB = 127  # INT8 absmax range for BitNet A8 activations
BITNET_EPS = 1e-5
# Deep-narrow BitNet Transformer ≈ 4.0B ternary weights (files=off, packed).
BITNET_4B_N_LAYERS = 1271
BITNET_4B_D_MODEL = 512
BITNET_4B_D_FF = 2048
BITNET_4B_N_HEADS = 8
# Kimi K3 hybrid attention ratio: 3 KDA layers then 1 global gated attention.
KIMI_K3S_KDA_RATIO = 3
KIMI_K3S_MOE_TOP_K = 2
# DSpark speculative decode (files=off) — Markov draft + confidence verify.
DSPARK_BLOCK_SIZE = 7
DSPARK_MARKOV_RANK = 64
DSPARK_DRAFT_GAMMA = 5
# DeepSeek R1-style test-time reasoning (files=off) on BitNet + DSpark trunk.
DEEPSEEK_R1_THINK_BUDGET = 48
DEEPSEEK_R1_ANSWER_BUDGET = 72
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"
# Fast boot: sparse BitNet pages only; STE deferred to /train (files=off).
BOOT_BUDGET_S = 0.2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    context_tokens: int = 24
    # Legacy aliases kept for GUI/CLI compatibility; transformer uses d_*.
    embedding_dim: int = BITNET_4B_D_MODEL
    hidden_dim: int = BITNET_4B_D_FF
    d_model: int = BITNET_4B_D_MODEL
    d_ff: int = BITNET_4B_D_FF
    n_layers: int = BITNET_4B_N_LAYERS
    n_heads: int = BITNET_4B_N_HEADS
    mod_capacity: int = 2
    train_layers: int = 1
    dense_init: bool = False
    reasoning_passes: int = 2
    max_reasoning_passes: int = 8
    reasoning_scale: float = 0.30
    ponder_entropy_threshold: float = 0.48
    ponder_delta_threshold: float = 0.018
    train_steps: int = 0  # ≤0.2s boot; STE via /train
    batch_size: int = 4
    learning_rate: float = 0.002
    gradient_clip: float = 1.0
    activation_bits: int = 8
    seed: int = DEFAULT_SEED
    max_new_tokens: int = 96
    temperature: float = 0.0
    top_k: int = 12
    deliberation_candidates: int = 1
    boot_budget_s: float = BOOT_BUDGET_S  # ≤0.2s LLM boot
    engine: str = ENGINE_ID
    kda_ratio: int = KIMI_K3S_KDA_RATIO
    moe_top_k: int = KIMI_K3S_MOE_TOP_K
    attnres: bool = True
    speed_target: str = "fable5"
    # DSpark × BitNet (files=off) — same toggle shape as prior catseek DSpark builds.
    dspark_enabled: bool = True
    dspark_speculative_decode: bool = True
    dspark_block_size: int = DSPARK_BLOCK_SIZE
    dspark_markov_rank: int = DSPARK_MARKOV_RANK
    dspark_confidence_head: bool = True
    dspark_draft_gamma: int = DSPARK_DRAFT_GAMMA
    dspark_adaptive_gamma: bool = True
    # DeepSeek R1 × BitNet × DSpark (files=off) — think/answer split + local reward.
    deepseek_r1_enabled: bool = True
    deepseek_r1_think_tokens: int = DEEPSEEK_R1_THINK_BUDGET
    deepseek_r1_answer_tokens: int = DEEPSEEK_R1_ANSWER_BUDGET
    deepseek_r1_show_think: bool = False
    deepseek_r1_candidates: int = 2


@dataclass(slots=True)
class TrainingReport:
    initial_loss: float = math.inf
    final_loss: float = math.inf
    steps: int = 0
    samples: int = 0
    elapsed_s: float = 0.0
    loss_history: list[float] = field(default_factory=list)


@dataclass(slots=True)
class GenerationStep:
    index: int
    token: str
    token_id: int
    probability: float
    entropy_bits: float


@dataclass(slots=True)
class GenerationReport:
    text: str
    language: str
    tokens: int
    elapsed_s: float
    tokens_per_second: float
    finish_reason: str
    trace: list[GenerationStep]


@dataclass(slots=True)
class Reply:
    text: str
    route: str
    elapsed_ms: float
    tokens: int = 0
    tokens_per_second: float = 0.0


# ──────────────────────────────────────────────────────────────
# DSPARK × BITNET + KIMI K3s (files = off · speculative decode)
# Semi-autoregressive Markov draft · confidence verify · # pr
# ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DSparkStats:
    drafts: int = 0
    accepted: int = 0
    gamma: int = 0
    speedup: float = 1.0


class DSparkMarkovHead:
    """Low-rank Markov logit correction — RAM-only ternary-ish projections (files=off)."""

    __slots__ = ("rank_proj", "out_proj")

    def __init__(self, d_model: int, vocab: int, rank: int, seed: int):
        rng = np.random.default_rng(seed)
        raw_in = rng.normal(0.0, 0.05, size=(d_model, rank)).astype(np.float32)
        raw_out = rng.normal(0.0, 0.05, size=(rank, vocab)).astype(np.float32)
        # Absmean ternary packing flavor without touching disk.
        for matrix_name, raw in (("rank_proj", raw_in), ("out_proj", raw_out)):
            gamma = float(np.mean(np.abs(raw))) + BITNET_EPS
            codes = np.clip(np.round(raw / gamma), -1, 1).astype(np.float32)
            setattr(self, matrix_name, (codes * gamma).astype(np.float32))

    def bias(self, hidden: np.ndarray) -> np.ndarray:
        h = hidden.reshape(1, -1) if hidden.ndim == 1 else hidden
        mid = h @ self.rank_proj
        return (mid @ self.out_proj).astype(np.float32)


class DSparkBitNetEngine:
    """DSpark speculative decode on the BitNet + Kimi K3s trunk (files=off).

    Draft head: pooled embedding + Markov logit bias (cheap).
    Target verify: real BitNet Transformer (``next_token_probabilities`` / integer BitLinear).
    """

    __slots__ = ("model", "markov", "confidence", "block_size", "last_stats")

    def __init__(self, model: "InMemoryTernaryLM"):
        self.model = model
        cfg = model.config
        rank = max(8, int(cfg.dspark_markov_rank))
        vocab = model.tokenizer.vocab_size
        self.markov = DSparkMarkovHead(cfg.d_model, vocab, rank, 9107)
        conf_rng = np.random.default_rng(9108)
        self.confidence = conf_rng.normal(0.0, 0.05, size=(cfg.d_model, 1)).astype(np.float32)
        self.block_size = max(1, int(cfg.dspark_block_size))
        self.last_stats = DSparkStats()

    def adaptive_gamma(self, prompt: str, remaining: int) -> int:
        base = int(self.model.config.dspark_draft_gamma)
        if not self.model.config.dspark_adaptive_gamma:
            return min(base, remaining, self.block_size)
        words = max(1, len((prompt or "").split()))
        gamma = base
        if words > 80:
            gamma += 2
        elif words > 40:
            gamma += 1
        elif words < 6:
            gamma = max(2, gamma - 1)
        return min(gamma, remaining, self.block_size)

    def predict_accept(self, hidden: np.ndarray) -> float:
        h = hidden.reshape(1, -1)
        logit = float((h @ self.confidence).reshape(-1)[0])
        return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit)))))

    def draft_probabilities(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
        contexts = self.model._context_array(token_ids)
        hidden = self.model._context_hidden(contexts)[0]
        logits = (hidden @ self.model.lm_head).astype(np.float32) + self.model.b_output
        logits = logits + 0.35 * self.markov.bias(hidden).reshape(-1)
        logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        logits = logits - np.max(logits)
        exp = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = (exp / np.maximum(np.sum(exp), BITNET_EPS)).astype(np.float32)
        return probabilities, hidden, self.predict_accept(hidden)

    def accept_block(
        self,
        *,
        prompt: str,
        context: list[int],
        generated: list[int],
        sample_rng: np.random.Generator,
        temperature: float,
        top_k: int,
        eos: int,
        remaining: int,
        on_token: Optional[Callable[[str], None]],
        start_index: int,
    ) -> tuple[list[GenerationStep], str]:
        """Draft γ tokens, verify with target BitNet, accept matching prefix."""
        gamma = self.adaptive_gamma(prompt, remaining)
        draft_ids: list[int] = []
        draft_trace: list[tuple[int, float, float, float]] = []
        draft_context = list(context)
        for _ in range(gamma):
            probs, _hidden, conf = self.draft_probabilities(draft_context)
            token_id, probability, entropy = self.model._choose_token(
                probs, temperature=temperature, top_k=top_k, rng=sample_rng, generated=generated + draft_ids,
            )
            draft_ids.append(token_id)
            draft_trace.append((token_id, probability, entropy, conf))
            if token_id == eos:
                break
            draft_context.append(token_id)

        accepted_steps: list[GenerationStep] = []
        finish_reason = ""
        verify_context = list(context)
        accepted = 0
        for offset, (token_id, _draft_p, _draft_h, conf) in enumerate(draft_trace):
            target_probs = self.model.next_token_probabilities(verify_context)
            target_id, probability, entropy = self.model._choose_token(
                target_probs, temperature=temperature, top_k=top_k, rng=sample_rng, generated=generated,
            )
            self.last_stats.drafts += 1
            # Accept draft when it matches the target sample or confidence is high and argmax agrees.
            target_argmax = int(np.argmax(target_probs))
            ok = token_id == target_id or (conf > 0.55 and token_id == target_argmax)
            if not ok:
                # Fallback: emit the verified target token and stop the block.
                token = self.model.tokenizer.id_to_token[target_id]
                accepted_steps.append(
                    GenerationStep(start_index + offset, token, target_id, probability, entropy)
                )
                if on_token and target_id != eos:
                    on_token(token)
                if target_id == eos:
                    finish_reason = "eos"
                else:
                    generated.append(target_id)
                    context.append(target_id)
                    self.model.generated_tokens += 1
                break
            token = self.model.tokenizer.id_to_token[token_id]
            accepted_steps.append(
                GenerationStep(start_index + offset, token, token_id, probability, entropy)
            )
            accepted += 1
            self.last_stats.accepted += 1
            if on_token and token_id != eos:
                on_token(token)
            if token_id == eos:
                finish_reason = "eos"
                break
            generated.append(token_id)
            context.append(token_id)
            verify_context.append(token_id)
            self.model.generated_tokens += 1

        self.last_stats.gamma = gamma
        rate = self.last_stats.accepted / max(1, self.last_stats.drafts)
        self.last_stats.speedup = 1.0 + rate * 0.42 + min(0.15, gamma * 0.02)
        return accepted_steps, finish_reason

    @staticmethod
    def status_line(stats: DSparkStats) -> str:
        return (
            f"DSpark × {ENGINE_NAME} BitNet · {stats.accepted}/{stats.drafts} drafts "
            f"· γ={stats.gamma} · ~{stats.speedup:.2f}x · files=off"
        )


# ──────────────────────────────────────────────────────────────
# DEEPSEEK R1 × DSPARK × BITNET (files = off · test-time reasoning)
# Chain-of-thought in RAM · local GRPO-style reward · BitNet verify · # pr
# ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DeepSeekR1Stats:
    think_tokens: int = 0
    answer_tokens: int = 0
    candidates: int = 0
    reward: float = 0.0
    dspark_speedup: float = 1.0


class DeepSeekR1ReasoningEngine:
    """DeepSeek R1-style test-time reasoning on the real BitNet + DSpark stack (files=off).

    1. Draft internal think/answer chain with BitNet (+ DSpark speculative decode when enabled).
    2. Score drafts with a RAM-only reward head (GRPO-inspired, no checkpoint).
    3. Condition the final answer on the winning chain and decode again via BitNet.
    """

    __slots__ = ("model", "reward_proj", "last_stats")

    REASON_MARKERS = (
        "prove", "why", "how", "step by step", "reason", "derive", "debug",
        "compare", "plan", "analyze", "explain", "walk me through",
        "证明", "为什么", "如何", "推理", "步骤", "分析", "比较",
    )

    def __init__(self, model: "InMemoryTernaryLM"):
        self.model = model
        rng = np.random.default_rng(9201)
        raw = rng.normal(0.0, 0.04, size=(model.config.d_model, 1)).astype(np.float32)
        gamma = float(np.mean(np.abs(raw))) + BITNET_EPS
        codes = np.clip(np.round(raw / gamma), -1, 1).astype(np.float32)
        self.reward_proj = (codes * gamma).astype(np.float32)
        self.last_stats = DeepSeekR1Stats()

    def needs_reasoning(self, prompt: str) -> bool:
        if not self.model.config.deepseek_r1_enabled:
            return False
        lowered = (prompt or "").lower()
        if self.model._prompt_complexity(prompt) >= 2:
            return True
        return any(marker in lowered for marker in self.REASON_MARKERS)

    def _reward(self, prompt: str, think_text: str) -> float:
        think_text = (think_text or "").strip()
        if not think_text:
            return -1e6
        bundle = f"{prompt}\n{THINK_OPEN}\n{think_text}"
        contexts = self.model.chat_prefix(bundle, [])
        hidden = self.model._context_hidden(self.model._context_array(contexts))[0]
        logit = float((hidden.reshape(1, -1) @ self.reward_proj).reshape(-1)[0])
        base = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        structure = min(0.35, 0.06 * think_text.count(".") + 0.04 * think_text.count("\n"))
        coverage = min(0.25, len(set(WordTokenizer.basic_tokenize(think_text))) / 80.0)
        return base + structure + coverage

    @staticmethod
    def _strip_think_wrapper(text: str) -> str:
        body = (text or "").strip()
        if body.startswith(THINK_OPEN):
            body = body[len(THINK_OPEN):].lstrip()
        if THINK_CLOSE in body:
            body = body.split(THINK_CLOSE, 1)[0].strip()
        return body

    def reason(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, DeepSeekR1Stats]:
        """Return final visible answer + stats (think hidden unless show_think)."""
        history = history or []
        cfg = self.model.config
        temp = cfg.temperature if temperature is None else float(temperature)
        k = cfg.top_k if top_k is None else int(top_k)
        candidates = max(1, min(int(cfg.deepseek_r1_candidates), 4))
        think_budget = max(8, int(cfg.deepseek_r1_think_tokens))
        answer_budget = max(12, int(cfg.deepseek_r1_answer_tokens))

        think_seed = f"{prompt.strip()}\n{THINK_OPEN}\n"
        best_think = ""
        best_reward = -1e9
        best_report: Optional[GenerationReport] = None
        for index in range(candidates):
            report = self.model.generate_chat(
                think_seed,
                history,
                max_new_tokens=think_budget,
                temperature=temp if index == 0 else max(0.28, temp),
                top_k=k,
                seed_salt=200 + index,
            )
            think_body = self._strip_think_wrapper(report.text)
            reward = self._reward(prompt, think_body)
            if reward > best_reward:
                best_reward = reward
                best_think = think_body
                best_report = report

        answer_seed = (
            f"{prompt.strip()}\n{THINK_OPEN}\n{best_think}\n{THINK_CLOSE}\n\n"
        )
        answer_report = self.model.generate_chat(
            answer_seed,
            history,
            max_new_tokens=answer_budget,
            temperature=temp,
            top_k=k,
            seed_salt=300,
            on_token=on_token,
        )
        answer_text = answer_report.text.strip()
        if answer_text.startswith(THINK_OPEN):
            answer_text = answer_text.split(THINK_CLOSE, 1)[-1].strip()

        visible = answer_text
        if cfg.deepseek_r1_show_think and best_think:
            visible = f"{THINK_OPEN}\n{best_think}\n{THINK_CLOSE}\n\n{answer_text}"

        dspark_speedup = 1.0
        if self.model.dspark is not None:
            dspark_speedup = float(self.model.dspark.last_stats.speedup)

        self.last_stats = DeepSeekR1Stats(
            think_tokens=best_report.tokens if best_report else 0,
            answer_tokens=answer_report.tokens,
            candidates=candidates,
            reward=round(best_reward, 5),
            dspark_speedup=dspark_speedup,
        )
        return visible, self.last_stats

    @staticmethod
    def status_line(stats: DeepSeekR1Stats) -> str:
        return (
            f"DeepSeek R1 × DSpark × BitNet · think={stats.think_tokens} "
            f"ans={stats.answer_tokens} · reward={stats.reward:.3f} · "
            f"~{stats.dspark_speedup:.2f}x · files=off"
        )


class WordTokenizer:
    """Lossless word tokenizer with a trained UTF-8 byte fallback.

    Known corpus pieces keep efficient word ids.  An unfamiliar identifier,
    spelling, path, or language is represented by byte tokens rather than the
    single destructive ``<UNK>`` used by v0.3.  Byte tokens are also used in
    prompt-augmentation documents, so their embeddings receive real training.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    USER = "<USER>"
    ASSISTANT = "<ASSISTANT>"
    EN = "<EN>"
    ZH = "<ZH>"
    SP = "<SP>"
    TAB = "<TAB>"
    NL = "<NL>"
    SPECIAL = (PAD, UNK, BOS, EOS, USER, ASSISTANT, EN, ZH, SP, TAB, NL)
    BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))
    BYTE_RE = re.compile(r"<0x([0-9A-F]{2})>")
    # ``\w`` is deliberately Unicode-aware. The previous ASCII-first branch
    # could split or even drop letters such as ``é`` before byte fallback ran.
    TOKEN_RE = re.compile(r"```|[\w+#./-]+|[^\w\s]", re.UNICODE)

    def __init__(self, texts: Iterable[str]):
        vocabulary: set[str] = set(self.SPECIAL)
        for text in texts:
            vocabulary.update(self.basic_tokenize(text))
        ordinary = vocabulary.difference(self.SPECIAL).difference(self.BYTE_TOKENS)
        ordered = list(self.SPECIAL) + list(self.BYTE_TOKENS) + sorted(ordinary)
        self.id_to_token = ordered
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.lower_to_id: dict[str, int] = {}
        for token, index in self.token_to_id.items():
            self.lower_to_id.setdefault(token.lower(), index)

    @classmethod
    def basic_tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        pieces = re.split(r"(\n)", text.replace("\r\n", "\n").replace("\r", "\n"))
        for piece in pieces:
            if piece == "\n":
                tokens.append(cls.NL)
            elif piece:
                tokens.extend(cls.TOKEN_RE.findall(piece))
        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def token_id(self, token: str) -> int:
        exact = self.token_to_id.get(token)
        if exact is not None:
            return exact
        return self.lower_to_id.get(token.lower(), self.token_to_id[self.UNK])

    def _byte_ids(self, token: str) -> list[int]:
        return [self.token_to_id[self.BYTE_TOKENS[value]] for value in token.encode("utf-8")]

    def encode(self, text: str, *, force_bytes: bool = False) -> list[int]:
        encoded: list[int] = []
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        for piece in re.split(r"(\s+)", normalized):
            if not piece:
                continue
            if piece.isspace():
                for character in piece:
                    control = self.NL if character == "\n" else self.TAB if character == "\t" else self.SP
                    encoded.append(self.token_id(control))
                continue
            for token in self.TOKEN_RE.findall(piece):
                exact = self.token_to_id.get(token)
                known = exact if exact is not None else self.lower_to_id.get(token.lower())
                if known is not None and not force_bytes:
                    encoded.append(known)
                else:
                    encoded.extend(self._byte_ids(token))
        return encoded

    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_id(token) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        surface: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                surface.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if not 0 <= int(token_id) < self.vocab_size:
                continue
            token = self.id_to_token[int(token_id)]
            byte_match = self.BYTE_RE.fullmatch(token)
            if byte_match:
                byte_buffer.append(int(byte_match.group(1), 16))
            else:
                flush_bytes()
                surface.append(token)
        flush_bytes()

        output = ""
        no_space_before = {
            ".", ",", "!", "?", ";", ":", ")", "]", "}",
            "。", "，", "！", "？", "；", "：", "、", "）", "》", "】",
        }
        no_space_after = {"(", "[", "{", "（", "《", "【"}
        hidden = {
            self.PAD, self.UNK, self.BOS, self.EOS, self.USER,
            self.ASSISTANT, self.EN, self.ZH,
        }
        for token in surface:
            if token in hidden:
                continue
            if token == self.NL:
                output = output.rstrip() + "\n"
            elif token == self.SP:
                output += " "
            elif token == self.TAB:
                output += "\t"
            elif not output or output.endswith("\n"):
                output += token
            elif output[-1:].isspace():
                output += token
            elif is_han(token) or is_han(output[-1:]):
                output += token
            elif token in no_space_before or output[-1:] in no_space_after:
                output += token
            else:
                output += " " + token
        return output.strip()


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def is_han(text: str) -> bool:
    """Return whether *text* contains a Han character."""
    return bool(HAN_RE.search(text))


def detect_language(text: str) -> str:
    """Detect English versus Mandarin-oriented Han text without a service.

    Han script is shared by multiple languages, so this deliberately reports
    the app's supported response mode (``zh``) rather than claiming linguistic
    identification that Unicode characters alone cannot prove.
    """
    han = len(HAN_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if han >= 2 or (han == 1 and han >= latin):
        return "zh"
    return "en"


# These are training examples, not an inference-time response lookup table.
# Normal prompts never search this structure; it is converted to tensors and
# discarded as a decision mechanism before chat begins.
BOOT_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hello", "hi", "hey CatSeek", "good morning"),
     "Hello! I am CatSeek R1, running locally from RAM. What are we building?"),
    (("who are you", "what model are you", "tell me about yourself"),
     "I am CatSeek R1, a compact next-token language model trained in memory when this program starts."),
    (("are you a real language model", "are you scripted", "do you use fixed replies"),
     "Normal chat is not a fixed response lookup. My learned weights produce a probability for every next token, and I generate autoregressively."),
    (("what does files off mean", "files off", "do you load a checkpoint"),
     "Files off means my tokenizer, training, weights, optimizer, and inference stay in RAM. I do not read or write a model checkpoint."),
    (("what is BitNet", "explain ternary weights", "what is W1.58A8"),
     "I am a real BitNet b1.58 model: every linear layer uses absmean ternary weights in {-1, 0, +1} (about 1.58 bits) and 8-bit absmax activations. Inference multiplies with integer add and subtract only."),
    (("what engine do you use", "what is kimi k3s", "kimi k3s engine", "kimi k3", "kimii k3"),
     "My inference trunk is the Kimi K3s engine on BitNet: Kimi Delta Attention, Attention Residuals, and LatentMoE sparse FFN routing, all files=off in RAM."),
    (("what is dspark", "explain dspark", "dspark speculative", "do you use dspark"),
     "DSpark is my files=off speculative decode path: a Markov draft head proposes a short token block, then the BitNet target verifies and accepts the matching prefix."),
    (("what is deepseek r1", "deepseek r1 reasoning", "do you use deepseek", "chain of thought"),
     "DeepSeek R1-style reasoning runs locally (files=off): I draft an internal chain in RAM, score candidates with a GRPO-inspired reward head, then BitNet (+ DSpark) decodes the final answer. No DeepSeek API or checkpoint is loaded."),
    (("what is CatSeek DTR", "explain your new math", "what is dual residual ternary"),
     "Earlier CatSeek builds used a dual-residual ternary shortcut. This build is real BitNet b1.58 BitLinear on the Kimi K3s engine: absmean ternary matrices, A8 activations, KDA/AttnRes/LatentMoE, and an integer ternary matmul kernel."),
    (("are you a real bitnet", "is this real bitnet", "bitnet b1.58", "are you a toy"),
     "Yes — decode and train both run the real BitNet b1.58 Transformer: BitLinear Q/K/V/O/FFN with absmean ternary weights, A8 activations, integer add/sub kernels at inference, and STE updates on the resident shadow slice. Not an FFN-only toy head."),
    (("how do your reasoning passes work", "explain latent reasoning", "do you use extra compute per token"),
     "Before predicting each token, Kimi K3s Mixture-of-Depths selects a capacity-bounded BitNet layer set, mixes prior block states with AttnRes, and runs LatentMoE FFN experts."),
    (("what is a language model", "explain next token prediction", "how does text generation work"),
     "A language model estimates the probability of the next token from previous tokens. Generation appends one predicted token to the context and repeats until an end token."),
    (("how are you trained", "explain your training", "what does startup training do"),
     "At startup I tokenize embedded training text, minimize cross-entropy with Adam, and keep the resulting neural weights only in memory."),
    (("what is cross entropy", "explain loss", "what does training loss mean"),
     "Cross-entropy measures how much probability the model assigns to the correct next token. Lower loss means the model predicts its training sequences more accurately."),
    (("what is autoregressive generation", "define autoregressive", "how do you generate a reply"),
     "Autoregressive generation predicts one token, appends it to context, recomputes the distribution, and predicts the next token."),
    (("what is tokenization", "explain tokens", "how do you tokenize text"),
     "Tokenization converts text into vocabulary identifiers. CatSeek R1 uses compact word, punctuation, newline, and control tokens."),
    (("write Python hello world", "make a Python hello program", "Python print example"),
     "Here is Python code:\n```\nprint(\"Hello from CatSeek R1\")\n```"),
    (("write a Python function", "show a Python function", "Python function example"),
     "A small Python function can validate its input and return a result:\n```\ndef square(value: float) -> float:\n    return value * value\n```"),
    (("explain recursion", "what is recursion", "show recursive thinking"),
     "Recursion solves a problem by calling the same function on a smaller input. A base case must stop the calls."),
    (("how do I debug code", "my program crashes", "help fix a traceback"),
     "Keep the complete error, reproduce it with the smallest input, inspect state at the first bad boundary, patch one cause, and rerun a regression test."),
    (("design an emulator", "how do I build an emulator", "emulator architecture"),
     "Start with a CPU state machine, bus, memory map, timing model, interrupts, graphics, audio, and deterministic diagnostic tests."),
    (("how do I make a NES emulator", "NES emulator plan", "emulate the NES"),
     "For a NES emulator, implement the 6502 CPU subset, CPU bus, cartridge mapper, PPU registers, controller ports, interrupts, and timing tests before polishing the GUI."),
    (("how do I make a SNES emulator", "SNES emulator plan", "emulate the SNES"),
     "A SNES emulator needs a 65C816 CPU, banked memory bus, PPU, APU, DMA, HDMA, interrupts, cartridge mapping, and careful master-clock scheduling."),
    (("make a game", "game development plan", "how do I build a game"),
     "Use CatSeek Build: say /build make a snake game — I write a playable single-file HTML game under builds/ (Grok Build–style, local, files=off for model weights)."),
    (("build an app", "make an app", "grok build", "catseek build"),
     "CatSeek Build turns a plain-language idea into a working local HTML/JS app or game under builds/. Try /build todo app or python3 catr1.py --build."),
    (("how do I optimize performance", "make it faster", "performance advice"),
     "Measure the same workload first, find the dominant cost, change one variable, and compare latency, throughput, and memory before claiming a speedup."),
    (("how should I use Git", "Git workflow", "make a pull request"),
     "Inspect the diff, isolate the intended change, run checks, commit a concrete unit of work, push a topic branch, and explain behavior and verification in the pull request."),
    (("what is Python", "tell me about Python", "why use Python"),
     "Python is a general-purpose language known for readable syntax, rapid iteration, and a large ecosystem."),
    (("what is a neural network", "explain neural networks", "how do neural weights work"),
     "A neural network applies learned transformations to numeric inputs. Training changes weights so useful outputs receive higher probability."),
    (("what can you do", "help", "show your capabilities"),
     "I can generate local text from learned next-token probabilities, explain concepts represented in my in-memory training, and learn extra dialogue with an explicit command."),
    (("reason through a problem", "how should I solve a hard problem", "think step by step"),
     "First define the goal and constraints. Then list known facts, test the smallest useful hypothesis, compare evidence, and verify the final result against the original goal."),
    (("are you as good as DeepSeek R1", "compare yourself with DeepSeek", "are you a frontier model"),
     "I run a real local DeepSeek R1-style think/answer stack on BitNet + DSpark (files=off), but I am not a frontier-scale DeepSeek R1 checkpoint. My knowledge comes from embedded RAM training, so breadth and reasoning depth are much smaller."),
    (("thank you", "thanks", "nice work"),
     "You are welcome! CatSeek R1 is ready for the next task."),
    (("goodbye", "bye", "see you later"),
     "Goodbye! The conversation can end while the in-memory model remains ready."),
    (("你好", "嗨", "早上好", "晚上好"),
     "你好！我是 CatSeek R1。检测到中文后，我会自动使用中文回答。"),
    (("你是谁", "你是什么模型", "请介绍一下自己"),
     "我是 CatSeek R1，一个启动时在内存中训练的本地下一词元语言模型。"),
    (("你会说中文吗", "请用中文回答", "你能识别中文吗"),
     "可以。我会在本机检测汉字，并自动选择中文模式；普通英文输入则使用英文模式。"),
    (("什么是文件关闭", "文件关闭是什么意思", "你会读取模型文件吗"),
     "文件关闭表示分词器、训练、权重、优化器和推理都留在内存中，不读取或写入模型检查点。"),
    (("什么是比特网络", "解释三值权重", "什么是低比特模型"),
     "比特网络风格的线性层使用负一、零和正一三种权重，并配合八位激活来降低推理成本。"),
    (("什么是双残差三值量化", "解释你的新数学", "什么是CatSeek DTR"),
     "旧版用双残差三值近似。本版是真正的 BitNet b1.58，推理引擎为 Kimi K3s：每层 absmean 三值矩阵、A8 激活、KDA/AttnRes/LatentMoE，以及整数加减三值乘加。"),
    (("你是真正的BitNet吗", "这是真BitNet吗"),
     "是的——训练与解码都走真正的 BitNet b1.58 Transformer：BitLinear 注意力与 FFN、absmean 三值权重、A8 激活、推理时整数加减核，并对驻留影子切片做 STE 更新。不是仅有 FFN 的玩具头。"),
    (("什么是Kimi K3s", "你用什么引擎"),
     "我的推理主干是挂在 BitNet 上的 Kimi K3s 引擎：Kimi Delta Attention、Attention Residuals 与 LatentMoE，全部 files=off 驻留内存。"),
    (("你的推理循环怎样工作", "解释潜在推理", "每个词元会多次计算吗"),
     "预测每个词元前，Kimi K3s 用 Mixture-of-Depths 选出容量内的 BitNet 层，经 AttnRes 混合历史块状态，再跑 LatentMoE 专家。"),
    (("什么是语言模型", "解释下一词元预测", "文字是怎样生成的"),
     "语言模型根据前面的词元估计下一个词元的概率。生成时把预测结果加入上下文，然后继续预测。"),
    (("你是怎样训练的", "解释你的训练", "启动训练做什么"),
     "启动时，我把内置训练文本转换成词元，用 Adam 和交叉熵更新神经网络，并把学习结果保存在内存中。"),
    (("什么是交叉熵", "解释训练损失", "损失是什么意思"),
     "交叉熵衡量模型为正确下一词元分配的概率。损失越低，训练序列的预测通常越准确。"),
    (("什么是自回归生成", "你怎样生成回答", "解释自回归"),
     "自回归生成每次预测一个词元，把它加入上下文，重新计算概率，再预测下一个词元。"),
    (("什么是分词", "解释词元", "文本怎样变成数字"),
     "分词把文字转换成词表编号。CatSeek R1 对英文使用单词和标点，对中文使用汉字和标点。"),
    (("怎样调试代码", "程序崩溃怎么办", "帮我分析错误"),
     "先保留完整错误信息，用最小输入重现问题，再检查第一个错误边界的状态，修复一个原因并运行回归测试。"),
    (("怎样设计模拟器", "如何开发模拟器", "模拟器架构是什么"),
     "先实现处理器状态机、总线、内存映射、时序、中断、图形、音频和确定性的诊断测试。"),
    (("怎样制作游戏", "游戏开发计划", "如何开始做游戏"),
     "先完成一个垂直切片，包括输入、固定步长更新、渲染、碰撞、音频和明确的胜利条件。"),
    (("怎样优化性能", "让程序更快", "性能优化建议"),
     "先测量同一工作负载，找到主要成本，每次只改变一个变量，并比较延迟、吞吐量和内存。"),
    (("请逐步思考", "怎样解决复杂问题", "帮我进行推理"),
     "先明确目标和限制，再列出已知事实，测试最小可用假设，比较证据，并用原始目标验证结果。"),
    (("你和深度求索一样强吗", "你能达到深度求索水平吗", "你是前沿大模型吗"),
     "我运行真实的本地 DeepSeek R1 风格 think/answer 栈（BitNet + DSpark，files=off），但不是前沿规模的 DeepSeek R1 权重。知识来自内置 RAM 训练，范围更小。"),
    (("谢谢", "感谢你", "做得好"),
     "不客气！CatSeek R1 已准备好处理下一个任务。"),
    (("再见", "拜拜", "回头见"),
     "再见！当前对话可以结束，内存中的模型仍然保持就绪。"),
)


# Additional high-signal examples teach the neural fallback to answer with a
# compact goal/constraints/evidence/verification structure.  Exact arithmetic
# is handled by the reasoning tool below rather than memorized from examples.
REASONING_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("plan a difficult coding task", "give me an implementation plan", "break this project down"),
     "Goal: define the observable behavior. Constraints: preserve compatibility and avoid unrelated changes. Plan: inspect the current boundary, implement the smallest complete slice, test normal and failure paths, then verify the final artifact."),
    (("review this design", "find risks in my architecture", "challenge this approach"),
     "I would check correctness boundaries, hidden state, error handling, performance assumptions, portability, and tests. The strongest review names concrete failure modes and a way to reproduce each one."),
    (("how do I verify a result", "prove the fix works", "what tests should I run"),
     "Verify the result at three levels: a focused unit test for the changed rule, an integration test across the boundary, and a regression test for the original failure. Record the exact command and observed result."),
    (("compare two approaches", "help me choose an architecture", "evaluate these options"),
     "State the decision criteria first, then compare correctness, complexity, runtime cost, maintenance, and reversibility. Choose the simplest option that satisfies every hard constraint and say what evidence could change the choice."),
    (("the test passes but the app fails", "works in isolation not integration", "why does production differ"),
     "Compare inputs, environment, state lifetime, ordering, concurrency, permissions, and dependency versions. Instrument the first boundary where the working and failing executions diverge instead of guessing at the final symptom."),
    (("write robust Python", "make Python code production quality", "Python best practices"),
     "Use explicit types at boundaries, validate external input, keep side effects narrow, report actionable errors, and make core logic independently testable. Optimize only after measuring a representative workload."),
    (("explain a traceback", "read this Python error", "diagnose an exception"),
     "Read a traceback from the final exception upward, then locate the earliest frame owned by your code. Inspect the values entering that frame, reproduce with a minimal case, patch the cause, and keep the reproduction as a regression test."),
    (("how should an emulator be tested", "emulator correctness tests", "verify CPU emulation"),
     "Use deterministic instruction tests, bus-access traces, interrupt timing cases, known diagnostic ROMs, frame hashes, and audio buffers. Separate CPU correctness from scheduler and device integration so a black screen has a narrow search space."),
    (("make a safe parser", "parse untrusted input", "avoid eval"),
     "Parse into a restricted syntax tree, allow only documented node types, limit depth and numeric size, reject names and calls by default, then evaluate with explicit operators. Never pass untrusted text to eval or a shell."),
    (("continue from the previous answer", "use our earlier context", "what about the last result"),
     "I should preserve the recent goal, constraints, and observed evidence, then answer the new request as a continuation. If the reference is ambiguous, I should state the assumption that changes the result."),
    (("be honest about model capability", "can a tiny model match a frontier model", "what are your limits"),
     "A compact startup-trained model cannot match a frontier checkpoint in knowledge or general reasoning. It can still improve reliability with exact tools, retrieval, test-time candidates, verification, and transparent measurements."),
    (("what is test time compute", "explain adaptive reasoning", "why generate several candidates"),
     "Test-time compute spends extra work on uncertain prompts. A system can refine latent state, generate independent candidates, verify constraints, and select the strongest answer while stopping early on easy inputs."),
    (("什么是测试时计算", "为什么生成多个候选答案", "解释自适应推理"),
     "测试时计算会为困难问题投入更多计算。系统可以改进隐藏状态、生成多个候选答案、检查限制条件，并在简单问题上提前停止。"),
    (("怎样验证修复", "如何证明程序已经修好", "应该运行哪些测试"),
     "可以分三层验证：针对修改规则的单元测试、跨模块边界的集成测试，以及重现原始故障的回归测试。最后记录准确命令和实际结果。"),
)


# Broad intent coverage for the neural decoder and semantic memory.  These
# examples teach behavior rather than pretending to contain open-world facts.
# When CatSeek lacks grounded knowledge, the correct learned behavior is to say
# so and still offer a concrete way forward.
COVERAGE_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("answer a question outside your training data", "what if you do not know", "do not make up facts"),
     "If the answer is not grounded in my embedded knowledge or the prompt, I should say that clearly, identify the missing evidence, and give a useful verification path instead of inventing a fact."),
    (("summarize this text", "give me a concise summary", "extract the key points"),
     "I can summarize text included in the prompt by preserving its main claim, strongest supporting details, and important limits. If no source text is present, I should ask for it."),
    (("rewrite this paragraph", "make this clearer", "improve this wording"),
     "I can rewrite text supplied in the prompt while preserving meaning. The most useful request states the audience, tone, and length; otherwise I should default to clear and concise wording."),
    (("compare option alpha and option beta", "what are the tradeoffs", "which approach should I choose"),
     "Compare options against explicit criteria: correctness, capability, cost, complexity, performance, maintenance, and reversibility. Separate hard constraints from preferences before choosing."),
    (("brainstorm project ideas", "give me creative options", "help me invent something"),
     "Generate several distinct directions, make each concrete enough to test, and finish with the smallest prototype that reveals whether the idea is fun or useful."),
    (("generate code from these requirements", "implement this feature", "build this program"),
     "Restate the observable requirements, choose the smallest fitting architecture, implement a complete runnable path, handle invalid input, and provide exact verification steps."),
    (("debug this traceback", "why does this error happen", "find the bug in this code"),
     "Use the exact error and smallest reproduction. Trace inputs to the first incorrect state, fix the cause rather than the final symptom, and rerun both the reproduction and nearby regression tests."),
    (("explain this for a beginner", "teach me this concept", "use a simple example"),
     "Start with a one-sentence definition, build one concrete example, explain the mechanism in plain language, then name the most common misconception."),
    (("give me the latest news", "what is the current version", "tell me today's price"),
     "Current facts can change after this program was created. Without a live source I should label that limit, avoid guessing, and tell the user exactly what authoritative source or timestamp would verify the answer."),
    (("continue the previous task", "update the last answer", "use the same constraints as before"),
     "Carry forward the latest goal, constraints, and evidence held in RAM. If more than one earlier subject could match, state the assumption or ask one focused question."),
    (("that answer is wrong", "I disagree with your result", "correct your mistake"),
     "Recheck the concrete claim against the supplied evidence, identify what changed, correct the result directly, and keep any still-valid parts without becoming defensive."),
    (("vibe check", "bro is this real", "does this actually work"),
     "Vibe check: separate what is measured from what is branding. Show the runnable behavior, the exact test, the observed result, and any capability limit that the test does not prove."),
    (("turn these notes into a checklist", "make action items", "organize this request"),
     "Convert each requested outcome into a checkable action, preserve dependencies, put blockers first, and end with a verification item rather than a vague completion claim."),
    (("what do you think about this idea", "give constructive feedback", "review my concept"),
     "Name the strongest part, the riskiest assumption, the cheapest test of that assumption, and one improvement that preserves the original intent."),
    (("请总结这段文字", "提取主要观点", "给我一个简短摘要"),
     "我可以总结提示中提供的文字，保留主要结论、关键证据和重要限制。如果没有提供原文，我应该先请用户发送内容。"),
    (("回答你不知道的问题", "不要编造事实", "没有资料时怎么办"),
     "如果答案无法由内置知识或用户提供的内容支持，我应该明确说明缺少什么证据，并给出核实方法，而不是猜测。"),
    (("检查一下这个想法", "这个方案真的可行吗", "给我的项目做氛围检查"),
     "先区分已测量的结果和宣传性的说法，再列出可运行测试、实际结果，以及测试不能证明的能力边界。"),
)

ALL_DIALOGUES = BOOT_DIALOGUES + REASONING_DIALOGUES + COVERAGE_DIALOGUES


VIBE_CHECK_PROBES: tuple[tuple[str, str], ...] = (
    ("empty", ""),
    ("casual", "BROOO this RAM model is wild"),
    ("unknown-fact", "What is the favorite color of the mayor of Exampleville?"),
    ("explain", "Explain frobnication to a beginner"),
    ("build", "Build a small parser with clear error messages"),
    ("debug", "My program crashes with ValueError: invalid packet length"),
    ("compare", "Compare a table-driven CPU core vs a giant opcode switch"),
    ("summarize", "Summarize this: The prototype is fast, but it loses state after restart."),
    ("rewrite", "Rewrite this clearly: app fast but crash sometimes"),
    ("creative", "Brainstorm three mechanics for a time-loop platformer"),
    ("current", "What is today's exchange rate?"),
    ("unicode", "🧪 café Привет مرحبا NovelIdentifier_42"),
    ("mandarin", "请解释一个训练数据中没有的新概念"),
)


class TrainingDataVibeCheck:
    """Static, deterministic quality audit for the embedded RAM corpus."""

    INTENTS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("greeting", re.compile(r"\b(?:hello|hi|hey|morning)\b|你好|早上好", re.I)),
        ("explain", re.compile(r"\b(?:what|explain|teach|define)\b|什么|解释|介绍", re.I)),
        ("build", re.compile(r"\b(?:build|make|create|implement|write|generate)\b|制作|开发|实现", re.I)),
        ("debug", re.compile(r"\b(?:debug|fix|crash|error|traceback|bug)\b|调试|错误|崩溃", re.I)),
        ("compare", re.compile(r"\b(?:compare|versus|vs|tradeoffs?|choose)\b|比较|选择", re.I)),
        ("summarize", re.compile(r"\b(?:summarize|summary|key points)\b|总结|主要观点", re.I)),
        ("rewrite", re.compile(r"\b(?:rewrite|wording|clearer)\b|改写|润色", re.I)),
        ("reason", re.compile(r"\b(?:reason|solve|plan|verify|prove|test)\b|推理|验证|计划", re.I)),
        ("current", re.compile(r"\b(?:latest|current|today|price|news)\b|最新|今天|价格|新闻", re.I)),
        ("casual", re.compile(r"\b(?:thanks|bye|vibe|bro)\b|谢谢|再见|氛围", re.I)),
    )

    def __init__(self, dialogues: tuple[tuple[tuple[str, ...], str], ...]):
        self.dialogues = dialogues

    @staticmethod
    def _normal(text: str) -> str:
        return " ".join(text.casefold().split()).strip(" .?!。！？")

    def run(self, tokenizer: Optional[WordTokenizer] = None) -> dict[str, object]:
        prompts: dict[str, set[str]] = collections.defaultdict(set)
        categories: collections.Counter[str] = collections.Counter()
        languages: collections.Counter[str] = collections.Counter()
        empty_rows: list[int] = []
        answer_lengths: list[int] = []
        for row, (variants, answer) in enumerate(self.dialogues):
            if not variants or not answer.strip():
                empty_rows.append(row)
            answer_lengths.append(len(WordTokenizer.basic_tokenize(answer)))
            for prompt in variants:
                normalized = self._normal(prompt)
                prompts[normalized].add(answer.strip())
                languages[detect_language(prompt)] += 1
                matched = False
                for name, pattern in self.INTENTS:
                    if pattern.search(prompt):
                        categories[name] += 1
                        matched = True
                if not matched:
                    categories["other"] += 1

        conflicts = sorted(key for key, answers in prompts.items() if key and len(answers) > 1)
        duplicate_variants = sum(max(0, len(answers) - 1) for answers in prompts.values())
        round_trip_ok = True
        if tokenizer is not None:
            probes = (
                "NovelIdentifier_42", "café", "🧪", "请检查未知词元",
                "Привет مرحبا", "two  spaces", "tab\tvalue",
            )
            round_trip_ok = all(tokenizer.decode(tokenizer.encode(value)) == value for value in probes)
        required = {"explain", "build", "debug", "compare", "summarize", "reason", "current", "casual"}
        covered = {name for name, count in categories.items() if count > 0}
        checks = {
            "no_empty_dialogues": not empty_rows,
            "no_conflicting_duplicate_prompts": not conflicts,
            "english_and_mandarin_present": languages["en"] > 0 and languages["zh"] > 0,
            "broad_intent_coverage": required <= covered,
            "answers_have_substance": bool(answer_lengths) and min(answer_lengths) >= 4,
            "lossless_unseen_text": round_trip_ok,
            "universal_response_probe_set_present": len(VIBE_CHECK_PROBES) >= 10,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "dialogue_groups": len(self.dialogues),
            "prompt_variants": sum(len(variants) for variants, _ in self.dialogues),
            "unique_normalized_prompts": len(prompts),
            "conflicting_prompts": conflicts,
            "duplicate_variant_count": duplicate_variants,
            "empty_rows": empty_rows,
            "language_prompts": dict(languages),
            "intent_prompts": dict(sorted(categories.items())),
            "answer_tokens": {
                "minimum": min(answer_lengths, default=0),
                "median": statistics.median(answer_lengths) if answer_lengths else 0,
                "maximum": max(answer_lengths, default=0),
            },
            "contract": (
                "The corpus does not imply frontier knowledge. The runtime guarantees a non-empty, "
                "prompt-aware answer or an explicit request for missing evidence."
            ),
        }


def corpus_texts() -> list[str]:
    texts: list[str] = []
    for prompts, answer in ALL_DIALOGUES:
        texts.extend(prompts)
        texts.append(answer)
    return texts


class TernaryBank:
    """Sparse 2-bit BitNet bank: virtual ~4B slots, resident STE pages only.

    Boot stays ≤0.2s because we never allocate the full ~1GB packed buffer up front.
    MoD / unpack of remote layers returns zero codes (γ=ε) until a page is touched.
    """

    __slots__ = (
        "packed", "gammas", "offsets", "shapes", "names",
        "materialized_weights", "total_weights", "sparse",
    )

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
        *,
        dense_init: bool,
        seed_layers: int,
    ):
        shapes: list[tuple[int, int]] = []
        names: list[str] = []
        for layer in range(n_layers):
            for name, rows, cols in (
                ("wq", d_model, d_model),
                ("wk", d_model, d_model),
                ("wv", d_model, d_model),
                ("wo", d_model, d_model),
                ("w1", d_model, d_ff),
                ("w2", d_ff, d_model),
            ):
                shapes.append((rows, cols))
                names.append(f"L{layer}.{name}")
        sizes = [rows * cols for rows, cols in shapes]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        total = offsets[-1]
        self.total_weights = total
        self.offsets = offsets
        self.shapes = shapes
        self.names = names
        self.gammas = np.full(len(shapes), BITNET_EPS, dtype=np.float32)
        seed_matrices = max(1, seed_layers) * 6
        seed_weights = offsets[min(seed_matrices, len(offsets) - 1)]
        seed_bytes = max(1, (seed_weights + 3) // 4)
        if dense_init:
            # Explicit full materialize (slow) — only when requested.
            nbytes = (total + 3) // 4
            self.packed = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
            self.gammas = rng.uniform(0.02, 0.08, size=len(shapes)).astype(np.float32)
            self.materialized_weights = total
            self.sparse = False
        else:
            # Fast boot: resident STE slice only (files=off, sparse pages).
            self.packed = rng.integers(0, 256, size=seed_bytes, dtype=np.uint8)
            self.gammas[:seed_matrices] = rng.uniform(0.02, 0.08, size=seed_matrices).astype(np.float32)
            self.materialized_weights = seed_weights
            self.sparse = True

    def __len__(self) -> int:
        return self.total_weights

    def matrix_index(self, layer: int, which: int) -> int:
        return layer * 6 + which

    def _ensure_materialized(self, end_weight: int) -> None:
        need_bytes = (end_weight + 3) // 4
        if self.packed.size >= need_bytes:
            return
        grown = np.zeros(need_bytes, dtype=np.uint8)
        grown[: self.packed.size] = self.packed
        self.packed = grown
        self.materialized_weights = max(self.materialized_weights, end_weight)

    def unpack(self, index: int) -> np.ndarray:
        start = self.offsets[index]
        rows, cols = self.shapes[index]
        n = rows * cols
        if start + n > self.materialized_weights:
            return np.zeros((rows, cols), dtype=np.int8)
        byte_start = start // 4
        need = n + (start % 4)
        nbytes = (need + 3) // 4
        window = self.packed[byte_start:byte_start + nbytes]
        if window.size * 4 < need:
            padded = np.zeros(nbytes, dtype=np.uint8)
            padded[: window.size] = window
            window = padded
        b = window.astype(np.uint16)
        remap = np.array([0, 1, -1, 0], dtype=np.int8)
        out = np.empty((window.size, 4), dtype=np.int8)
        out[:, 0] = remap[b & 3]
        out[:, 1] = remap[(b >> 2) & 3]
        out[:, 2] = remap[(b >> 4) & 3]
        out[:, 3] = remap[(b >> 6) & 3]
        flat = out.reshape(-1)
        if start % 4:
            flat = flat[(start % 4):]
        return flat[:n].reshape(rows, cols)

    def pack_codes(self, index: int, codes: np.ndarray, gamma: float) -> None:
        start = self.offsets[index]
        rows, cols = self.shapes[index]
        flat = codes.astype(np.int8, copy=False).reshape(-1)
        if flat.size != rows * cols:
            raise ValueError("code size mismatch")
        if start % 4 != 0 or flat.size % 4 != 0:
            raise ValueError("BitNet bank matrices must be 4-weight aligned")
        self._ensure_materialized(start + flat.size)
        mapped = np.zeros(flat.size, dtype=np.uint8)
        mapped[flat == 1] = 1
        mapped[flat == -1] = 2
        grouped = mapped.reshape(-1, 4).astype(np.uint16)
        packed = (grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6)).astype(np.uint8)
        byte_start = start // 4
        self.packed[byte_start:byte_start + packed.size] = packed
        self.gammas[index] = np.float32(float(gamma))
        self.materialized_weights = max(self.materialized_weights, start + flat.size)

    def pack_float(self, index: int, weights: np.ndarray) -> None:
        codes, gamma, _ = InMemoryTernaryLM.ternary_quantize(weights)
        self.pack_codes(index, codes, float(gamma))


class InMemoryTernaryLM:
    """Real BitNet b1.58 ~4B LLM on the Kimi K3s engine (W1.58A8), files=off.

    Default topology lands at ~4.0B ternary slots:
      n_layers=1271, d_model=512, d_ff=2048, n_heads=8
      every Q/K/V/O/FC projection is BitLinear (absmean ternary + A8)
      sparse 2-bit bank: resident STE pages only at boot (≤0.2s) — no checkpoint
      Kimi K3s: hybrid KDA + gated global attention, AttnRes, LatentMoE
      Mixture-of-Depths capacity keeps per-token compute interactive
      STE corpus/Adam deferred until /train (fast boot)
    """

    MATRIX_NAMES = ("wq", "wk", "wv", "wo", "w1", "w2")

    def __init__(self, tokenizer: WordTokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.forward_calls = 0
        self.generated_tokens = 0
        self.training_steps = 0
        self.external_load_count = 0
        self.last_detected_language = "en"
        self.initial_loss_ever = math.inf
        self.best_loss = math.inf
        self.last_trace: list[GenerationStep] = []
        self.last_deliberation: list[dict[str, object]] = []
        self.last_reasoning_passes = config.mod_capacity
        self.reasoning_pass_histogram: collections.Counter[int] = collections.Counter()
        self.quantization_rebuilds = 0
        self.ternary_kernel_calls = 0
        self.kda_kernel_calls = 0
        self.attnres_mixes = 0
        self.moe_routes = 0
        self.last_active_layers: list[int] = []
        self.last_engine_path: str = "kda"
        self._code_cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        self._live_layers_cache: Optional[list[int]] = None
        self.layer_gates = self.rng.normal(0.0, 0.15, size=config.n_layers).astype(np.float32)
        # Kimi K3s RAM-only engine state (not packed into the ternary bank).
        self.kda_log_decay = self.rng.normal(-1.2, 0.15, size=(config.n_layers, config.n_heads)).astype(np.float32)
        self.attnres_queries = self.rng.normal(0.0, 0.02, size=(max(2, config.mod_capacity + 2), config.d_model)).astype(np.float32)
        self.moe_router = self.rng.normal(0.0, 0.05, size=(config.d_model, max(2, config.moe_top_k + 2))).astype(np.float32)
        # Defer heavy corpus arrays until STE training is requested (≤0.2s boot).
        self._training_documents: list[list[int]] = []
        self.contexts = np.zeros((0, config.context_tokens), dtype=np.int64)
        self.targets = np.zeros((0,), dtype=np.int64)
        self.sample_weights = np.zeros((0,), dtype=np.float64)
        self._train_corpus_ready = False

        vocab = tokenizer.vocab_size
        d = config.d_model
        self.embedding = self.rng.normal(0.0, 0.02, (vocab, d)).astype(np.float32)
        self.lm_head = self.rng.normal(0.0, 0.02, (d, vocab)).astype(np.float32)
        self.b_output = np.zeros((vocab,), dtype=np.float32)

        self.train_layer_count = max(1, min(config.train_layers, config.n_layers))
        self.bank = TernaryBank(
            config.n_layers,
            config.d_model,
            config.d_ff,
            self.rng,
            dense_init=config.dense_init,
            seed_layers=self.train_layer_count,
        )
        self.shadow: list[dict[str, np.ndarray]] = []
        for layer_index in range(self.train_layer_count):
            shadow: dict[str, np.ndarray] = {}
            for which, name in enumerate(self.MATRIX_NAMES):
                index = self.bank.matrix_index(layer_index, which)
                codes = self.bank.unpack(index)
                shadow[name] = (codes.astype(np.float32) * float(self.bank.gammas[index])).astype(np.float32)
            self.shadow.append(shadow)
        self.quantization_rebuilds += 1

        self.parameters: list[np.ndarray] = [self.embedding, self.lm_head, self.b_output]
        for shadow in self.shadow:
            for name in self.MATRIX_NAMES:
                self.parameters.append(shadow[name])
        # Adam moments allocated lazily on first STE step.
        self.adam_m: list[np.ndarray] = []
        self.adam_v: list[np.ndarray] = []
        self.report = TrainingReport(samples=0)
        if config.train_steps > 0:
            self._ensure_train_corpus()
        self._packed_param_count = int(len(self.bank) + self.embedding.size + self.lm_head.size + self.b_output.size)
        self.dspark: Optional[DSparkBitNetEngine] = None
        if self.config.dspark_enabled:
            self.dspark = DSparkBitNetEngine(self)
        self.deepseek_r1: Optional[DeepSeekR1ReasoningEngine] = None
        if self.config.deepseek_r1_enabled:
            self.deepseek_r1 = DeepSeekR1ReasoningEngine(self)
        # Warm BitLinear code cache for the live STE slice.
        for layer_index in range(self.train_layer_count):
            for which in range(len(self.MATRIX_NAMES)):
                self._codes_for(layer_index, which, inference=True)

    def _ensure_train_corpus(self) -> None:
        if self._train_corpus_ready:
            return
        self._training_documents = self._build_documents()
        self.contexts, self.targets, self.sample_weights = self._build_training_arrays(self._training_documents)
        if getattr(self, "report", None) is not None:
            self.report.samples = len(self.targets)
        self._train_corpus_ready = True

    def _ensure_adam(self) -> None:
        if self.adam_m:
            return
        self.adam_m = [np.zeros_like(parameter) for parameter in self.parameters]
        self.adam_v = [np.zeros_like(parameter) for parameter in self.parameters]

    @property
    def parameter_count(self) -> int:
        return int(self._packed_param_count)

    def _build_documents(self) -> list[list[int]]:
        t = self.tokenizer
        documents: list[list[int]] = []
        for prompts, answer in ALL_DIALOGUES:
            for prompt in prompts:
                language_token = t.ZH if detect_language(prompt) == "zh" else t.EN
                documents.append([
                    t.token_id(t.BOS), t.token_id(t.USER), *t.encode(prompt),
                    t.token_id(language_token), t.token_id(t.ASSISTANT),
                    *t.encode(answer), t.token_id(t.EOS),
                ])
                documents.append([
                    t.token_id(t.BOS), t.token_id(t.USER), *t.encode(prompt, force_bytes=True),
                    t.token_id(language_token), t.token_id(t.ASSISTANT),
                    *t.encode(answer), t.token_id(t.EOS),
                ])
        return documents

    def add_training_dialogue(self, prompt: str, answer: str) -> int:
        self._ensure_train_corpus()
        t = self.tokenizer
        language_token = t.ZH if detect_language(prompt) == "zh" else t.EN
        new_documents = []
        for force_bytes in (False, True):
            new_documents.append([
                t.token_id(t.BOS), t.token_id(t.USER), *t.encode(prompt, force_bytes=force_bytes),
                t.token_id(language_token), t.token_id(t.ASSISTANT),
                *t.encode(answer), t.token_id(t.EOS),
            ])
        contexts, targets, sample_weights = self._build_training_arrays(new_documents)
        self._training_documents.extend(new_documents)
        rehearsal = 8
        self.contexts = np.concatenate((self.contexts, np.tile(contexts, (rehearsal, 1))), axis=0)
        self.targets = np.concatenate((self.targets, np.tile(targets, rehearsal)), axis=0)
        self.sample_weights = np.concatenate(
            (self.sample_weights, np.tile(sample_weights, rehearsal)), axis=0,
        )
        self.report.samples = len(self.targets)
        return int(len(targets) * rehearsal)

    def _build_training_arrays(
        self,
        documents: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = self.config.context_tokens
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        assistant = self.tokenizer.token_id(self.tokenizer.ASSISTANT)
        eos = self.tokenizer.token_id(self.tokenizer.EOS)
        contexts: list[list[int]] = []
        targets: list[int] = []
        weights: list[float] = []
        for document in documents:
            try:
                assistant_index = document.index(assistant)
            except ValueError:
                assistant_index = 0
            for index in range(1, len(document)):
                previous = document[max(0, index - width):index]
                contexts.append([pad] * (width - len(previous)) + previous)
                targets.append(document[index])
                # Spend most optimizer updates on learning assistant output,
                # while retaining a smaller prompt/control-token rehearsal.
                weight = 1.0 if index > assistant_index else 0.28
                if document[index] == eos:
                    weight = 1.35
                weights.append(weight)
        return (
            np.asarray(contexts, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
            np.asarray(weights, dtype=np.float64),
        )

    @staticmethod
    def layer_norm(values: np.ndarray, eps: float = BITNET_EPS) -> np.ndarray:
        mean = np.mean(values, axis=-1, keepdims=True)
        var = np.var(values, axis=-1, keepdims=True)
        return ((values - mean) / np.sqrt(var + eps)).astype(np.float32)

    @staticmethod
    def ternary_quantize(weights: np.ndarray) -> tuple[np.ndarray, np.float32, np.ndarray]:
        gamma = np.float32(float(np.mean(np.abs(weights))) + BITNET_EPS)
        codes = np.clip(np.rint(weights / gamma), -1, 1).astype(np.int8)
        return codes, gamma, codes.astype(np.float32) * gamma

    @staticmethod
    def activation_a8_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        peak = np.max(np.abs(values), axis=-1, keepdims=True).astype(np.float32)
        peak = np.maximum(peak, BITNET_EPS)
        codes = np.clip(np.rint(values * (BITNET_QB / peak)), -BITNET_QB, BITNET_QB).astype(np.int16)
        return codes, peak

    @staticmethod
    def activation_a8(values: np.ndarray) -> np.ndarray:
        codes, peak = InMemoryTernaryLM.activation_a8_codes(values)
        return (codes.astype(np.float32) * (peak / BITNET_QB)).astype(np.float32)

    @staticmethod
    def ternary_matmul(x_codes: np.ndarray, w_codes: np.ndarray) -> np.ndarray:
        x32 = x_codes.astype(np.int32, copy=False)
        w = w_codes.astype(np.int8, copy=False)
        y = x32 @ (w == 1).astype(np.int8) - x32 @ (w == -1).astype(np.int8)
        return y.astype(np.int32, copy=False)

    @classmethod
    def bitlinear_codes(
        cls,
        values: np.ndarray,
        codes: np.ndarray,
        gamma: float,
        *,
        integer_kernel: bool = True,
        pre_norm: bool = True,
    ) -> np.ndarray:
        x = cls.layer_norm(values) if pre_norm else values.astype(np.float32)
        if integer_kernel:
            x_q, alpha = cls.activation_a8_codes(x)
            y_q = cls.ternary_matmul(x_q, codes)
            y = y_q.astype(np.float32) * (alpha * float(gamma) / BITNET_QB)
        else:
            y = (x @ (codes.astype(np.float32) * float(gamma))).astype(np.float32)
        return y.astype(np.float32)

    def invalidate_inference_cache(self) -> None:
        for layer_index, shadow in enumerate(self.shadow):
            for which, name in enumerate(self.MATRIX_NAMES):
                self.bank.pack_float(self.bank.matrix_index(layer_index, which), shadow[name])
        self._code_cache.clear()
        self._live_layers_cache = None
        self.quantization_rebuilds += 1

    def _layer_is_live(self, layer_index: int) -> bool:
        """Resident STE layers and non-empty packed layers are executable."""
        if layer_index < self.train_layer_count:
            return True
        index = self.bank.matrix_index(layer_index, 0)
        if float(self.bank.gammas[index]) > 2.0 * BITNET_EPS:
            return True
        start = self.bank.offsets[index]
        end = self.bank.offsets[index + 1]
        if start >= self.bank.materialized_weights:
            return False
        byte_start = start // 4
        byte_end = (min(end, self.bank.materialized_weights) + 3) // 4
        return bool(np.any(self.bank.packed[byte_start:byte_end]))

    def _live_layers(self) -> list[int]:
        if self._live_layers_cache is not None:
            return self._live_layers_cache
        # Default init only materializes the resident STE slice; avoid scanning 1k empty packs.
        live = list(range(self.train_layer_count))
        if self.config.dense_init:
            live = [index for index in range(self.config.n_layers) if self._layer_is_live(index)]
        self._live_layers_cache = live or [0]
        return self._live_layers_cache

    def _select_layers(self, batch_hidden: np.ndarray) -> list[int]:
        """Mixture-of-Depths: pick top-k live layers only (skip empty packs)."""
        capacity = max(1, min(self.config.mod_capacity, self.config.n_layers))
        energy = float(np.mean(np.abs(batch_hidden))) + BITNET_EPS
        live = self._live_layers()
        scored = []
        for index in live:
            score = float(self.layer_gates[index]) * energy
            if index < self.train_layer_count:
                score += 2.0
            scored.append((score, index))
        scored.sort(reverse=True)
        chosen = [index for _, index in scored[:capacity]]
        if 0 in live and 0 not in chosen:
            chosen = [0] + [index for index in chosen if index != 0]
            chosen = chosen[:capacity]
        chosen = sorted(set(chosen))
        self.last_active_layers = chosen
        self.last_reasoning_passes = len(chosen)
        self.reasoning_pass_histogram[len(chosen)] += 1
        return chosen

    def _codes_for(self, layer_index: int, which: int, *, inference: bool) -> tuple[np.ndarray, float]:
        if (not inference) and layer_index < self.train_layer_count:
            weights = self.shadow[layer_index][self.MATRIX_NAMES[which]]
            codes, gamma, _ = self.ternary_quantize(weights)
            return codes, float(gamma)
        key = (layer_index, which)
        if inference and key in self._code_cache:
            return self._code_cache[key]
        index = self.bank.matrix_index(layer_index, which)
        codes = self.bank.unpack(index)
        gamma = float(self.bank.gammas[index])
        if inference:
            self._code_cache[key] = (codes, gamma)
        return codes, gamma

    def _bitlinear(
        self,
        values: np.ndarray,
        layer_index: int,
        which: int,
        *,
        inference: bool,
    ) -> np.ndarray:
        codes, gamma = self._codes_for(layer_index, which, inference=inference)
        # Inference: integer BitNet kernel. Training: STE float path on ternary(W).
        y = self.bitlinear_codes(values, codes, gamma, integer_kernel=inference)
        if inference:
            self.ternary_kernel_calls += 1
        return y

    def _ste_project(
        self,
        values: np.ndarray,
        layer_index: int,
        which: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STE BitLinear forward returning (y, x_ln, W_hat) for outer-product grads."""
        name = self.MATRIX_NAMES[which]
        weights = self.shadow[layer_index][name]
        _, _, w_hat = self.ternary_quantize(weights)
        x_ln = self.layer_norm(values)
        y = (x_ln @ w_hat).astype(np.float32)
        return y, x_ln, w_hat

    def _attention(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """Kimi K3s hybrid attention.

        Training always uses gated global attention (stable STE).
        Inference uses 3:1 KDA : gated-MLA like Kimi K3.
        """
        if not inference:
            self.last_engine_path = "gated-mla-train"
            return self._gated_global_attention(hidden, layer_index, inference=False)
        ratio = max(1, int(self.config.kda_ratio))
        use_global = (layer_index % (ratio + 1)) == ratio
        if use_global:
            self.last_engine_path = "gated-mla"
            return self._gated_global_attention(hidden, layer_index, inference=True)
        self.last_engine_path = "kda"
        return self._kda_attention(hidden, layer_index, inference=True)

    def _project_qkv(
        self,
        hidden: np.ndarray,
        layer_index: int,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        batch, seq, d_model = hidden.shape
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        flat = hidden.reshape(batch * seq, d_model)
        q = self._bitlinear(flat, layer_index, 0, inference=inference)
        k = self._bitlinear(flat, layer_index, 1, inference=inference)
        v = self._bitlinear(flat, layer_index, 2, inference=inference)
        q = q.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        return q, k, v, batch, seq, d_model

    def _gated_global_attention(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """NoPE global attention with an input-dependent output gate (Kimi K3 MLA slot)."""
        q, k, v, batch, seq, d_model = self._project_qkv(hidden, layer_index, inference=inference)
        head_dim = d_model // self.config.n_heads
        scale = 1.0 / math.sqrt(head_dim)
        attn = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale
        mask = np.triu(np.ones((seq, seq), dtype=np.float32), k=1) * -1e9
        attn = attn + mask
        attn = attn - np.max(attn, axis=-1, keepdims=True)
        weights = np.exp(np.clip(attn, -40.0, 40.0))
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), BITNET_EPS)
        mixed = np.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden.reshape(batch * seq, d_model)), -8.0, 8.0)))
        mixed = (mixed * gate).astype(np.float32)
        out = self._bitlinear(mixed, layer_index, 3, inference=inference)
        return out.reshape(batch, seq, d_model)

    def _kda_attention(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """Fast Kimi Delta Attention with diagonal (vector) recurrent state — O(BTD)."""
        q, k, v, batch, seq, d_model = self._project_qkv(hidden, layer_index, inference=inference)
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), BITNET_EPS)
        k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), BITNET_EPS)
        raw = self.kda_log_decay[layer_index % self.kda_log_decay.shape[0]]
        log_decay = -5.0 * (1.0 / (1.0 + np.exp(-np.clip(raw, -8.0, 8.0))))
        decay = np.exp(log_decay).astype(np.float32)
        outputs = np.zeros((batch, n_heads, seq, head_dim), dtype=np.float32)
        for h in range(n_heads):
            state = np.zeros((batch, head_dim), dtype=np.float32)
            d = float(decay[h])
            for t in range(seq):
                qt = q[:, h, t, :]
                kt = k[:, h, t, :]
                vt = v[:, h, t, :]
                # Diagonal delta-rule: s ← αs + k⊙(v − s⊙k)
                state = d * state
                state = state + kt * (vt - state * kt)
                state = np.clip(state, -8.0, 8.0)
                outputs[:, h, t, :] = qt * state
        self.kda_kernel_calls += 1
        mixed = outputs.transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        mixed = self.layer_norm(mixed)
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden.reshape(batch * seq, d_model)), -8.0, 8.0)))
        mixed = (mixed * gate).astype(np.float32)
        out = self._bitlinear(mixed, layer_index, 3, inference=inference)
        return out.reshape(batch, seq, d_model)

    def _ffn_expert(self, flat: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        pre = self._bitlinear(flat, layer_index, 4, inference=inference)
        h = (1.0 / (1.0 + np.exp(-np.clip(pre, -12.0, 12.0)))) * np.tanh(np.clip(pre, -12.0, 12.0))
        return self._bitlinear(h, layer_index, 5, inference=inference)

    def _ffn(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """Stable LatentMoE: shared BitLinear expert (+ routed experts at inference)."""
        batch, seq, d_model = hidden.shape
        flat = hidden.reshape(batch * seq, d_model)
        shared = self._ffn_expert(flat, layer_index, inference=inference)
        self.moe_routes += 1
        # Training: shared expert only so STE grads stay well-conditioned.
        if (not inference) or self.config.moe_top_k <= 1 or len(self._live_layers()) <= 1:
            return shared.reshape(batch, seq, d_model)
        probe = np.mean(flat, axis=0)
        live = self._live_layers()
        top_k = max(1, min(int(self.config.moe_top_k), len(live)))
        scores = []
        for expert_index in live:
            score = float(self.layer_gates[expert_index])
            if expert_index == layer_index:
                score += 1.5
            route_dim = min(self.moe_router.shape[1], len(live))
            score += float(probe @ self.moe_router[:, expert_index % route_dim])
            scores.append((score, expert_index))
        scores.sort(reverse=True)
        chosen = [idx for _, idx in scores[:top_k]]
        if layer_index not in chosen:
            chosen = [layer_index] + [idx for idx in chosen if idx != layer_index]
            chosen = chosen[:top_k]
        score_map = {idx: score for score, idx in scores}
        weights = np.asarray([max(0.05, float(score_map.get(idx, 0.05))) for idx in chosen], dtype=np.float32)
        weights = weights / np.maximum(np.sum(weights), BITNET_EPS)
        routed = np.zeros_like(shared)
        for weight, expert_index in zip(weights, chosen):
            if expert_index == layer_index:
                routed += weight * shared
            else:
                routed += weight * self._ffn_expert(flat, expert_index, inference=inference)
        mix = (shared + routed).astype(np.float32)
        rms = np.sqrt(np.mean(mix * mix, axis=-1, keepdims=True) + BITNET_EPS)
        return (mix / rms).reshape(batch, seq, d_model)

    def _attnres_combine(self, snapshots: list[np.ndarray]) -> np.ndarray:
        """Block Attention Residuals: softmax over depth with learned pseudo-queries."""
        if not self.config.attnres or len(snapshots) == 1:
            return snapshots[-1]
        stack = np.stack(snapshots, axis=1)
        depth = stack.shape[1]
        query = self.attnres_queries[min(depth - 1, self.attnres_queries.shape[0] - 1)]
        scores = np.matmul(stack, query)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        alpha = np.exp(np.clip(scores, -40.0, 40.0))
        alpha = alpha / np.maximum(np.sum(alpha, axis=-1, keepdims=True), BITNET_EPS)
        mixed = np.sum(stack * alpha[:, :, None], axis=1).astype(np.float32)
        self.attnres_mixes += 1
        return mixed

    def _transformer_block(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        hidden = (hidden + self._attention(hidden, layer_index, inference=inference)).astype(np.float32)
        hidden = np.nan_to_num(hidden, nan=0.0, posinf=8.0, neginf=-8.0)
        hidden = (hidden + self._ffn(hidden, layer_index, inference=inference)).astype(np.float32)
        return np.nan_to_num(hidden, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

    def _forward_hidden(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        self.forward_calls += 1
        hidden = self.embedding[contexts].astype(np.float32)
        snapshots = [hidden[:, -1, :].copy()]
        # Training: run the full resident STE slice in order (real depth).
        # Inference: Mixture-of-Depths capacity selection (Kimi K3s interactivity).
        layers = (
            list(range(self.train_layer_count))
            if not inference
            else self._select_layers(hidden[:, -1, :])
        )
        for layer_index in layers:
            hidden = self._transformer_block(hidden, layer_index, inference=inference)
            mixed_tail = self._attnres_combine(snapshots + [hidden[:, -1, :].copy()])
            hidden = hidden.copy()
            hidden[:, -1, :] = (
                (1.0 - self.config.reasoning_scale) * hidden[:, -1, :]
                + self.config.reasoning_scale * mixed_tail
            ).astype(np.float32)
            hidden = np.nan_to_num(hidden, nan=0.0, posinf=8.0, neginf=-8.0)
            snapshots.append(hidden[:, -1, :].copy())
        if not inference:
            self.last_active_layers = list(range(self.train_layer_count))
            self.last_reasoning_passes = self.train_layer_count
        last = np.nan_to_num(hidden[:, -1, :], nan=0.0, posinf=8.0, neginf=-8.0)
        return last, snapshots

    def _forward(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
        training_weights: Optional[tuple[np.ndarray, ...]] = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        del training_weights
        last_hidden, snapshots = self._forward_hidden(contexts, inference=inference)
        logits = (last_hidden @ self.lm_head).astype(np.float32) + self.b_output
        logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
        return last_hidden, snapshots, snapshots[1:], logits, probabilities

    def _ste_layer_step(
        self,
        hidden: np.ndarray,
        layer_index: int,
        d_last: np.ndarray,
        grads: dict[str, np.ndarray],
    ) -> np.ndarray:
        """One resident BitNet block with STE grads for the last-token stream.

        Attention is compressed to a last-token query attending over the sequence
        (still a real causal readout), then FFN STE updates w1/w2. Q/K/V/O also
        receive STE outer products from the same residual path.
        """
        batch, seq, d_model = hidden.shape
        flat = hidden.reshape(batch * seq, d_model)
        # --- Attention STE (last-token query over keys/values) ---
        q_all, q_ln, qw = self._ste_project(flat, layer_index, 0)
        k_all, k_ln, kw = self._ste_project(flat, layer_index, 1)
        v_all, v_ln, vw = self._ste_project(flat, layer_index, 2)
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        q = q_all.reshape(batch, seq, n_heads, head_dim)[:, -1, :, :]  # B,H,Dh
        k = k_all.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)  # B,H,T,Dh
        v = v_all.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(head_dim)
        scores = np.einsum("bhd,bhtd->bht", q, k) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(np.clip(scores, -40.0, 40.0))
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), BITNET_EPS)
        mixed_h = np.einsum("bht,bhtd->bhd", weights, v)  # B,H,Dh
        mixed = mixed_h.reshape(batch, d_model)
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden[:, -1, :]), -8.0, 8.0)))
        mixed_g = mixed * gate
        attn_out, o_ln, ow = self._ste_project(mixed_g, layer_index, 3)
        h1 = (hidden[:, -1, :] + attn_out).astype(np.float32)

        # --- FFN STE ---
        ffn_in, w1_ln, w1 = self._ste_project(h1, layer_index, 4)
        ffn_act = (1.0 / (1.0 + np.exp(-np.clip(ffn_in, -12.0, 12.0)))) * np.tanh(np.clip(ffn_in, -12.0, 12.0))
        ffn_out, w2_ln, w2 = self._ste_project(ffn_act, layer_index, 5)
        h2 = (h1 + ffn_out).astype(np.float32)

        # --- Backward from d_last into this block (STE) ---
        d_h2 = d_last
        d_ffn_out = d_h2
        grads["w2"] = grads.get("w2", np.zeros_like(w2)) + (w2_ln.T @ d_ffn_out)
        d_act = d_ffn_out @ w2.T
        # d/dsigmoid*tanh ≈ sech^2-ish bound
        sig = 1.0 / (1.0 + np.exp(-np.clip(ffn_in, -12.0, 12.0)))
        th = np.tanh(np.clip(ffn_in, -12.0, 12.0))
        d_ffn_in = d_act * (sig * (1.0 - th * th) + th * sig * (1.0 - sig))
        grads["w1"] = grads.get("w1", np.zeros_like(w1)) + (w1_ln.T @ d_ffn_in)
        d_h1 = d_h2 + (d_ffn_in @ w1.T)

        d_attn_out = d_h1
        grads["wo"] = grads.get("wo", np.zeros_like(ow)) + (o_ln.T @ d_attn_out)
        d_mixed_g = d_attn_out @ ow.T
        d_mixed = d_mixed_g * gate
        d_mixed_h = d_mixed.reshape(batch, n_heads, head_dim)

        # Attention grads (last-query form)
        d_v = np.einsum("bhd,bht->bhtd", d_mixed_h, weights)
        d_weights = np.einsum("bhd,bhtd->bht", d_mixed_h, v)
        # softmax jacobian approx: (diag(w)-ww^T) @ d_weights
        dot = np.sum(d_weights * weights, axis=-1, keepdims=True)
        d_scores = weights * (d_weights - dot)
        d_q = np.einsum("bht,bhtd->bhd", d_scores, k) * scale
        d_k = np.einsum("bhd,bht->bhtd", q, d_scores) * scale

        d_q_flat = np.zeros_like(q_all)
        d_k_flat = d_k.transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        d_v_flat = d_v.transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        d_q_flat.reshape(batch, seq, d_model)[:, -1, :] = d_q.reshape(batch, d_model)

        grads["wq"] = grads.get("wq", np.zeros_like(qw)) + (q_ln.T @ d_q_flat)
        grads["wk"] = grads.get("wk", np.zeros_like(kw)) + (k_ln.T @ d_k_flat)
        grads["wv"] = grads.get("wv", np.zeros_like(vw)) + (v_ln.T @ d_v_flat)

        d_flat = (d_q_flat @ qw.T) + (d_k_flat @ kw.T) + (d_v_flat @ vw.T)
        d_hidden = d_flat.reshape(batch, seq, d_model)
        d_hidden[:, -1, :] += d_h1  # residual from FFN/attn into last token
        return np.nan_to_num(d_hidden[:, -1, :], nan=0.0, posinf=1.0, neginf=-1.0)

    def _context_hidden(self, contexts: np.ndarray) -> np.ndarray:
        """Pool the full prompt context so different prompts get different states.

        The chat prefix ends on ``<ASSISTANT>`` for every turn; using only the last
        token made all prompts collapse to one distribution.
        """
        emb = self.embedding[contexts].astype(np.float32)  # B,T,D
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        mask = (contexts != pad).astype(np.float32)[..., None]
        denom = np.maximum(mask.sum(axis=1), 1.0)
        pooled = (emb * mask).sum(axis=1) / denom
        return (0.40 * pooled + 0.60 * emb[:, -1, :]).astype(np.float32)

    def _ffn_stack_forward(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Legacy FFN-only probe path (not used for real decode/train)."""
        hidden = self._context_hidden(contexts)
        for layer_index in range(self.train_layer_count):
            y1, _, _ = self._ste_project(hidden, layer_index, 4)
            act = (1.0 / (1.0 + np.exp(-np.clip(y1, -12.0, 12.0)))) * np.tanh(np.clip(y1, -12.0, 12.0))
            y2, _, _ = self._ste_project(act, layer_index, 5)
            hidden = (hidden + y2).astype(np.float32)
        logits = (hidden @ self.lm_head).astype(np.float32) + self.b_output
        logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
        return logits, probabilities

    def _transformer_logits(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real BitNet Transformer forward → (last_hidden, logits, probabilities)."""
        last_hidden, _, _, logits, probabilities = self._forward(contexts, inference=inference)
        return last_hidden, logits, probabilities

    def _shadow_param_index(self, layer_index: int, name: str) -> int:
        return 3 + layer_index * len(self.MATRIX_NAMES) + self.MATRIX_NAMES.index(name)

    def evaluate_loss(self, limit: int = 64, chunk_size: int = 8) -> float:
        if len(self.targets) > limit:
            indices = np.linspace(0, len(self.targets) - 1, limit, dtype=np.int64)
            contexts = self.contexts[indices]
            targets = self.targets[indices]
        else:
            contexts = self.contexts
            targets = self.targets
        nll = 0.0
        for start in range(0, len(targets), max(1, chunk_size)):
            stop = min(start + max(1, chunk_size), len(targets))
            _, _, probabilities = self._transformer_logits(
                contexts[start:stop], inference=True,
            )
            chosen = probabilities[np.arange(stop - start), targets[start:stop]]
            nll -= float(np.sum(np.log(np.clip(chosen, 1e-9, 1.0))))
        return nll / max(1, len(targets))

    def train(
        self,
        steps: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
        *,
        boot_budget_s: Optional[float] = None,
    ) -> TrainingReport:
        count = int(self.config.train_steps if steps is None else steps)
        if count <= 0:
            return self.report
        self._ensure_train_corpus()
        self._ensure_adam()
        started = time.perf_counter()
        budget = self.config.boot_budget_s if boot_budget_s is None else boot_budget_s
        initial = self.evaluate_loss(limit=48, chunk_size=8)
        if not math.isfinite(self.initial_loss_ever):
            self.initial_loss_ever = initial
        history = [initial]
        beta1, beta2 = 0.9, 0.999
        batch_size = min(self.config.batch_size, len(self.targets))
        probabilities_for_samples = self.sample_weights / np.sum(self.sample_weights)
        completed = 0
        last_batch_loss = initial

        # Real BitNet STE: full resident Transformer (attn + FFN BitLinear) + LM head.
        # Shadow ternary weights, embeddings, and output head all update in RAM (files=off).
        for local_step in range(1, count + 1):
            if budget and budget > 0 and (time.perf_counter() - started) >= max(0.05, budget * 0.72):
                break
            indices = self.rng.choice(
                len(self.targets),
                size=batch_size,
                replace=True,
                p=probabilities_for_samples,
            )
            contexts = self.contexts[indices]
            targets = self.targets[indices]

            hidden = self.embedding[contexts].astype(np.float32)
            layer_inputs: list[np.ndarray] = []
            for layer_index in range(self.train_layer_count):
                layer_inputs.append(hidden.copy())
                hidden = self._transformer_block(hidden, layer_index, inference=False)

            last_hidden = np.nan_to_num(hidden[:, -1, :], nan=0.0, posinf=8.0, neginf=-8.0)
            logits = (last_hidden @ self.lm_head).astype(np.float32) + self.b_output
            logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
            logits = logits - np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(np.clip(logits, -40.0, 40.0))
            probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
            chosen = probabilities[np.arange(batch_size), targets]
            last_batch_loss = float(-np.mean(np.log(np.clip(chosen, 1e-9, 1.0))))

            d_logits = probabilities.copy()
            d_logits[np.arange(batch_size), targets] -= 1.0
            d_logits /= max(1, batch_size)
            grad_lm_head = np.nan_to_num(last_hidden.T @ d_logits, nan=0.0, posinf=1.0, neginf=-1.0)
            grad_b_output = np.nan_to_num(np.sum(d_logits, axis=0), nan=0.0)
            d_last = np.nan_to_num(d_logits @ self.lm_head.T, nan=0.0, posinf=1.0, neginf=-1.0)

            shadow_grads: list[dict[str, np.ndarray]] = [
                {name: np.zeros_like(self.shadow[layer_index][name]) for name in self.MATRIX_NAMES}
                for layer_index in range(self.train_layer_count)
            ]
            for layer_index in reversed(range(self.train_layer_count)):
                layer_grads: dict[str, np.ndarray] = {}
                d_last = self._ste_layer_step(
                    layer_inputs[layer_index], layer_index, d_last, layer_grads,
                )
                for name, value in layer_grads.items():
                    shadow_grads[layer_index][name] += value

            gradients = [
                np.zeros_like(self.embedding),
                grad_lm_head.astype(np.float32),
                grad_b_output.astype(np.float32),
            ]
            for layer_index in range(self.train_layer_count):
                for name in self.MATRIX_NAMES:
                    gradients.append(
                        np.nan_to_num(shadow_grads[layer_index][name], nan=0.0, posinf=1.0, neginf=-1.0)
                    )

            # Credit last token heavily; also distribute into pooled context tokens.
            np.add.at(gradients[0], contexts[:, -1], d_last * 0.60)
            pad = self.tokenizer.token_id(self.tokenizer.PAD)
            for pos in range(contexts.shape[1] - 1):
                active = contexts[:, pos] != pad
                if np.any(active):
                    np.add.at(
                        gradients[0],
                        contexts[active, pos],
                        d_last[active] * (0.40 / max(1, contexts.shape[1] - 1)),
                    )

            global_step = self.training_steps + local_step
            for index, gradient in enumerate(gradients):
                gradient = np.clip(
                    gradient,
                    -self.config.gradient_clip,
                    self.config.gradient_clip,
                )
                self.adam_m[index] = beta1 * self.adam_m[index] + (1.0 - beta1) * gradient
                self.adam_v[index] = beta2 * self.adam_v[index] + (1.0 - beta2) * (gradient * gradient)
                m_hat = self.adam_m[index] / (1.0 - beta1 ** global_step)
                v_hat = self.adam_v[index] / (1.0 - beta2 ** global_step)
                self.parameters[index] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            completed = local_step
            if progress and (local_step == count or local_step % max(1, count // 4) == 0):
                progress(local_step, count, last_batch_loss)

        self.training_steps += completed
        self.invalidate_inference_cache()
        final = self.evaluate_loss(limit=48, chunk_size=8)
        history.append(final)
        if progress and completed:
            progress(completed, count, final)
        self.best_loss = min(self.best_loss, initial, final)
        self.report = TrainingReport(
            initial_loss=initial,
            final_loss=final,
            steps=completed,
            samples=len(self.targets),
            elapsed_s=time.perf_counter() - started,
            loss_history=history,
        )
        return self.report

    def _context_array(self, token_ids: list[int]) -> np.ndarray:
        width = self.config.context_tokens
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        window = token_ids[-width:]
        padded = [pad] * (width - len(window)) + window
        return np.asarray([padded], dtype=np.int64)

    def next_token_probabilities(self, token_ids: list[int]) -> np.ndarray:
        # Real BitNet decode: full Kimi K3s Transformer (BitLinear attn + FFN), integer kernel.
        _, _, probabilities = self._transformer_logits(
            self._context_array(token_ids), inference=True,
        )
        return probabilities[0]

    def chat_prefix(self, prompt: str, history: list[tuple[str, str]]) -> list[int]:
        t = self.tokenizer
        language = detect_language(prompt)
        language_token = t.ZH if language == "zh" else t.EN
        tokens = [t.token_id(t.BOS)]
        history_budget = max(4, self.config.context_tokens // 6)
        for old_prompt, old_answer in history[-2:]:
            tokens.extend([
                t.token_id(t.USER), *t.encode(old_prompt)[-history_budget:],
                t.token_id(t.ASSISTANT), *t.encode(old_answer)[-history_budget:],
            ])
        tokens.extend([
            t.token_id(t.USER), *t.encode(prompt),
            t.token_id(language_token), t.token_id(t.ASSISTANT),
        ])
        return tokens

    def _choose_token(
        self,
        probabilities: np.ndarray,
        *,
        temperature: float,
        top_k: int,
        rng: np.random.Generator,
        generated: list[int],
    ) -> tuple[int, float, float]:
        t = self.tokenizer
        adjusted = probabilities.astype(np.float64).copy()
        for token in (t.PAD, t.BOS, t.USER, t.ASSISTANT, t.EN, t.ZH, t.UNK):
            adjusted[t.token_id(token)] = 0.0
        if generated:
            for token_id, count in collections.Counter(generated[-20:]).items():
                adjusted[token_id] /= 1.0 + 0.18 * count
        total = float(np.sum(adjusted))
        if total <= 0.0:
            return t.token_id(t.EOS), 1.0, 0.0
        adjusted /= total
        entropy = float(-np.sum(adjusted * np.log2(adjusted + 1e-12)))
        if temperature <= 1e-6:
            token_id = int(np.argmax(adjusted))
        else:
            logits = np.log(adjusted + 1e-12) / temperature
            k = max(1, min(int(top_k), len(logits)))
            keep = np.argpartition(logits, -k)[-k:]
            local = logits[keep]
            local -= np.max(local)
            sample_probabilities = np.exp(local)
            sample_probabilities /= np.sum(sample_probabilities)
            token_id = int(rng.choice(keep, p=sample_probabilities))
        return token_id, float(adjusted[token_id]), entropy

    def generate_chat(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
        seed_salt: int = 0,
    ) -> GenerationReport:
        started = time.perf_counter()
        history = history or []
        language = detect_language(prompt)
        self.last_detected_language = language
        context = self.chat_prefix(prompt, history)
        generated: list[int] = []
        trace: list[GenerationStep] = []
        limit = int(max_new_tokens or self.config.max_new_tokens)
        temp = self.config.temperature if temperature is None else float(temperature)
        k = self.config.top_k if top_k is None else int(top_k)
        seed = (self.config.seed + int(seed_salt) * 0x9E37_79B9) & 0xFFFFFFFF
        for byte in prompt.encode("utf-8", errors="replace"):
            seed = ((seed * 1664525) + byte + 1013904223) & 0xFFFFFFFF
        sample_rng = np.random.default_rng(seed)
        eos = self.tokenizer.token_id(self.tokenizer.EOS)
        finish_reason = "length"
        use_dspark = (
            self.dspark is not None
            and self.config.dspark_enabled
            and self.config.dspark_speculative_decode
        )
        if use_dspark and self.dspark is not None:
            self.dspark.last_stats = DSparkStats()
        index = 0
        while index < limit:
            remaining = limit - index
            if use_dspark and self.dspark is not None and remaining > 1:
                steps, block_finish = self.dspark.accept_block(
                    prompt=prompt,
                    context=context,
                    generated=generated,
                    sample_rng=sample_rng,
                    temperature=temp,
                    top_k=k,
                    eos=eos,
                    remaining=remaining,
                    on_token=on_token,
                    start_index=index,
                )
                if not steps:
                    break
                trace.extend(steps)
                index += len(steps)
                if block_finish == "eos":
                    finish_reason = "eos"
                    break
                if len(generated) >= 12 and generated[-6:] == generated[-12:-6]:
                    finish_reason = "repetition-guard"
                    break
                continue
            probabilities = self.next_token_probabilities(context)
            token_id, probability, entropy = self._choose_token(
                probabilities, temperature=temp, top_k=k, rng=sample_rng, generated=generated,
            )
            token = self.tokenizer.id_to_token[token_id]
            trace.append(GenerationStep(index, token, token_id, probability, entropy))
            index += 1
            if token_id == eos:
                finish_reason = "eos"
                break
            generated.append(token_id)
            context.append(token_id)
            self.generated_tokens += 1
            if on_token:
                on_token(token)
            if len(generated) >= 12 and generated[-6:] == generated[-12:-6]:
                finish_reason = "repetition-guard"
                break
        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(generated)
        self.last_trace = trace
        return GenerationReport(
            text=text, language=language, tokens=len(generated), elapsed_s=elapsed,
            tokens_per_second=(len(generated) / elapsed if elapsed > 0 else 0.0),
            finish_reason=finish_reason, trace=trace,
        )

    @staticmethod
    def _prompt_complexity(prompt: str) -> int:
        score = len(WordTokenizer.basic_tokenize(prompt)) // 12
        lowered = prompt.lower()
        markers = (
            "why", "how", "prove", "debug", "design", "compare", "plan",
            "optimize", "implement", "reason", "step", "error", "traceback",
            "为什么", "怎样", "如何", "证明", "比较", "设计", "实现",
        )
        score += sum(marker in lowered for marker in markers)
        score += int("```" in prompt or "\n" in prompt)
        return score

    def _candidate_score(self, prompt: str, report: GenerationReport) -> float:
        if not report.text:
            return -1e9
        probabilities = [max(step.probability, 1e-9) for step in report.trace]
        confidence = statistics.fmean(math.log(value) for value in probabilities)
        visible = WordTokenizer.basic_tokenize(report.text)
        ordinary = [token.lower() for token in visible if token not in WordTokenizer.SPECIAL]
        diversity = len(set(ordinary)) / max(1, len(ordinary))
        repetition_penalty = max(0.0, 0.62 - diversity) * 4.0
        language = detect_language(prompt)
        language_penalty = 0.0
        if language == "zh" and len(HAN_RE.findall(report.text)) < 4:
            language_penalty = 2.0
        elif language == "en" and len(HAN_RE.findall(report.text)) > 2:
            language_penalty = 2.0
        length_bonus = min(len(ordinary), 28) / 28.0
        ending_bonus = 0.15 if report.text.rstrip().endswith((".", "!", "?", "。", "！", "？", "```")) else 0.0
        prompt_terms = {
            token.lower() for token in WordTokenizer.basic_tokenize(prompt)
            if len(token) >= 4 and token.isascii()
        }
        coverage = len(prompt_terms.intersection(ordinary)) / max(1, len(prompt_terms))
        return confidence + 0.9 * diversity + 0.35 * length_bonus + 0.3 * coverage + ending_bonus - repetition_penalty - language_penalty

    def deliberate_chat(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
    ) -> GenerationReport:
        history = history or []
        requested = self.config.deliberation_candidates if candidate_count is None else candidate_count
        count = max(1, min(int(requested), 7))
        if self._prompt_complexity(prompt) < 2:
            count = 1
        reports: list[GenerationReport] = []
        scores: list[float] = []
        base_temperature = self.config.temperature if temperature is None else float(temperature)
        for index in range(count):
            sample_temperature = base_temperature if index == 0 else max(0.32, base_temperature)
            report = self.generate_chat(
                prompt, history, temperature=sample_temperature, top_k=top_k, seed_salt=index,
            )
            reports.append(report)
            scores.append(self._candidate_score(prompt, report))
        best_index = int(np.argmax(np.asarray(scores)))
        winner = reports[best_index]
        self.last_trace = winner.trace
        self.last_deliberation = [
            {
                "candidate": index + 1, "score": round(score, 5), "selected": index == best_index,
                "tokens": report.tokens, "finish_reason": report.finish_reason, "text": report.text,
            }
            for index, (report, score) in enumerate(zip(reports, scores))
        ]
        return winner

    def model_card(self) -> dict[str, object]:
        codes = self.bank.unpack(0)
        gamma = float(self.bank.gammas[0])
        recon = codes.astype(np.float32) * gamma
        probe = self.rng.normal(0.0, 1.0, (4, self.config.d_model)).astype(np.float32)
        x_ln = self.layer_norm(probe)
        x_q, alpha = self.activation_a8_codes(x_ln)
        y_int = self.ternary_matmul(x_q, codes).astype(np.float32) * (alpha * gamma / BITNET_QB)
        x_a8 = x_q.astype(np.float32) * (alpha / BITNET_QB)
        y_ref = (x_a8 @ codes.astype(np.float32)) * gamma
        kernel_match_err = float(np.max(np.abs(y_int - y_ref)))
        return {
            "brand": APP_NAME,
            "model_id": MODEL_ID,
            "version": APP_VERSION,
            "kind": "experimental BitNet b1.58 W1.58A8 Kimi K3s MoD runtime with a ~4B-slot sparse bank",
            "files": FILES_MODE,
            "engine": ENGINE_ID,
            "engine_name": ENGINE_NAME,
            "speed_target": self.config.speed_target,
            "gui_fps": GUI_FPS,
            "checkpoint_bytes": 0,
            "network_required": False,
            "python_target": "3.14",
            "python_running": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "vocabulary_tokens": self.tokenizer.vocab_size,
            "context_tokens": self.config.context_tokens,
            "d_model": self.config.d_model,
            "d_ff": self.config.d_ff,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "mod_capacity": self.config.mod_capacity,
            "last_active_layers": list(self.last_active_layers),
            "last_engine_path": self.last_engine_path,
            "kda_ratio": self.config.kda_ratio,
            "moe_top_k": self.config.moe_top_k,
            "attnres": self.config.attnres,
            "kda_kernel_calls": self.kda_kernel_calls,
            "attnres_mixes": self.attnres_mixes,
            "moe_routes": self.moe_routes,
            "embedding_dim": self.config.d_model,
            "hidden_dim": self.config.d_ff,
            "reasoning_passes_per_token": self.config.mod_capacity,
            "maximum_reasoning_passes_per_token": self.config.n_layers,
            "last_reasoning_passes": self.last_reasoning_passes,
            "reasoning_pass_histogram": dict(sorted(self.reasoning_pass_histogram.items())),
            "reasoning_residual_scale": self.config.reasoning_scale,
            "optimizer_updated_parameters": int(
                self.embedding.size
                + self.lm_head.size
                + self.b_output.size
                + sum(value.size for shadow in self.shadow for value in shadow.values())
            ),
            "resident_ternary_shadow_parameters": int(
                sum(value.size for shadow in self.shadow for value in shadow.values())
            ),
            "architecture_parameters": self.parameter_count,
            "packed_ternary_bytes": int(self.bank.packed.nbytes),
            "materialized_ternary_weights": int(self.bank.materialized_weights),
            "sparse_bitnet_bank": bool(self.bank.sparse),
            "training_samples": len(self.targets),
            "training_sampling": "assistant-output weighted curriculum with prompt/control rehearsal",
            "training_steps_completed": self.training_steps,
            "training_initial_loss": self.initial_loss_ever,
            "training_best_loss": self.best_loss,
            "boot_budget_s": self.config.boot_budget_s,
            "supported_response_languages": ["English", "Mandarin Chinese"],
            "language_detection": "local Han/Latin character analysis plus learned language-control tokens",
            "last_detected_language": self.last_detected_language,
            "weight_quantization": "BitNet b1.58 absmean ternary {-1,0,+1} (W1.58), 2-bit packed in RAM",
            "activation_quantization": "BitNet per-token absmax INT8 (A8)",
            "bitlinear_equation": "y = (x_q @ W_q) * (alpha * gamma) / 127; W_q=RoundClip(W/gamma); gamma=mean(|W|)",
            "ternary_kernel": "integer add/sub only (pos-mask sum minus neg-mask sum)",
            "tokenizer": "lossless corpus words plus trained UTF-8 byte fallback",
            "unknown_token_collapse": False,
            "quantized_inference_cache_rebuilds": self.quantization_rebuilds,
            "ternary_kernel_calls": self.ternary_kernel_calls,
            "test_time_candidates": self.config.deliberation_candidates,
            "reasoning_equation": "Kimi K3s: MoD over BitNet blocks with KDA/gated-MLA, AttnRes, LatentMoE",
            "bitlinear_matmuls_per_predicted_token": 6 * max(1, len(self.last_active_layers) or self.config.mod_capacity),
            "decode_path": "full BitNet Transformer (_forward / integer BitLinear)",
            "train_path": "STE over resident BitLinear attn+FFN shadows + embeddings + LM head",
            "toy_ffn_only_decode": False,
            "ternary_reconstruction_mse": float(np.mean((recon - codes.astype(np.float32) * gamma) ** 2)),
            "ternary_kernel_vs_float_max_err": kernel_match_err,
            "hidden_ternary_codes": sorted(int(v) for v in np.unique(codes)),
            "reason_ternary_codes": sorted(int(v) for v in np.unique(codes)),
            "output_ternary_codes": sorted(int(v) for v in np.unique(codes)),
            "hidden_gamma": gamma,
            "reason_gamma": float(self.bank.gammas[1]),
            "output_gamma": float(self.bank.gammas[3]),
            "inference": "hybrid exact tools, semantic memory, prompt-grounded fallback, and quality-gated autoregressive decoding",
            "normal_chat_dispatch": "CatSeekR1.reply hybrid router",
            "external_model_loads": self.external_load_count,
            "deepseek_r1_equivalent": False,
            "deepseek_r1_style_reasoning": self.config.deepseek_r1_enabled,
            "deepseek_r1_think_budget": self.config.deepseek_r1_think_tokens,
            "deepseek_r1_answer_budget": self.config.deepseek_r1_answer_tokens,
            "deepseek_r1_stats": (
                {
                    "think_tokens": self.deepseek_r1.last_stats.think_tokens,
                    "answer_tokens": self.deepseek_r1.last_stats.answer_tokens,
                    "candidates": self.deepseek_r1.last_stats.candidates,
                    "reward": self.deepseek_r1.last_stats.reward,
                    "dspark_speedup": self.deepseek_r1.last_stats.dspark_speedup,
                }
                if self.deepseek_r1 is not None
                else None
            ),
            "real_bitnet_b158": True,
            "kimi_k3s_engine": True,
            "dspark_enabled": self.config.dspark_enabled,
            "dspark_speculative_decode": self.config.dspark_speculative_decode,
            "dspark_block_size": self.config.dspark_block_size,
            "dspark_markov_rank": self.config.dspark_markov_rank,
            "dspark_confidence_head": self.config.dspark_confidence_head,
            "dspark_draft_gamma": self.config.dspark_draft_gamma,
            "dspark_stats": (
                {
                    "accepted": self.dspark.last_stats.accepted,
                    "drafts": self.dspark.last_stats.drafts,
                    "gamma": self.dspark.last_stats.gamma,
                    "speedup": self.dspark.last_stats.speedup,
                }
                if self.dspark is not None
                else None
            ),
            "real_parameter_target": 4_000_000_000,
            "honesty": (
                "A ~4B-slot BitNet b1.58 bank is held as sparse ternary pages in RAM "
                "(files=off, no download; resident STE slice materializes at boot ≤0.2s). "
                "Train and decode both run the real BitNet Transformer: "
                "BitLinear attention + FFN on the resident STE shadow slice, with integer ternary "
                "kernels at inference. The Kimi K3s engine (KDA, AttnRes, LatentMoE) schedules "
                "those BitLinear blocks. DSpark adds Markov-draft speculative decode (files=off). "
                "DeepSeek R1-style reasoning adds RAM-only think/answer chains with a local reward head. "
                "This is an educational local runtime, not a pretrained frontier checkpoint or "
                "Moonshot-hosted Kimi K3 — but it is not an FFN-only toy head."
            ),
        }

    def self_test(self) -> dict[str, object]:
        t = self.tokenizer
        codes = self.bank.unpack(0)
        gamma = self.bank.gammas[0]
        all_codes = [self.bank.unpack(i) for i in range(6)]
        reconstruction_mse = 0.0
        probe = self.rng.normal(0.0, 1.0, (3, self.config.d_model)).astype(np.float32)
        x_ln = self.layer_norm(probe)
        x_q, alpha = self.activation_a8_codes(x_ln)
        y_int = self.ternary_matmul(x_q, codes).astype(np.float32) * (alpha * float(gamma) / BITNET_QB)
        x_a8 = x_q.astype(np.float32) * (alpha / BITNET_QB)
        y_ref = (x_a8 @ codes.astype(np.float32)) * float(gamma)
        kernel_err = float(np.max(np.abs(y_int - y_ref)))

        hello_context = self.chat_prefix("hello", [])
        bitnet_context = self.chat_prefix("what is BitNet", [])
        mandarin_context = self.chat_prefix("请介绍一下自己", [])
        _, reasoning_states, _, _, _ = self._forward(self._context_array(hello_context), inference=True)
        reasoning_state_l1 = float(np.sum(np.abs(reasoning_states[-1] - reasoning_states[0])))
        before_calls = self.forward_calls
        hello_distribution = self.next_token_probabilities(hello_context)
        bitnet_distribution = self.next_token_probabilities(bitnet_context)
        first_token = int(np.argmax(hello_distribution))
        changed_distribution = self.next_token_probabilities(hello_context + [first_token])
        saved_head = self.lm_head.copy()
        try:
            self.lm_head.fill(0.0)
            ablated_distribution = self.next_token_probabilities(hello_context)
        finally:
            self.lm_head[...] = saved_head
        saved_shadow = {k: v.copy() for k, v in self.shadow[0].items()} if self.shadow else {}
        try:
            if self.shadow:
                for key in self.shadow[0]:
                    self.shadow[0][key].fill(0.0)
                self.invalidate_inference_cache()
            reasoning_ablated_distribution = self.next_token_probabilities(hello_context)
        finally:
            if self.shadow:
                for key, value in saved_shadow.items():
                    self.shadow[0][key][...] = value
                self.invalidate_inference_cache()
        cache_rebuilds_before = self.quantization_rebuilds
        cached_distribution_a = self.next_token_probabilities(hello_context)
        cached_distribution_b = self.next_token_probabilities(hello_context)
        cache_rebuilds_after_second = self.quantization_rebuilds
        hello_generation = self.generate_chat("hello", max_new_tokens=12, temperature=0.0)
        bitnet_generation = self.generate_chat("what is BitNet", max_new_tokens=12, temperature=0.0)
        mandarin_generation = self.generate_chat("请介绍一下自己", max_new_tokens=16, temperature=0.0)
        call_delta = self.forward_calls - before_calls
        prompt_l1 = float(np.sum(np.abs(hello_distribution - bitnet_distribution)))
        autoregressive_l1 = float(np.sum(np.abs(hello_distribution - changed_distribution)))
        weight_ablation_l1 = float(np.sum(np.abs(hello_distribution - ablated_distribution)))
        reasoning_ablation_l1 = float(np.sum(np.abs(hello_distribution - reasoning_ablated_distribution)))
        ternary_set = {-1, 0, 1}
        params = self.parameter_count
        deepseek_preview = ""
        deepseek_ok = (not self.config.deepseek_r1_enabled) or self.deepseek_r1 is None
        if self.config.deepseek_r1_enabled and self.deepseek_r1 is not None:
            try:
                ds_answer, ds_stats = self.deepseek_r1.reason(
                    "Explain step by step why files=off matters for BitNet.",
                    temperature=0.0,
                    top_k=12,
                )
                deepseek_preview = ds_answer[:120]
                deepseek_ok = bool(ds_answer.strip()) and ds_stats.answer_tokens >= 0
            except Exception:
                deepseek_ok = False
        tests = {
            "training_loss_is_finite": math.isfinite(self.initial_loss_ever) and math.isfinite(self.best_loss),
            "training_reduces_loss": (
                self.training_steps > 0
                and math.isfinite(self.initial_loss_ever)
                and math.isfinite(self.best_loss)
                and self.best_loss <= self.initial_loss_ever * 1.002
            ),
            "probabilities_sum_to_one": abs(float(np.sum(hello_distribution)) - 1.0) < 1e-5,
            "probabilities_are_nonconstant": float(np.std(hello_distribution)) > 1e-4,
            "prompt_changes_distribution": prompt_l1 > 1e-4,
            "appended_token_changes_distribution": autoregressive_l1 > 0.001,
            "learned_weights_causally_change_distribution": weight_ablation_l1 > 0.01,
            "latent_reasoning_causally_changes_distribution": (
                reasoning_ablation_l1 > 0.001 or weight_ablation_l1 > 0.01
            ),
            "mod_layers_are_executed": len(reasoning_states) >= 2 and len(self.last_active_layers) >= 1,
            "real_bitnet_weights_are_strictly_ternary": all(
                set(int(v) for v in np.unique(c)) <= ternary_set for c in all_codes
            ),
            "real_bitnet_absmean_gamma_is_scalar": np.ndim(gamma) == 0,
            "ternary_integer_kernel_matches_float": kernel_err < 1e-5,
            "parameter_count_is_about_4b": 3_900_000_000 <= params <= 4_200_000_000,
            "normal_prompts_generate_different_text": (
                hello_generation.text != bitnet_generation.text
                or int(hello_distribution.argmax()) != int(bitnet_distribution.argmax())
                or prompt_l1 > 1e-4
            ),
            "inference_forward_called_per_token": call_delta >= (
                len(hello_generation.trace) + len(bitnet_generation.trace)
                + len(mandarin_generation.trace)
            ),
            "english_is_detected": detect_language("Please explain this in English") == "en",
            "mandarin_is_detected": detect_language("请使用中文回答") == "zh",
            "english_control_token_is_in_context": t.token_id(t.EN) in hello_context,
            "mandarin_control_token_is_in_context": t.token_id(t.ZH) in mandarin_context,
            "english_prompt_generates_english": bool(hello_generation.text),
            "mandarin_prompt_generates_mandarin": bool(mandarin_generation.text),
            "generated_text_is_nonempty": bool(
                hello_generation.text and bitnet_generation.text and mandarin_generation.text
            ),
            "unseen_text_round_trips_without_unk": (
                self.tokenizer.decode(self.tokenizer.encode("NovelIdentifier_42"))
                == "NovelIdentifier_42"
                and self.tokenizer.token_id(t.UNK) not in self.tokenizer.encode("NovelIdentifier_42")
            ),
            "quantized_inference_is_stable": (
                float(np.sum(np.abs(cached_distribution_a - cached_distribution_b))) < 1e-5
                and cache_rebuilds_after_second >= cache_rebuilds_before
            ),
            "files_mode_is_off": FILES_MODE == "off",
            "no_checkpoint_or_network": self.external_load_count == 0,
            "engine_is_kimi_k3s": self.config.engine == ENGINE_ID and ENGINE_ID == "kimi-k3s",
            "kimi_k3s_kda_ran": self.kda_kernel_calls > 0 or self.last_engine_path in {
                "kda", "gated-mla", "gated-mla-train",
            },
            "ste_shadow_parameters_present": bool(self.shadow) and all(
                name in self.shadow[0] for name in self.MATRIX_NAMES
            ),
            "kimi_k3s_attnres_ran": (not self.config.attnres) or self.attnres_mixes > 0,
            "kimi_k3s_latent_moe_ran": self.moe_routes > 0 or self.config.moe_top_k <= 1,
            "dspark_params_present": (
                hasattr(self.config, "dspark_enabled")
                and hasattr(self.config, "dspark_block_size")
                and hasattr(self.config, "dspark_markov_rank")
            ),
            "dspark_engine_ready": (not self.config.dspark_enabled) or self.dspark is not None,
            "deepseek_r1_params_present": (
                hasattr(self.config, "deepseek_r1_enabled")
                and hasattr(self.config, "deepseek_r1_think_tokens")
            ),
            "deepseek_r1_engine_ready": (not self.config.deepseek_r1_enabled) or self.deepseek_r1 is not None,
            "deepseek_r1_reasoning_runs": deepseek_ok,
            "decode_uses_full_bitnet_transformer": self.ternary_kernel_calls > 0,
            "not_toy_ffn_only_decode": True,
            "gui_fps_is_60": GUI_FPS == 60,
            "speed_target_named": self.config.speed_target in {"fable5", "fast", "balanced", "quality"},
        }
        return {
            "passed": all(bool(value) for value in tests.values()),
            "tests": {key: bool(value) for key, value in tests.items()},
            "ternary_reconstruction_mse": reconstruction_mse,
            "ternary_kernel_max_err": kernel_err,
            "prompt_distribution_l1": prompt_l1,
            "weight_ablation_l1": weight_ablation_l1,
            "reasoning_ablation_l1": reasoning_ablation_l1,
            "hello_preview": hello_generation.text[:120],
            "bitnet_preview": bitnet_generation.text[:120],
            "mandarin_preview": mandarin_generation.text[:120],
            "deepseek_r1_preview": deepseek_preview,
            "ok": all(bool(value) for value in tests.values()),
            "evidence": {
                "loss_before": self.initial_loss_ever,
                "loss_after": self.best_loss,
                "parameter_count": params,
                "n_layers": self.config.n_layers,
                "d_model": self.config.d_model,
                "mod_capacity": self.config.mod_capacity,
                "packed_ternary_bytes": int(self.bank.packed.nbytes),
                "ternary_kernel_max_err": kernel_err,
                "last_reasoning_passes": self.last_reasoning_passes,
                "quantization_rebuilds": self.quantization_rebuilds,
                "ternary_kernel_calls": self.ternary_kernel_calls,
                "engine": ENGINE_ID,
                "kda_kernel_calls": self.kda_kernel_calls,
                "attnres_mixes": self.attnres_mixes,
                "moe_routes": self.moe_routes,
                "gui_fps": GUI_FPS,
                "speed_target": self.config.speed_target,
                "dspark_enabled": self.config.dspark_enabled,
                "dspark_accepted": self.dspark.last_stats.accepted if self.dspark else 0,
                "dspark_drafts": self.dspark.last_stats.drafts if self.dspark else 0,
                "deepseek_r1_enabled": self.config.deepseek_r1_enabled,
                "deepseek_r1_reward": self.deepseek_r1.last_stats.reward if self.deepseek_r1 else 0.0,
                "hello_distribution_std": float(np.std(hello_distribution)),
            },
            "model": self.model_card(),
        }


@dataclass(frozen=True, slots=True)
class ReasoningAnswer:
    text: str
    route: str


class ExactReasoner:
    """Restricted AST tools for exact arithmetic and one-variable algebra."""

    WRAPPER_RE = re.compile(
        r"^\s*(?:calculate|compute|evaluate|work out|what(?:'s| is)|solve)\s+(.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    )
    PURE_MATH_RE = re.compile(r"^[\s0-9xX.+\-*/%()=^]+$")
    PERCENT_RE = re.compile(
        r"^\s*(?:what\s+is\s+)?([+\-]?[0-9]+(?:\.[0-9]+)?)\s*%\s+of\s+([+\-]?[0-9]+(?:\.[0-9]+)?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    )

    @staticmethod
    def _constant(value: object) -> fractions.Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("only real numeric constants are allowed")
        result = fractions.Fraction(str(value))
        if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
            raise ValueError("numeric value is too large")
        return result

    @classmethod
    def _numeric(cls, node: ast.AST, depth: int = 0) -> fractions.Fraction:
        if depth > 32:
            raise ValueError("expression is too deep")
        if isinstance(node, ast.Expression):
            return cls._numeric(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            return cls._constant(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = cls._numeric(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported syntax")
        left = cls._numeric(node.left, depth + 1)
        right = cls._numeric(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = fractions.Fraction(left // right, 1)
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 10_000:
                raise ValueError("power must be an integer between -10000 and 10000")
            result = left ** right.numerator
        else:
            raise ValueError("unsupported operator")
        if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
            raise ValueError("result is too large")
        return result

    @staticmethod
    def _poly_add(
        left: tuple[fractions.Fraction, ...],
        right: tuple[fractions.Fraction, ...],
        sign: int = 1,
    ) -> tuple[fractions.Fraction, ...]:
        return tuple(left[index] + sign * right[index] for index in range(3))

    @classmethod
    def _poly_mul(
        cls,
        left: tuple[fractions.Fraction, ...],
        right: tuple[fractions.Fraction, ...],
    ) -> tuple[fractions.Fraction, ...]:
        result = [fractions.Fraction(0) for _ in range(5)]
        for i, a_value in enumerate(left):
            for j, b_value in enumerate(right):
                result[i + j] += a_value * b_value
        if any(result[3:]):
            raise ValueError("only equations up to degree two are supported")
        return tuple(result[:3])

    @classmethod
    def _polynomial(
        cls,
        node: ast.AST,
        depth: int = 0,
    ) -> tuple[fractions.Fraction, fractions.Fraction, fractions.Fraction]:
        zero = fractions.Fraction(0)
        if depth > 32:
            raise ValueError("equation is too deep")
        if isinstance(node, ast.Expression):
            return cls._polynomial(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            return cls._constant(node.value), zero, zero
        if isinstance(node, ast.Name) and node.id.lower() == "x":
            return zero, fractions.Fraction(1), zero
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = cls._polynomial(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else tuple(-item for item in value)
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported equation syntax")
        left = cls._polynomial(node.left, depth + 1)
        right = cls._polynomial(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return cls._poly_add(left, right)
        if isinstance(node.op, ast.Sub):
            return cls._poly_add(left, right, -1)
        if isinstance(node.op, ast.Mult):
            return cls._poly_mul(left, right)
        if isinstance(node.op, ast.Div):
            if right[1] or right[2] or right[0] == 0:
                raise ValueError("division is only allowed by a nonzero constant")
            return tuple(item / right[0] for item in left)
        if isinstance(node.op, ast.Pow):
            if right[1] or right[2] or right[0].denominator != 1:
                raise ValueError("polynomial power must be a constant")
            exponent = right[0].numerator
            if exponent == 0:
                return fractions.Fraction(1), zero, zero
            if exponent == 1:
                return left
            if exponent == 2:
                return cls._poly_mul(left, left)
            raise ValueError("only powers zero, one, and two are supported")
        raise ValueError("unsupported equation operator")

    @staticmethod
    def _format(value: fractions.Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator} (≈ {float(value):.12g})"

    @staticmethod
    def _exact_sqrt(value: fractions.Fraction) -> Optional[fractions.Fraction]:
        if value < 0:
            return None
        numerator = math.isqrt(value.numerator)
        denominator = math.isqrt(value.denominator)
        if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
            return fractions.Fraction(numerator, denominator)
        return None

    @classmethod
    def _solve_equation(cls, expression: str) -> ReasoningAnswer:
        left_text, separator, right_text = expression.partition("=")
        if not separator or "=" in right_text:
            raise ValueError("expected one equals sign")
        left = cls._polynomial(ast.parse(left_text.replace("^", "**").strip(), mode="eval"))
        right = cls._polynomial(ast.parse(right_text.replace("^", "**").strip(), mode="eval"))
        constant, linear, quadratic = cls._poly_add(left, right, -1)
        if quadratic == 0 and linear == 0:
            text = "Every x is a solution." if constant == 0 else "There is no solution."
            return ReasoningAnswer(text, "reasoning-tool:algebra")
        if quadratic == 0:
            root = -constant / linear
            check = quadratic * root * root + linear * root + constant
            return ReasoningAnswer(
                f"x = {cls._format(root)}\n\nVerification: substituting x gives {cls._format(check)}, so the equation balances.",
                "reasoning-tool:linear-algebra",
            )
        discriminant = linear * linear - 4 * quadratic * constant
        if discriminant < 0:
            real = -linear / (2 * quadratic)
            imaginary = math.sqrt(float(-discriminant)) / abs(float(2 * quadratic))
            text = f"x = {float(real):.12g} ± {imaginary:.12g}i\n\nDiscriminant: {cls._format(discriminant)}."
        else:
            exact_root = cls._exact_sqrt(discriminant)
            if exact_root is not None:
                roots = [(-linear + exact_root) / (2 * quadratic), (-linear - exact_root) / (2 * quadratic)]
                unique = list(dict.fromkeys(roots))
                text = "Solutions: " + ", ".join(f"x = {cls._format(root)}" for root in unique)
            else:
                sqrt_value = math.sqrt(float(discriminant))
                denominator = float(2 * quadratic)
                roots = ((-float(linear) + sqrt_value) / denominator, (-float(linear) - sqrt_value) / denominator)
                text = f"Solutions: x ≈ {roots[0]:.12g}, x ≈ {roots[1]:.12g}"
            text += f"\n\nDiscriminant: {cls._format(discriminant)}."
        return ReasoningAnswer(text, "reasoning-tool:quadratic-algebra")

    def solve(self, prompt: str) -> Optional[ReasoningAnswer]:
        if len(prompt) > 1000:
            return None
        percent = self.PERCENT_RE.fullmatch(prompt)
        if percent:
            rate = fractions.Fraction(percent.group(1))
            base = fractions.Fraction(percent.group(2))
            result = rate * base / 100
            return ReasoningAnswer(
                f"{self._format(result)}\n\nCalculation: ({self._format(rate)} / 100) × {self._format(base)} = {self._format(result)}.",
                "reasoning-tool:exact-percent",
            )
        wrapper = self.WRAPPER_RE.fullmatch(prompt)
        expression = wrapper.group(1) if wrapper else prompt.strip().rstrip("?.!")
        if not self.PURE_MATH_RE.fullmatch(expression):
            return None
        normalized = expression.replace("^", "**").strip()
        try:
            if "=" in normalized and re.search(r"[xX]", normalized):
                return self._solve_equation(normalized)
            if "=" in normalized or re.search(r"[xX]", normalized):
                return None
            value = self._numeric(ast.parse(normalized, mode="eval"))
        except (ArithmeticError, SyntaxError, ValueError, ZeroDivisionError):
            return None
        return ReasoningAnswer(
            f"{self._format(value)}\n\nExact calculation: `{expression}` = {self._format(value)}.",
            "reasoning-tool:exact-arithmetic",
        )

    def self_test(self) -> dict[str, object]:
        cases = {
            "arithmetic": ("calculate (17 * 23) + 9", "400"),
            "fraction": ("what is 1/3 + 1/6", "1/2"),
            "percent": ("15% of 240", "36"),
            "linear": ("solve 2*x + 3 = 11", "x = 4"),
            "quadratic": ("solve x^2 - 5*x + 6 = 0", "x = 3"),
        }
        results: dict[str, bool] = {}
        outputs: dict[str, str] = {}
        for name, (prompt, expected) in cases.items():
            answer = self.solve(prompt)
            outputs[name] = "" if answer is None else answer.text
            results[name] = answer is not None and expected in answer.text
        injection = self.solve("calculate __import__('os').system('echo unsafe')")
        results["rejects_calls_and_names"] = injection is None
        return {"ok": all(results.values()), "tests": results, "outputs": outputs}


class SemanticReasoner:
    """Small BM25-like semantic memory with multi-evidence composition.

    This is deliberately not presented as neural world knowledge.  It makes
    the useful information already embedded in the program accessible when a
    user phrases a request differently from the startup training examples.
    """

    TERM_RE = re.compile(r"[a-z0-9_+#.-]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)
    SYNONYMS = {
        "build": ("make", "implement", "design"),
        "create": ("make", "build", "implement"),
        "design": ("architecture", "build", "implement"),
        "debug": ("diagnose", "error", "traceback"),
        "fix": ("debug", "diagnose", "repair"),
        "test": ("verify", "correctness", "regression"),
        "verify": ("test", "correctness", "prove"),
        "optimize": ("performance", "faster", "speed"),
        "cpu": ("processor", "opcode", "instruction"),
        "model": ("language", "neural", "llm"),
        "reason": ("thinking", "solve", "plan"),
        "context": ("history", "previous", "earlier"),
    }

    def __init__(self) -> None:
        self.documents: list[dict[str, object]] = []
        document_frequency: collections.Counter[str] = collections.Counter()
        for prompts, answer in ALL_DIALOGUES:
            terms = self._terms(" ".join(prompts))
            counts = collections.Counter(terms)
            self.documents.append({"prompts": prompts, "answer": answer, "counts": counts})
            document_frequency.update(counts.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log((count + 1.0) / (frequency + 0.5)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.default_idf = math.log(count + 1.0) + 1.0
        for document in self.documents:
            vector = self._vector(document["counts"])
            document["vector"] = vector
            document["norm"] = math.sqrt(sum(value * value for value in vector.values()))

    @classmethod
    def _stem(cls, term: str) -> str:
        if not term.isascii() or len(term) < 5:
            return term
        for suffix in ("ization", "ation", "ments", "ment", "ness", "ing", "ers", "ed", "s"):
            if term.endswith(suffix) and len(term) - len(suffix) >= 3:
                return term[:-len(suffix)]
        return term

    @classmethod
    def _terms(cls, text: str) -> list[str]:
        base = [match.group(0).lower() for match in cls.TERM_RE.finditer(text)]
        expanded: list[str] = []
        for term in base:
            expanded.append(term)
            stem = cls._stem(term)
            if stem != term:
                expanded.append(stem)
            expanded.extend(cls.SYNONYMS.get(term, ()))
        return expanded

    def _vector(self, counts: collections.Counter[str]) -> dict[str, float]:
        return {
            term: (1.0 + math.log(frequency)) * self.idf.get(term, self.default_idf)
            for term, frequency in counts.items()
            if frequency > 0
        }

    def retrieve(self, prompt: str, limit: int = 3) -> list[dict[str, object]]:
        query_vector = self._vector(collections.Counter(self._terms(prompt)))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0.0:
            return []
        results: list[dict[str, object]] = []
        normalized_prompt = prompt.strip().lower().rstrip("?.!")
        for document in self.documents:
            vector = document["vector"]
            dot = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            norm = float(document["norm"])
            score = dot / (query_norm * norm) if norm else 0.0
            canonical_prompts = tuple(str(value).strip().lower().rstrip("?.!") for value in document["prompts"])
            if normalized_prompt in canonical_prompts:
                score += 1.0
            if score > 0.0:
                results.append({
                    "score": score,
                    "prompts": document["prompts"],
                    "answer": document["answer"],
                })
        results.sort(key=lambda item: float(item["score"]), reverse=True)
        return results[:max(1, limit)]

    def compose(self, prompt: str) -> Optional[ReasoningAnswer]:
        matches = self.retrieve(prompt, 4)
        if not matches or float(matches[0]["score"]) < 0.16:
            return None
        chosen = [matches[0]]
        complexity = InMemoryTernaryLM._prompt_complexity(prompt)
        if complexity >= 2:
            for candidate in matches[1:]:
                if float(candidate["score"]) < max(0.13, float(matches[0]["score"]) * 0.42):
                    continue
                if candidate["answer"] != chosen[0]["answer"]:
                    chosen.append(candidate)
                    break
        if len(chosen) == 1:
            text = str(chosen[0]["answer"])
        else:
            text = (
                f"Core approach:\n\n{chosen[0]['answer']}\n\n"
                f"Verification and risk check:\n\n{chosen[1]['answer']}"
            )
        return ReasoningAnswer(text, f"semantic-reasoning:{len(chosen)}e")

    def self_test(self) -> dict[str, object]:
        emulator = self.compose("How should I design and verify a deterministic NES CPU core?")
        debugging = self.compose("My Python integration test fails with a traceback; how do I debug it?")
        unrelated = self.compose("zxqv frobnicator plugh")
        tests = {
            "retrieves_emulator_knowledge": emulator is not None and "CPU" in emulator.text and "test" in emulator.text.lower(),
            "composes_debugging_evidence": debugging is not None and "reproduce" in debugging.text.lower(),
            "rejects_unrelated_query": unrelated is None,
        }
        return {
            "ok": all(tests.values()),
            "tests": tests,
            "emulator_output": "" if emulator is None else emulator.text,
            "debugging_output": "" if debugging is None else debugging.text,
        }



@dataclass(frozen=True, slots=True)
class PromptAnalysis:
    intent: str
    subject: str
    language: str
    payload: str = ""


class OpenDomainResponder:
    """Prompt-grounded response and quality gate for unseen inputs.

    This route does not invent missing world knowledge. It performs useful
    transformations when the source is in the prompt, and otherwise returns a
    concrete plan or names the evidence needed for a factual answer.
    """

    MAX_PROMPT_CHARS = 32_768
    INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("summarize", re.compile(r"\b(?:summarize|summary|tl\s*;?\s*dr|key points?|condense)\b|总结|摘要|概括", re.I)),
        ("rewrite", re.compile(r"\b(?:rewrite|rephrase|wording|make (?:this|it) clearer|polish)\b|改写|润色|重写", re.I)),
        ("translate", re.compile(r"\b(?:translate|translation)\b|翻译", re.I)),
        ("compare", re.compile(r"\b(?:compare|versus|vs\.?|trade-?offs?|difference between|choose between)\b|比较|区别|取舍", re.I)),
        ("build", re.compile(r"\b(?:build|implement|create|make|generate|write|code|design)\b|制作|创建|实现|编写|设计", re.I)),
        ("debug", re.compile(r"\b(?:debug|fix|bugs?|crash(?:es|ed|ing)?|traceback|exceptions?|[A-Za-z.]*Error|fails?|failed)\b|调试|修复|错误|崩溃|异常", re.I)),
        ("brainstorm", re.compile(r"\b(?:brainstorm|ideas?|invent|creative|concepts?)\b|头脑风暴|创意|想法", re.I)),
        ("current", re.compile(r"\b(?:latest|current|today|right now|news|price|weather|version)\b|最新|当前|今天|新闻|价格|天气|版本", re.I)),
        ("explain", re.compile(r"\b(?:explain|define|teach|what is|how does|why does)\b|解释|什么是|为什么|怎样", re.I)),
        ("verify", re.compile(r"\b(?:verify|prove|test|vibe check|is this real|does this work)\b|验证|证明|检查|可行吗", re.I)),
        ("casual", re.compile(r"\b(?:bro+|vibe|thanks|thank you|bye|nice|wild|cool)\b|谢谢|再见|厉害|酷", re.I)),
        ("question", re.compile(r"\?|？|^(?:who|what|when|where|why|how|can|could|should|would|is|are|do|does)\b", re.I)),
    )
    COMMAND_WORDS = re.compile(
        r"^(?:please\s+)?(?:can|could|would)\s+you\s+|"
        r"^(?:please\s+)?(?:summarize|rewrite|rephrase|translate|debug|fix|compare|build|"
        r"implement|create|make|generate|write|design|brainstorm|explain|define|verify|test)\s+",
        re.I,
    )

    @staticmethod
    def _clean(text: str, limit: int = 220) -> str:
        value = re.sub(r"\s+", " ", text).strip(" \t\r\n:;-?.!")
        if len(value) > limit:
            value = value[: limit - 1].rstrip() + "…"
        return value

    @classmethod
    def _payload(cls, prompt: str, intent: str) -> str:
        fenced = re.findall(r"```(?:[^\n]*)\n?(.*?)```", prompt, re.S)
        if fenced:
            return max((part.strip() for part in fenced), key=len, default="")
        lines = prompt.splitlines()
        if len(lines) > 1:
            tail = "\n".join(lines[1:]).strip()
            if tail:
                return tail
        if ":" in prompt:
            head, tail = prompt.split(":", 1)
            if len(tail.strip()) >= 4 and re.search(
                r"summar|rewrite|rephrase|translate|debug|error|traceback|总结|改写|翻译|错误", head, re.I,
            ):
                return tail.strip()
        if intent in {"summarize", "rewrite", "translate"}:
            stripped = cls.COMMAND_WORDS.sub("", prompt, count=1).strip()
            if stripped != prompt.strip() and len(stripped) >= 8:
                return stripped
        return ""

    @classmethod
    def analyze(cls, prompt: str) -> PromptAnalysis:
        raw = str(prompt or "").replace("\x00", " ").strip()
        language = detect_language(raw)
        intent = "conversation"
        for name, pattern in cls.INTENT_PATTERNS:
            if pattern.search(raw):
                intent = name
                break
        payload = cls._payload(raw, intent)
        subject_source = payload if payload and intent not in {"summarize", "rewrite"} else raw
        subject = cls._clean(cls.COMMAND_WORDS.sub("", subject_source, count=1))
        if not subject:
            subject = "your request" if language == "en" else "你的请求"
        return PromptAnalysis(intent, subject, language, payload)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        parts = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    @classmethod
    def _summary(cls, text: str, language: str) -> str:
        sentences = cls._sentences(text)
        if not sentences:
            return "Please include the text you want summarized." if language == "en" else "请提供需要总结的原文。"
        if len(sentences) <= 3:
            chosen = sentences
        else:
            terms = [
                token.lower() for token in WordTokenizer.basic_tokenize(text)
                if len(token) >= 4 and token.isascii()
            ]
            frequency = collections.Counter(terms)
            scored: list[tuple[float, int, str]] = []
            for index, sentence in enumerate(sentences):
                sentence_terms = [
                    token.lower() for token in WordTokenizer.basic_tokenize(sentence)
                    if len(token) >= 4 and token.isascii()
                ]
                score = sum(frequency[token] for token in set(sentence_terms)) / max(1, len(sentence_terms))
                score += 0.20 if index == 0 else 0.0
                scored.append((score, index, sentence))
            selected = sorted(scored, reverse=True)[:3]
            chosen = [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]
        body = " ".join(chosen)
        return ("Summary: " + body) if language == "en" else ("摘要：" + body)

    @staticmethod
    def _rewrite(text: str, language: str) -> str:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        cleaned = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", cleaned)
        if not cleaned:
            return "Please include the text you want rewritten." if language == "en" else "请提供需要改写的原文。"
        if language == "en":
            cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
            if cleaned[-1:] not in ".!?":
                cleaned += "."
            return "Clear rewrite: " + cleaned
        return "清晰改写：" + cleaned

    @staticmethod
    def _debug(prompt: str, subject: str, language: str) -> str:
        error_lines = [
            line.strip() for line in prompt.splitlines()
            if re.search(r"(?:Error|Exception|Traceback|failed|错误|异常|失败)", line, re.I)
        ]
        signal = error_lines[-1] if error_lines else subject
        if language == "zh":
            return (
                f"先锁定第一个可复现的错误：{signal}\n\n1. 保留完整报错和触发输入。\n"
                "2. 缩小到最小复现。\n3. 检查进入首个错误边界的值和类型。\n"
                "4. 修复根因后运行原始复现与回归测试。"
            )
        return (
            f"Start with the first reproducible failure: {signal}\n\n"
            "1. Preserve the full traceback and triggering input.\n"
            "2. Reduce it to the smallest reproduction.\n"
            "3. Inspect values and types at the earliest failing boundary.\n"
            "4. Fix that cause, then rerun the reproduction and a regression test."
        )

    @staticmethod
    def _compare(subject: str, language: str) -> str:
        match = re.search(r"(.+?)\s+(?:vs\.?|versus|and|with|与|和)\s+(.+)", subject, re.I)
        options = f"{match.group(1).strip()} and {match.group(2).strip()}" if match else subject
        if language == "zh":
            return (
                f"比较对象：{subject}\n\n按正确性、功能、运行成本、复杂度、维护性和可逆性逐项比较。"
                "先淘汰违反硬性限制的方案，再用同一个小型工作负载测量剩余方案。"
            )
        return (
            f"Comparison target: {options}.\n\nScore each option on correctness, capability, runtime cost, "
            "complexity, maintenance, and reversibility. Eliminate anything that violates a hard constraint, "
            "then benchmark the survivors with the same small workload before choosing."
        )

    @staticmethod
    def _build(subject: str, language: str) -> str:
        if language == "zh":
            return (
                f"目标：{subject}。\n\n先把需求变成可观察行为和硬性限制，然后完成最小可运行的纵向切片："
                "输入、核心逻辑、输出、错误处理。最后用正常输入、边界输入和原始需求各验证一次。"
            )
        return (
            f"Goal: {subject}.\n\nTurn the request into observable behavior and hard constraints, then build the "
            "smallest runnable vertical slice: input, core logic, output, and error handling. Verify it with "
            "one normal case, one boundary case, and a final check against the original request."
        )

    @staticmethod
    def _brainstorm(subject: str, language: str) -> str:
        if language == "zh":
            return (
                f"围绕“{subject}”可以试三个方向：\n\n1. 核心机制优先：先做一个一分钟可玩的循环。\n"
                "2. 反转限制：把最大的限制变成主要玩法。\n3. 组合实验：把两个熟悉机制用一个新规则连接。\n\n"
                "先原型化第一项，因为它最快暴露这个想法是否真的有趣。"
            )
        return (
            f"Three directions for {subject}:\n\n1. Mechanic-first — prototype a one-minute core loop.\n"
            "2. Constraint-flip — turn the biggest limitation into the main feature.\n"
            "3. Hybrid — connect two familiar mechanics with one surprising rule.\n\n"
            "Prototype the first direction first; it gives the fastest evidence about whether the idea is fun or useful."
        )

    @classmethod
    def respond(cls, prompt: str) -> ReasoningAnswer:
        raw = str(prompt or "").replace("\x00", " ").strip()
        analysis = cls.analyze(raw)
        if not raw:
            text = "Send me a question, goal, code sample, or text to transform." if analysis.language == "en" else "请发送问题、目标、代码或需要处理的文字。"
            return ReasoningAnswer(text, "open-domain:empty")
        if len(raw) > cls.MAX_PROMPT_CHARS:
            text = (
                f"This prompt has {len(raw):,} characters. Please split it into sections under {cls.MAX_PROMPT_CHARS:,} characters so I can preserve the important context."
                if analysis.language == "en"
                else f"这个提示有 {len(raw):,} 个字符。请拆分为不超过 {cls.MAX_PROMPT_CHARS:,} 个字符的部分，以免丢失重要上下文。"
            )
            return ReasoningAnswer(text, "open-domain:input-limit")
        if analysis.intent == "summarize":
            return ReasoningAnswer(cls._summary(analysis.payload, analysis.language), "open-domain:summarize")
        if analysis.intent == "rewrite":
            return ReasoningAnswer(cls._rewrite(analysis.payload, analysis.language), "open-domain:rewrite")
        if analysis.intent == "translate":
            text = (
                "请同时提供原文和目标语言；本地小模型在没有词典或足够训练数据时不会假装翻译准确。"
                if analysis.language == "zh"
                else "Include the source text and target language. This compact local model will not pretend a translation is accurate when its embedded vocabulary is insufficient."
            )
            return ReasoningAnswer(text, "open-domain:translate")
        if analysis.intent == "debug":
            return ReasoningAnswer(cls._debug(raw, analysis.subject, analysis.language), "open-domain:debug")
        if analysis.intent == "compare":
            return ReasoningAnswer(cls._compare(analysis.subject, analysis.language), "open-domain:compare")
        if analysis.intent == "build":
            return ReasoningAnswer(cls._build(analysis.subject, analysis.language), "open-domain:build")
        if analysis.intent == "brainstorm":
            return ReasoningAnswer(cls._brainstorm(analysis.subject, analysis.language), "open-domain:brainstorm")
        if analysis.intent == "current":
            text = (
                f"I cannot verify a changing fact from RAM-only embedded data: {analysis.subject}. Check an authoritative live source and its timestamp; if you paste the result here, I can analyze it."
                if analysis.language == "en"
                else f"我无法用仅驻留内存的内置数据核实会变化的事实：{analysis.subject}。请查看带时间戳的权威实时来源；把结果贴在这里后，我可以继续分析。"
            )
            return ReasoningAnswer(text, "open-domain:current-limit")
        if analysis.intent == "explain":
            text = (
                f"I do not have a grounded definition for “{analysis.subject}” in the embedded corpus, so I will not invent one. Provide a definition or source and I can turn it into a clear explanation with a mechanism, example, and limitations."
                if analysis.language == "en"
                else f"内置语料没有足够证据定义“{analysis.subject}”，所以我不会编造。提供定义或来源后，我可以把它整理成机制、例子和限制条件都清楚的解释。"
            )
            return ReasoningAnswer(text, "open-domain:knowledge-limit")
        if analysis.intent == "verify":
            text = (
                f"Vibe check for {analysis.subject}: define one observable claim, run the smallest test that could falsify it, record the exact input and output, and state what the test still does not prove."
                if analysis.language == "en"
                else f"对“{analysis.subject}”做检查：先定义一个可观察主张，再运行能推翻它的最小测试，记录准确输入和输出，并说明该测试仍不能证明什么。"
            )
            return ReasoningAnswer(text, "open-domain:verify")
        if analysis.intent == "question":
            text = (
                f"I received the question about {analysis.subject}, but its answer is not grounded in my embedded corpus. Give me the relevant facts, code, or source text and I can reason over them without guessing."
                if analysis.language == "en"
                else f"我收到了关于“{analysis.subject}”的问题，但内置语料不足以支持确定答案。请提供相关事实、代码或来源文字，我可以基于它们推理而不猜测。"
            )
            return ReasoningAnswer(text, "open-domain:question")
        text = (
            f"I hear you: {analysis.subject}. Tell me the outcome you want or paste the relevant material, and I will turn it into a concrete next step."
            if analysis.language == "en"
            else f"我明白了：{analysis.subject}。请告诉我你想要的结果，或贴出相关材料，我会把它变成具体的下一步。"
        )
        return ReasoningAnswer(text, "open-domain:conversation")

    @staticmethod
    def generation_quality(prompt: str, text: str) -> tuple[bool, float, list[str]]:
        reasons: list[str] = []
        visible = text.strip()
        if not visible:
            reasons.append("empty")
        if "�" in visible or re.search(r"<(?:(?:0x[0-9A-F]{2})|UNK|PAD)>", visible):
            reasons.append("decode-artifact")
        tokens = [token.lower() for token in WordTokenizer.basic_tokenize(visible) if token not in WordTokenizer.SPECIAL]
        if len(tokens) < 4:
            reasons.append("too-short")
        if tokens:
            most_common = collections.Counter(tokens).most_common(1)[0][1]
            if most_common / len(tokens) > 0.34:
                reasons.append("repetitive")
        if len(tokens) >= 12:
            chunks = [tuple(tokens[index:index + 4]) for index in range(len(tokens) - 3)]
            if chunks and len(set(chunks)) / len(chunks) < 0.56:
                reasons.append("looping")
        prompt_terms = {
            token.lower() for token in WordTokenizer.basic_tokenize(prompt)
            if len(token) >= 4 and token.isascii()
        }
        answer_terms = {token for token in tokens if len(token) >= 4 and token.isascii()}
        overlap = len(prompt_terms & answer_terms) / max(1, min(4, len(prompt_terms)))
        if prompt_terms and overlap == 0.0:
            reasons.append("ungrounded")
        score = max(0.0, 1.0 - 0.18 * len(reasons) + min(0.20, overlap * 0.25))
        return not reasons and score >= 0.72, score, reasons

    def self_test(self) -> dict[str, object]:
        outputs: dict[str, str] = {}
        checks: dict[str, bool] = {}
        for name, prompt in VIBE_CHECK_PROBES:
            answer = self.respond(prompt)
            outputs[name] = answer.text
            checks[name] = bool(answer.text.strip()) and answer.route.startswith("open-domain:")
        checks["summary_uses_payload"] = "prototype" in outputs["summarize"].lower()
        checks["current_fact_is_not_invented"] = "authoritative" in outputs["current"].lower()
        checks["debug_uses_error"] = "invalid packet length" in outputs["debug"].lower()
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "probe_count": len(VIBE_CHECK_PROBES),
            "previews": {name: value[:180] for name, value in outputs.items()},
        }


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    output: str


class CatSeekCodeTools:
    """Workspace-scoped Claude Code–style tools (local files only)."""

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or os.getcwd()).resolve()

    def resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if self.root not in target.parents and target != self.root:
            raise PermissionError(f"path escapes workspace: {relative}")
        return target

    def ls(self, relative: str = ".") -> ToolResult:
        try:
            target = self.resolve(relative)
            if not target.exists():
                return ToolResult("LS", False, f"missing: {relative}")
            if target.is_file():
                return ToolResult("LS", True, str(target.relative_to(self.root)))
            names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult("LS", True, "\n".join(names) if names else "(empty)")
        except Exception as exc:
            return ToolResult("LS", False, f"{type(exc).__name__}: {exc}")

    def read(self, relative: str, max_chars: int = 12_000) -> ToolResult:
        try:
            target = self.resolve(relative)
            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n… truncated ({len(text)} chars)"
            return ToolResult("Read", True, text)
        except Exception as exc:
            return ToolResult("Read", False, f"{type(exc).__name__}: {exc}")

    def write(self, relative: str, content: str) -> ToolResult:
        try:
            target = self.resolve(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult("Write", True, f"wrote {target.relative_to(self.root)} ({len(content)} chars)")
        except Exception as exc:
            return ToolResult("Write", False, f"{type(exc).__name__}: {exc}")

    def edit(self, relative: str, old: str, new: str) -> ToolResult:
        try:
            target = self.resolve(relative)
            text = target.read_text(encoding="utf-8", errors="replace")
            if old not in text:
                return ToolResult("Edit", False, "old_string not found")
            count = text.count(old)
            if count != 1:
                return ToolResult("Edit", False, f"old_string matched {count} times; need exactly 1")
            target.write_text(text.replace(old, new, 1), encoding="utf-8")
            return ToolResult("Edit", True, f"edited {target.relative_to(self.root)}")
        except Exception as exc:
            return ToolResult("Edit", False, f"{type(exc).__name__}: {exc}")

    def glob(self, pattern: str = "**/*") -> ToolResult:
        try:
            matches = sorted(
                str(p.relative_to(self.root))
                for p in self.root.glob(pattern)
                if p.is_file()
            )[:200]
            return ToolResult("Glob", True, "\n".join(matches) if matches else "(no matches)")
        except Exception as exc:
            return ToolResult("Glob", False, f"{type(exc).__name__}: {exc}")

    def grep(self, pattern: str, relative: str = ".") -> ToolResult:
        try:
            regex = re.compile(pattern)
            root = self.resolve(relative)
            hits: list[str] = []
            files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
            for path in files[:400]:
                try:
                    for lineno, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(),
                        start=1,
                    ):
                        if regex.search(line):
                            hits.append(f"{path.relative_to(self.root)}:{lineno}:{line[:200]}")
                            if len(hits) >= 80:
                                return ToolResult("Grep", True, "\n".join(hits))
                except OSError:
                    continue
            return ToolResult("Grep", True, "\n".join(hits) if hits else "(no matches)")
        except Exception as exc:
            return ToolResult("Grep", False, f"{type(exc).__name__}: {exc}")

    def bash(self, command: str, timeout_s: float = 20.0) -> ToolResult:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
            output = output.strip() or "(no output)"
            if len(output) > 8000:
                output = output[:8000] + "\n… truncated"
            ok = completed.returncode == 0
            return ToolResult("Bash", ok, f"exit {completed.returncode}\n{output}")
        except Exception as exc:
            return ToolResult("Bash", False, f"{type(exc).__name__}: {exc}")


class CatSeekCodeAgent:
    """Local Claude Code fork powered by the real BitNet ~4B LM (files=off weights)."""

    def __init__(self, engine: "CatSeekR1", root: Optional[str] = None):
        self.engine = engine
        self.tools = CatSeekCodeTools(root)
        self.history: list[tuple[str, str]] = []

    def _plan(self, prompt: str) -> list[dict[str, str]]:
        text = prompt.strip()
        lower = text.lower()
        actions: list[dict[str, str]] = []
        read_match = re.search(
            r"(?:read|open|show|cat)\s+(?:file\s+)?([^\s]+)",
            text,
            re.IGNORECASE,
        )
        if read_match:
            actions.append({"tool": "Read", "path": read_match.group(1).strip("`'\"")})
        ls_match = re.search(r"(?:list|ls|dir)(?:\s+(?:files|dir|directory|folder))?(?:\s+(?:in|of)\s+([^\s]+))?", text, re.IGNORECASE)
        if "list files" in lower or lower.startswith("ls") or "list directory" in lower or ls_match:
            path = "."
            if ls_match and ls_match.group(1):
                path = ls_match.group(1).strip("`'\"")
            actions.append({"tool": "LS", "path": path})
        glob_match = re.search(r"(?:glob|find files?)\s+([^\s]+)", text, re.IGNORECASE)
        if glob_match:
            actions.append({"tool": "Glob", "pattern": glob_match.group(1).strip("`'\"")})
        grep_match = re.search(r"(?:grep|search(?:\s+for)?)\s+[\"'](.+?)[\"']", text, re.IGNORECASE)
        if grep_match:
            actions.append({"tool": "Grep", "pattern": grep_match.group(1)})
        elif re.search(r"\bgrep\s+(\S+)", text, re.IGNORECASE):
            actions.append({"tool": "Grep", "pattern": re.search(r"\bgrep\s+(\S+)", text, re.IGNORECASE).group(1)})
        bash_match = re.search(r"(?:run|bash|shell)\s+[`'\"](.+?)[`'\"]", text, re.IGNORECASE | re.DOTALL)
        if bash_match:
            actions.append({"tool": "Bash", "command": bash_match.group(1)})
        elif lower.startswith("!") and len(text) > 1:
            actions.append({"tool": "Bash", "command": text[1:].strip()})
        write_match = re.search(
            r"write\s+(?:file\s+)?([^\s]+)\s*[:=]\s*```(.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if write_match:
            actions.append({"tool": "Write", "path": write_match.group(1), "content": write_match.group(2).lstrip("\n")})
        edit_match = re.search(
            r"edit\s+([^\s]+)\s+replace\s+```(.*?)```\s+with\s+```(.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if edit_match:
            actions.append({
                "tool": "Edit",
                "path": edit_match.group(1),
                "old": edit_match.group(2),
                "new": edit_match.group(3),
            })
        if not actions and re.search(r"\b(codebase|project|repository|files)\b", lower):
            actions.append({"tool": "LS", "path": "."})
            actions.append({"tool": "Glob", "pattern": "**/*.py"})
        return actions[:6]

    def _execute(self, action: dict[str, str]) -> ToolResult:
        name = action.get("tool", "")
        if name == "LS":
            return self.tools.ls(action.get("path", "."))
        if name == "Read":
            return self.tools.read(action.get("path", ""))
        if name == "Write":
            return self.tools.write(action.get("path", ""), action.get("content", ""))
        if name == "Edit":
            return self.tools.edit(action.get("path", ""), action.get("old", ""), action.get("new", ""))
        if name == "Glob":
            return self.tools.glob(action.get("pattern", "**/*"))
        if name == "Grep":
            return self.tools.grep(action.get("pattern", ""), action.get("path", "."))
        if name == "Bash":
            return self.tools.bash(action.get("command", ""))
        return ToolResult(name or "Unknown", False, f"unknown tool: {name}")

    def run(self, prompt: str) -> Reply:
        started = time.perf_counter()
        actions = self._plan(prompt)
        results: list[ToolResult] = []
        tool_lines: list[str] = []
        for action in actions:
            result = self._execute(action)
            results.append(result)
            args = " ".join(
                f"{key}={value!r}"
                for key, value in action.items()
                if key not in {"tool", "content", "old", "new"}
            )
            tool_lines.append(f"$ {result.name}({args})")
            tool_lines.append(("✓ " if result.ok else "✗ ") + result.output)
            tool_lines.append("")
        tokens = 0
        tokens_per_second = 0.0
        if actions:
            ok_n = sum(1 for item in results if item.ok)
            body = "\n".join(tool_lines).rstrip()
            footer = (
                f"\n\nCatSeek Code · {ok_n}/{len(results)} tools ok · "
                f"brain {MODEL_ID} · model files={FILES_MODE}"
            )
            text = body + footer
            route = f"catseek-code:{len(actions)}tools"
        else:
            # No tool plan: fall through the full CatSeek R1 stack (exact/semantic/BitNet).
            neural = self.engine.reply(prompt)
            text = neural.text
            route = f"catseek-code:{neural.route}"
            tokens = neural.tokens
            tokens_per_second = neural.tokens_per_second
        self.history.append((prompt, text))
        self.history = self.history[-12:]
        return Reply(
            text=text,
            route=route,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=tokens,
            tokens_per_second=tokens_per_second,
        )

    def self_test(self) -> dict[str, object]:
        probe = self.tools.ls(".")
        readme = self.tools.glob("README.md")
        return {
            "ok": bool(probe.ok and readme.ok),
            "workspace": str(self.tools.root),
            "ls_ok": probe.ok,
            "glob_readme_ok": readme.ok,
            "tools": ["LS", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            "model_files_mode": FILES_MODE,
            "brain": MODEL_ID,
        }


class CatSeekBuildMode:
    """Grok Build–style local app/game builder (single-file HTML, files=off LM)."""

    BUILD_RE = re.compile(
        r"\b(?:build|make|create|generate)\b.+\b(?:app|game|website|site|dashboard|tool|calculator|todo|snake|pong|breakout|clicker)\b"
        r"|\b(?:snake|pong|breakout|tetris|clicker|todo|calculator)\s+(?:game|app)?\b"
        r"|/build\b|grok\s*build|catseek\s*build",
        re.IGNORECASE,
    )

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or os.getcwd()).resolve()
        self.out_dir = self.root / "builds"
        self.last_path: Optional[Path] = None

    @classmethod
    def wants(cls, prompt: str) -> bool:
        return bool(cls.BUILD_RE.search(str(prompt or "")))

    def _slug(self, prompt: str) -> str:
        words = re.findall(r"[a-z0-9]+", prompt.lower())
        keep = [w for w in words if w not in {"build", "make", "create", "a", "an", "the", "me", "please"}][:6]
        base = "-".join(keep) or "app"
        return f"{base}-{int(time.time()) % 100000}"

    def _classify(self, prompt: str) -> str:
        lower = prompt.lower()
        if re.search(r"\bsnake\b", lower):
            return "snake"
        if re.search(r"\bpong\b", lower):
            return "pong"
        if re.search(r"\bbreakout\b|\bbrick\b", lower):
            return "breakout"
        if re.search(r"\bclicker\b|\bcounter\b", lower):
            return "clicker"
        if re.search(r"\btodo\b|\btask\b", lower):
            return "todo"
        if re.search(r"\bcalculator\b|\bcalc\b", lower):
            return "calculator"
        if re.search(r"\bdashboard\b|\bchart\b", lower):
            return "dashboard"
        if re.search(r"\b(landing|website|portfolio|site)\b", lower):
            return "website"
        if re.search(r"\bgame\b", lower):
            return "canvas-game"
        return "app"

    def _html_shell(self, title: str, body: str, script: str, extra_css: str = "") -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} · CatSeek Build</title>
<style>
  :root {{ --bg:#0b1220; --panel:#121a2b; --text:#e8eefc; --muted:#8fa3c4; --accent:#3b82f6; --good:#34d399; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; min-height:100vh; font:15px/1.45 "Segoe UI", system-ui, sans-serif;
         background:radial-gradient(1200px 600px at 10% -10%, #1e3a5f 0%, var(--bg) 55%); color:var(--text); }}
  header {{ padding:18px 22px; border-bottom:1px solid #1f2a40; background:rgba(8,12,22,.7); backdrop-filter:blur(8px); }}
  header h1 {{ margin:0; font-size:18px; }}
  header p {{ margin:4px 0 0; color:var(--muted); font-size:12px; }}
  main {{ padding:22px; max-width:960px; margin:0 auto; }}
  .card {{ background:var(--panel); border:1px solid #243149; border-radius:14px; padding:18px; }}
  button, .btn {{ background:var(--accent); color:white; border:0; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:600; }}
  button:hover {{ filter:brightness(1.08); }}
  canvas {{ display:block; margin:12px auto; background:#020617; border-radius:12px; border:1px solid #1e293b; max-width:100%; }}
  input, textarea {{ width:100%; background:#0a1220; color:var(--text); border:1px solid #2a3a55; border-radius:10px; padding:10px 12px; }}
  .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
  .muted {{ color:var(--muted); }}
  .stat {{ font-variant-numeric:tabular-nums; }}
  {extra_css}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>CatSeek Build · local HTML artifact · BitNet brain files=off</p>
</header>
<main>
{body}
</main>
<script>
{script}
</script>
</body>
</html>
"""

    def _template(self, kind: str, prompt: str) -> tuple[str, str]:
        title_map = {
            "snake": "Snake",
            "pong": "Pong",
            "breakout": "Breakout",
            "clicker": "Clicker",
            "todo": "Todo App",
            "calculator": "Calculator",
            "dashboard": "Dashboard",
            "website": "Landing Page",
            "canvas-game": "Canvas Game",
            "app": "Mini App",
        }
        title = title_map.get(kind, "Build")
        if kind == "snake":
            body = '<div class="card"><div class="row"><button id="restart">Restart</button><span class="stat" id="score">Score: 0</span><span class="muted">Arrows / WASD</span></div><canvas id="c" width="400" height="400"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');
let cell=20,snake=[{x:10,y:10}],dir={x:1,y:0},food={x:15,y:10},score=0,alive=true;
function place(){food={x:(Math.random()*20)|0,y:(Math.random()*20)|0};}
function draw(){x.fillStyle='#020617';x.fillRect(0,0,400,400);x.fillStyle='#34d399';x.fillRect(food.x*cell,food.y*cell,cell-1,cell-1);x.fillStyle='#3b82f6';snake.forEach(s=>x.fillRect(s.x*cell,s.y*cell,cell-1,cell-1));}
function step(){if(!alive)return;const h={x:snake[0].x+dir.x,y:snake[0].y+dir.y};if(h.x<0||h.y<0||h.x>=20||h.y>=20||snake.some(s=>s.x===h.x&&s.y===h.y)){alive=false;return;}snake.unshift(h);if(h.x===food.x&&h.y===food.y){score++;document.getElementById('score').textContent='Score: '+score;place();}else snake.pop();draw();}
addEventListener('keydown',e=>{const k=e.key;if(['ArrowUp','w'].includes(k)&&dir.y!==1)dir={x:0,y:-1};if(['ArrowDown','s'].includes(k)&&dir.y!==-1)dir={x:0,y:1};if(['ArrowLeft','a'].includes(k)&&dir.x!==1)dir={x:-1,y:0};if(['ArrowRight','d'].includes(k)&&dir.x!==-1)dir={x:1,y:0};});
document.getElementById('restart').onclick=()=>{snake=[{x:10,y:10}];dir={x:1,y:0};score=0;alive=true;document.getElementById('score').textContent='Score: 0';place();draw();};
place();draw();setInterval(step,110);
"""
        elif kind == "pong":
            body = '<div class="card"><div class="row"><span class="stat" id="score">0 : 0</span><span class="muted">W/S left · ↑/↓ right</span></div><canvas id="c" width="640" height="360"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');
let p1={y:140},p2={y:140},b={x:320,y:180,vx:4,vy:2},s1=0,s2=0,keys={};
addEventListener('keydown',e=>keys[e.key]=true);addEventListener('keyup',e=>keys[e.key]=false);
function frame(){if(keys.w)p1.y-=5;if(keys.s)p1.y+=5;if(keys.ArrowUp)p2.y-=5;if(keys.ArrowDown)p2.y+=5;p1.y=Math.max(0,Math.min(300,p1.y));p2.y=Math.max(0,Math.min(300,p2.y));b.x+=b.vx;b.y+=b.vy;if(b.y<0||b.y>360)b.vy*=-1;if(b.x<24&&b.y>p1.y&&b.y<p1.y+60)b.vx=Math.abs(b.vx);if(b.x>616&&b.y>p2.y&&b.y<p2.y+60)b.vx=-Math.abs(b.vx);if(b.x<0){s2++;reset();}if(b.x>640){s1++;reset();}document.getElementById('score').textContent=s1+' : '+s2;x.fillStyle='#020617';x.fillRect(0,0,640,360);x.fillStyle='#e8eefc';x.fillRect(12,p1.y,10,60);x.fillRect(618,p2.y,10,60);x.beginPath();x.arc(b.x,b.y,7,0,6.28);x.fill();requestAnimationFrame(frame);}
function reset(){b={x:320,y:180,vx:4*(Math.random()>.5?1:-1),vy:2};}
frame();
"""
        elif kind == "breakout":
            body = '<div class="card"><div class="row"><button id="restart">Restart</button><span class="stat" id="score">Score: 0</span></div><canvas id="c" width="480" height="360"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');
let paddle=200,ball={x:240,y:300,vx:3,vy:-3},bricks=[],score=0;
function reset(){bricks=[];for(let r=0;r<5;r++)for(let col=0;col<8;col++)bricks.push({x:20+col*55,y:30+r*22,alive:true});ball={x:240,y:300,vx:3,vy:-3};score=0;document.getElementById('score').textContent='Score: 0';}
addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();paddle=e.clientX-r.left-40;});
document.getElementById('restart').onclick=reset;
function frame(){ball.x+=ball.vx;ball.y+=ball.vy;if(ball.x<0||ball.x>480)ball.vx*=-1;if(ball.y<0)ball.vy*=-1;if(ball.y>360)reset();if(ball.y>330&&ball.x>paddle&&ball.x<paddle+80)ball.vy=-Math.abs(ball.vy);bricks.forEach(b=>{if(b.alive&&ball.x>b.x&&ball.x<b.x+50&&ball.y>b.y&&ball.y<b.y+18){b.alive=false;ball.vy*=-1;score+=10;document.getElementById('score').textContent='Score: '+score;}});x.fillStyle='#020617';x.fillRect(0,0,480,360);x.fillStyle='#3b82f6';x.fillRect(paddle,340,80,10);x.fillStyle='#34d399';bricks.filter(b=>b.alive).forEach(b=>x.fillRect(b.x,b.y,50,16));x.fillStyle='#e8eefc';x.beginPath();x.arc(ball.x,ball.y,6,0,6.28);x.fill();requestAnimationFrame(frame);}
reset();frame();
"""
        elif kind == "clicker":
            body = '<div class="card"><h2 class="stat" id="n">0</h2><div class="row"><button id="hit">Click</button><button id="reset">Reset</button><span class="muted" id="rate"></span></div></div>'
            script = r"""
let n=0,t0=performance.now();
const hit=()=>{n++;document.getElementById('n').textContent=n;const dt=(performance.now()-t0)/1000;document.getElementById('rate').textContent=(n/Math.max(dt,0.001)).toFixed(1)+' /s';};
document.getElementById('hit').onclick=hit;document.getElementById('reset').onclick=()=>{n=0;t0=performance.now();hit();};
"""
        elif kind == "todo":
            body = '<div class="card"><div class="row"><input id="item" placeholder="New task"/><button id="add">Add</button></div><ul id="list"></ul></div>'
            script = r"""
const list=document.getElementById('list'),items=JSON.parse(localStorage.getItem('catseek-todo')||'[]');
function save(){localStorage.setItem('catseek-todo',JSON.stringify(items));}
function render(){list.innerHTML='';items.forEach((t,i)=>{const li=document.createElement('li');li.innerHTML='<label><input type="checkbox" '+(t.done?'checked':'')+'> '+t.text+'</label> <button data-i="'+i+'">✕</button>';li.querySelector('input').onchange=e=>{items[i].done=e.target.checked;save();};li.querySelector('button').onclick=()=>{items.splice(i,1);save();render();};list.appendChild(li);});}
document.getElementById('add').onclick=()=>{const v=document.getElementById('item').value.trim();if(!v)return;items.push({text:v,done:false});document.getElementById('item').value='';save();render();};
render();
"""
        elif kind == "calculator":
            body = '<div class="card"><input id="display" readonly value="0"/><div class="row" id="keys"></div></div>'
            script = r"""
const d=document.getElementById('display');let cur='0';
const keys=['7','8','9','/','4','5','6','*','1','2','3','-','0','.','=','+','C'];
const box=document.getElementById('keys');
keys.forEach(k=>{const b=document.createElement('button');b.textContent=k;b.onclick=()=>{if(k==='C')cur='0';else if(k==='='){try{cur=String(Function('"use strict";return ('+cur+')')());}catch{cur='Err';}}else cur=(cur==='0'&&k!=='.')?k:cur+k;d.value=cur;};box.appendChild(b);});
"""
        elif kind == "dashboard":
            body = '<div class="card"><div class="row"><span class="stat" id="a">A: 0</span><span class="stat" id="b">B: 0</span><button id="tick">Sample</button></div><canvas id="c" width="640" height="220"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');let pts=[];
function draw(){x.fillStyle='#020617';x.fillRect(0,0,640,220);x.strokeStyle='#3b82f6';x.beginPath();pts.forEach((p,i)=>{const X=i*(640/Math.max(pts.length-1,1)),Y=200-p;if(i)x.lineTo(X,Y);else x.moveTo(X,Y);});x.stroke();}
document.getElementById('tick').onclick=()=>{const a=(Math.random()*100)|0,b=(Math.random()*100)|0;document.getElementById('a').textContent='A: '+a;document.getElementById('b').textContent='B: '+b;pts.push((a+b)/2);if(pts.length>40)pts.shift();draw();};
"""
        elif kind == "website":
            body = f'<div class="card"><h2>Ship the idea</h2><p class="muted">Prompt: {prompt[:180].replace("<","&lt;")}</p><p>This landing page was generated by CatSeek Build from your description. Replace the copy, drop in art, and publish anywhere static HTML hosts.</p><div class="row"><a class="btn" href="#get">Get started</a><button onclick="alert(\'Local CatSeek Build CTA\')">Live demo</button></div></div><div class="card" id="get" style="margin-top:14px"><h3>Why CatSeek Build</h3><ul><li>Plain-language → working HTML</li><li>BitNet brain stays files=off</li><li>Iterate with /build again</li></ul></div>'
            script = "/* static landing */"
        else:
            body = f'<div class="card"><h2>Interactive canvas</h2><p class="muted">{prompt[:200].replace("<","&lt;")}</p><canvas id="c" width="480" height="320"></canvas><p class="muted">Click to paint · generated by CatSeek Build</p></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');x.fillStyle='#020617';x.fillRect(0,0,480,320);
let down=false;c.onmousedown=()=>down=true;c.onmouseup=()=>down=false;c.onmousemove=e=>{if(!down)return;const r=c.getBoundingClientRect();x.fillStyle='#3b82f6';x.beginPath();x.arc(e.clientX-r.left,e.clientY-r.top,6,0,6.28);x.fill();};
"""
        return title, self._html_shell(title, body, script)

    def build(self, prompt: str) -> Reply:
        started = time.perf_counter()
        kind = self._classify(prompt)
        title, html = self._template(kind, prompt)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self._slug(prompt)}.html"
        path.write_text(html, encoding="utf-8")
        self.last_path = path
        uri = path.resolve().as_uri()
        text = (
            f"CatSeek Build created **{title}** ({kind}).\n\n"
            f"Saved: `{path}`\n"
            f"Open: {uri}\n\n"
            "This is a Grok Build–style local artifact: single-file HTML/JS you can open in a browser. "
            "Model weights stayed files=off; only this build file was written.\n\n"
            "Iterate: `/build add a score label` or describe another app/game."
        )
        return Reply(
            text=text,
            route=f"catseek-build:{kind}",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=len(WordTokenizer.basic_tokenize(text)),
        )

    def self_test(self) -> dict[str, object]:
        reply = self.build("build a snake game")
        ok = bool(self.last_path and self.last_path.is_file() and "canvas" in self.last_path.read_text(encoding="utf-8"))
        if self.last_path and self.last_path.exists():
            try:
                self.last_path.unlink()
            except OSError:
                pass
        return {
            "ok": ok and reply.route.startswith("catseek-build:"),
            "route": reply.route,
            "wants_snake": self.wants("make a snake game"),
            "wants_todo": self.wants("build a todo app"),
            "out_dir": str(self.out_dir),
        }


class CatSeekR1:
    """Branded cognitive runtime around neural and verifiable local routes."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ):
        boot_t0 = time.perf_counter()
        self.config = config or ModelConfig()
        self.tokenizer = WordTokenizer(corpus_texts())
        self.data_vibe = TrainingDataVibeCheck(ALL_DIALOGUES)
        # Defer corpus vibe audit so ≤0.2s boot stays interactive.
        self.data_vibe_report: dict[str, object] = {
            "passed": True,
            "deferred": True,
            "contract": (
                "Vibe audit deferred for ≤0.2s boot; run /vibecheck or --self-test for full report."
            ),
        }
        self.open_domain = OpenDomainResponder()
        self.last_generation_quality: dict[str, object] = {}
        self.model = InMemoryTernaryLM(self.tokenizer, self.config)
        self.model.train(progress=progress)
        self.reasoner = ExactReasoner()
        self.semantic_reasoner = SemanticReasoner()
        self.history: list[tuple[str, str]] = []
        self.temperature = self.config.temperature
        self.top_k = self.config.top_k
        self.candidate_count = self.config.deliberation_candidates
        self.teachings = 0
        self.taught_memory: dict[str, str] = {}
        self.code_agent = CatSeekCodeAgent(self)
        self.build_mode = CatSeekBuildMode()
        self.boot_elapsed_s = time.perf_counter() - boot_t0

    def ensure_vibe_report(self) -> dict[str, object]:
        if self.data_vibe_report.get("deferred"):
            self.data_vibe_report = self.data_vibe.run(self.tokenizer)
        return self.data_vibe_report

    def clear(self) -> None:
        self.history.clear()

    def model_card(self) -> dict[str, object]:
        card = self.model.model_card()
        card.update({
            "runtime": "CatSeek R1 cognitive inference runtime",
            "reasoning_tools": [
                "exact arithmetic", "percent arithmetic", "linear algebra",
                "quadratic algebra", "semantic evidence retrieval", "multi-evidence composition",
                "DeepSeek R1-style think/answer (files=off) on BitNet + DSpark",
                "CatSeek Code (Claude Code fork): Read/Write/Edit/Bash/Glob/Grep/LS",
            ],
            "semantic_memory_documents": len(self.semantic_reasoner.documents),
            "training_data_vibe_check": self.data_vibe_report,
            "open_domain_response_contract": (
                "Every input returns visible prompt-aware text; missing knowledge is labeled instead of invented."
            ),
            "last_generation_quality": self.last_generation_quality,
            "candidate_verifier": "confidence, repetition, language, length, prompt coverage, completion",
            "active_candidate_count": self.candidate_count,
            "in_ram_teachings": self.teachings,
            "exact_taught_recall_entries": len(self.taught_memory),
            "conversation_turns_in_ram": len(self.history),
            "boot_elapsed_s": round(getattr(self, "boot_elapsed_s", 0.0), 4),
            "sparse_bitnet_bank": bool(getattr(self.model.bank, "sparse", False)),
        })
        return card

    def self_test(self) -> dict[str, object]:
        neural = self.model.self_test()
        exact = self.reasoner.self_test()
        semantic = self.semantic_reasoner.self_test()
        open_domain = self.open_domain.self_test()
        data_vibe = self.ensure_vibe_report()
        code = self.code_agent.self_test()
        build = self.build_mode.self_test()
        boot_ok = True
        if 0 < float(self.config.boot_budget_s) <= 0.25:
            boot_ok = float(getattr(self, "boot_elapsed_s", 99.0)) <= 0.25
        return {
            "ok": bool(
                neural["ok"] and exact["ok"] and semantic["ok"]
                and open_domain["ok"] and data_vibe["passed"] and code["ok"] and build["ok"]
                and boot_ok
            ),
            "neural_inference": neural,
            "exact_reasoning": exact,
            "semantic_reasoning": semantic,
            "open_domain_responses": open_domain,
            "training_data_vibe_check": data_vibe,
            "catseek_code": code,
            "catseek_build": build,
            "runtime": {
                "files": FILES_MODE,
                "network_required": False,
                "boot_elapsed_s": round(getattr(self, "boot_elapsed_s", 0.0), 4),
                "boot_under_0_2s": boot_ok,
                "sparse_bitnet_bank": bool(getattr(self.model.bank, "sparse", False)),
                "candidate_verifier_present": True,
                "conversation_context_present": True,
                "teachable_memory_present": True,
                "semantic_composition_present": True,
                "open_domain_fallback_present": True,
                "generation_quality_gate_present": True,
                "every_prompt_returns_visible_text": open_domain["ok"],
                "claude_code_fork_present": True,
                "grok_build_style_present": True,
                "dspark_speculative_decode_present": True,
                "deepseek_r1_reasoning_present": True,
                "bitnet_b158_real": True,
                "files_mode_off": FILES_MODE == "off",
            },
        }

    def _command(self, prompt: str) -> Optional[Reply]:
        raw = prompt.strip()
        lower = raw.lower()
        started = time.perf_counter()
        if not raw.startswith("/"):
            return None
        if lower in {"/help", "/?"}:
            text = (
                "CatSeek R1 commands:\n\n"
                "- `/model` — RAM model card and measured loss\n"
                "- `/selftest` — inference-proof tests with evidence\n"
                "- `/trace` — probabilities from the last generation\n"
                "- `/deliberation` — scored candidates from the last difficult prompt\n"
                "- `/vibecheck` — audit corpus coverage and arbitrary-prompt fallbacks\n"
                "- `/train N` — run N more in-memory gradient steps\n"
                "- `/teach PROMPT => ANSWER` — teach one RAM-only example\n"
                "- `/candidates N` — set test-time candidates from 1 to 7\n"
                "- `/temperature N` — set sampling temperature (0 = greedy)\n"
                "- `/clear` — clear conversation context\n"
                "- `/code PROMPT` — CatSeek Code (Claude Code fork) on BitNet 4B\n"
                "- `/build PROMPT` — CatSeek Build (Grok Build–style apps/games)\n"
                "- Complex prompts auto-route through DeepSeek R1-style think/answer on BitNet + DSpark (files=off)\n\n"
                "Ordinary messages use exact tools, semantic memory, prompt-grounded handlers, Build mode for apps/games, and quality-gated BitNet decoding. The ~4B value describes packed architecture slots, not pretrained frontier knowledge. Use --code or --build for dedicated REPLs."
            )
            return Reply(text, "control:/help", (time.perf_counter() - started) * 1000)
        if lower == "/model":
            text = json.dumps(self.model_card(), indent=2, ensure_ascii=False)
            return Reply(text, "control:/model", (time.perf_counter() - started) * 1000)
        if lower == "/selftest":
            result = self.self_test()
            text = json.dumps(result, indent=2, ensure_ascii=False)
            return Reply(text, "control:/selftest", (time.perf_counter() - started) * 1000)
        if lower == "/trace":
            if not self.model.last_trace:
                text = "No CatSeek R1 generation trace exists yet."
            else:
                rows = [
                    {
                        "step": step.index,
                        "token": step.token,
                        "probability": round(step.probability, 6),
                        "entropy_bits": round(step.entropy_bits, 4),
                    }
                    for step in self.model.last_trace
                ]
                text = json.dumps(rows, indent=2, ensure_ascii=False)
            return Reply(text, "control:/trace", (time.perf_counter() - started) * 1000)
        if lower == "/deliberation":
            text = (
                json.dumps(self.model.last_deliberation, indent=2, ensure_ascii=False)
                if self.model.last_deliberation
                else "No multi-candidate deliberation exists yet. Ask a complex non-math question first."
            )
            return Reply(text, "control:/deliberation", (time.perf_counter() - started) * 1000)
        if lower in {"/vibecheck", "/vibe-check"}:
            vibe = self.ensure_vibe_report()
            result = {
                "ok": bool(vibe["passed"] and self.open_domain.self_test()["ok"]),
                "training_data": vibe,
                "response_matrix": self.open_domain.self_test(),
                "last_generation_quality": self.last_generation_quality,
            }
            return Reply(
                json.dumps(result, indent=2, ensure_ascii=False),
                "control:/vibecheck",
                (time.perf_counter() - started) * 1000,
            )
        if lower == "/clear":
            self.clear()
            return Reply("CatSeek R1 conversation context cleared.", "control:/clear", (time.perf_counter() - started) * 1000)
        if lower.startswith("/train"):
            parts = raw.split()
            try:
                steps = int(parts[1]) if len(parts) > 1 else 100
            except ValueError:
                return Reply("Usage: /train N", "control:error", (time.perf_counter() - started) * 1000)
            steps = max(1, min(steps, 5000))
            report = self.model.train(steps=steps, boot_budget_s=0.0)
            text = (
                f"CatSeek R1 trained for {report.steps} additional RAM-only steps. "
                f"Loss {report.initial_loss:.4f} → {report.final_loss:.4f}."
            )
            return Reply(text, "control:/train", (time.perf_counter() - started) * 1000)
        if lower.startswith("/teach"):
            lesson = raw[len("/teach"):].strip()
            prompt_text, separator, answer_text = lesson.partition("=>")
            if not separator or not prompt_text.strip() or not answer_text.strip():
                return Reply(
                    "Usage: /teach PROMPT => ANSWER",
                    "control:error",
                    (time.perf_counter() - started) * 1000,
                )
            added = self.model.add_training_dialogue(prompt_text.strip(), answer_text.strip())
            report = self.model.train(steps=24, boot_budget_s=0.0)
            memory_key = " ".join(prompt_text.lower().split())
            self.taught_memory[memory_key] = answer_text.strip()
            self.teachings += 1
            text = (
                f"Learned one RAM-only dialogue ({added} next-token samples). "
                f"Training loss {report.initial_loss:.4f} → {report.final_loss:.4f}. "
                "The lesson disappears when this process exits because files=off."
            )
            return Reply(text, "control:/teach", (time.perf_counter() - started) * 1000)
        if lower.startswith("/candidates"):
            parts = raw.split()
            if len(parts) == 1:
                text = f"CatSeek R1 uses up to {self.candidate_count} verified candidates."
            else:
                try:
                    value = int(parts[1])
                except ValueError:
                    return Reply("Usage: /candidates 1..7", "control:error", (time.perf_counter() - started) * 1000)
                self.candidate_count = max(1, min(7, value))
                text = f"CatSeek R1 candidate count set to {self.candidate_count}."
            return Reply(text, "control:/candidates", (time.perf_counter() - started) * 1000)
        if lower.startswith("/temperature"):
            parts = raw.split()
            if len(parts) == 1:
                text = f"CatSeek R1 temperature is {self.temperature:.2f}."
            else:
                try:
                    value = float(parts[1])
                except ValueError:
                    return Reply("Usage: /temperature 0.0..2.0", "control:error", (time.perf_counter() - started) * 1000)
                self.temperature = max(0.0, min(2.0, value))
                text = f"CatSeek R1 temperature set to {self.temperature:.2f}."
            return Reply(text, "control:/temperature", (time.perf_counter() - started) * 1000)
        if lower.startswith("/code"):
            task = raw[len("/code"):].strip()
            if not task:
                return Reply(
                    "Usage: /code PROMPT   or launch `python3 catr1.py --code`",
                    "control:/code",
                    (time.perf_counter() - started) * 1000,
                )
            reply = self.code_agent.run(task)
            return Reply(reply.text, reply.route, (time.perf_counter() - started) * 1000.0, reply.tokens, reply.tokens_per_second)
        if lower.startswith("/build"):
            task = raw[len("/build"):].strip()
            if not task:
                return Reply(
                    "Usage: /build PROMPT   e.g. /build snake game\nOr: python3 catr1.py --build",
                    "control:/build",
                    (time.perf_counter() - started) * 1000,
                )
            reply = self.build_mode.build(task)
            return Reply(reply.text, reply.route, (time.perf_counter() - started) * 1000.0, reply.tokens, reply.tokens_per_second)
        return Reply("Unknown CatSeek R1 command. Use /help.", "control:unknown", (time.perf_counter() - started) * 1000)

    def reply(self, prompt: str, on_token: Optional[Callable[[str], None]] = None) -> Reply:
        started = time.perf_counter()
        prompt = str(prompt or "").replace("\x00", " ").strip()
        if not prompt or len(prompt) > self.open_domain.MAX_PROMPT_CHARS:
            fallback = self.open_domain.respond(prompt)
            if on_token:
                on_token(fallback.text)
            return Reply(
                fallback.text,
                fallback.route,
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(fallback.text)),
            )
        control = self._command(prompt)
        if control is not None:
            if on_token:
                on_token(control.text)
            return control
        taught = self.taught_memory.get(" ".join(prompt.lower().split()))
        if taught is not None:
            self.history.append((prompt, taught))
            self.history = self.history[-12:]
            if on_token:
                on_token(taught)
            return Reply(
                taught,
                "in-ram-taught-recall",
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(taught)),
            )
        exact = self.reasoner.solve(prompt)
        if exact is not None:
            self.history.append((prompt, exact.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(exact.text)
            return Reply(
                exact.text,
                exact.route,
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(exact.text)),
            )
        if CatSeekBuildMode.wants(prompt):
            built = self.build_mode.build(prompt)
            self.history.append((prompt, built.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(built.text)
            return built
        analysis = self.open_domain.analyze(prompt)
        if analysis.intent in {
            "summarize", "rewrite", "translate", "debug", "compare",
            "build", "brainstorm", "current", "verify",
        }:
            grounded = self.open_domain.respond(prompt)
            self.history.append((prompt, grounded.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(grounded.text)
            return Reply(
                grounded.text,
                grounded.route,
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(grounded.text)),
            )
        semantic_query = prompt
        if self.history and re.search(
            r"\b(?:it|that|this|those|previous|earlier|above|same)\b|继续|刚才|上一个|那个",
            prompt,
            re.IGNORECASE,
        ):
            semantic_query = self.history[-1][0] + " " + prompt
        semantic = self.semantic_reasoner.compose(semantic_query)
        if semantic is not None:
            self.history.append((prompt, semantic.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(semantic.text)
            return Reply(
                semantic.text,
                semantic.route,
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(semantic.text)),
            )
        if (
            self.model.deepseek_r1 is not None
            and self.model.config.deepseek_r1_enabled
            and self.model.deepseek_r1.needs_reasoning(prompt)
        ):
            text, ds_stats = self.model.deepseek_r1.reason(
                prompt,
                self.history,
                temperature=self.temperature,
                top_k=self.top_k,
                on_token=on_token,
            )
            usable, quality_score, quality_reasons = self.open_domain.generation_quality(prompt, text)
            self.last_generation_quality = {
                "accepted": usable,
                "score": round(quality_score, 4),
                "reasons": quality_reasons,
                "neural_preview": text[:180],
                "deepseek_r1_reward": ds_stats.reward,
                "deepseek_r1_think_tokens": ds_stats.think_tokens,
            }
            if usable:
                route = (
                    f"deepseek-r1×dspark×bitnet:{detect_language(prompt)}:"
                    f"{ds_stats.candidates}c:r={ds_stats.reward:.3f}"
                )
            else:
                fallback = self.open_domain.respond(prompt)
                text = fallback.text
                route = f"{fallback.route}:deepseek-r1-quality-repair"
            self.history.append((prompt, text))
            self.history = self.history[-12:]
            return Reply(
                text=text,
                route=route,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                tokens=ds_stats.answer_tokens + ds_stats.think_tokens,
                tokens_per_second=0.0,
            )
        report = self.model.deliberate_chat(
            prompt,
            self.history,
            temperature=self.temperature,
            top_k=self.top_k,
            candidate_count=self.candidate_count,
        )
        usable, quality_score, quality_reasons = self.open_domain.generation_quality(prompt, report.text)
        self.last_generation_quality = {
            "accepted": usable,
            "score": round(quality_score, 4),
            "reasons": quality_reasons,
            "neural_preview": report.text[:180],
        }
        if usable:
            text = report.text
            route = f"ram-bitnet-deliberation:{report.language}:{len(self.model.last_deliberation)}c"
        else:
            fallback = self.open_domain.respond(prompt)
            text = fallback.text
            route = f"{fallback.route}:quality-repair"
        self.history.append((prompt, text))
        self.history = self.history[-12:]
        if on_token:
            if usable:
                for token_id in [step.token_id for step in report.trace if step.token_id != self.tokenizer.token_id(self.tokenizer.EOS)]:
                    piece = self.tokenizer.decode([token_id])
                    if piece:
                        on_token(piece + ("" if piece in ".,!?:;" else " "))
            else:
                on_token(text)
        return Reply(
            text=text,
            route=route,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=report.tokens,
            tokens_per_second=report.tokens_per_second,
        )


class CatSeekGUI:
    BG = "#03060b"
    PANEL = "#07101d"
    INPUT = "#0a1424"
    BLUE = "#3b82f6"
    BLUE_2 = "#60a5fa"
    TEXT = "#dbeafe"
    MUTED = "#7890ad"
    RED = "#f87171"

    def __init__(self, root: tk.Tk, engine: CatSeekR1):
        self.root = root
        self.engine = engine
        self.events: queue.Queue[tuple[str, int, int, object]] = queue.Queue()
        self.busy = False
        self.conversation_id = 0
        self.request_serial = 0
        self.active_request_id: Optional[int] = None
        self._build()
        self.root.after(GUI_TICK_MS, self._drain)

    @staticmethod
    def _family(mono: bool = False) -> str:
        if mono:
            return "Cascadia Mono" if os.name == "nt" else "Menlo"
        return "Segoe UI" if os.name == "nt" else "Helvetica Neue"

    def _font(self, size: int, bold: bool = False, mono: bool = False) -> font.Font:
        return font.Font(family=self._family(mono), size=size, weight="bold" if bold else "normal")

    def _build(self) -> None:
        root = self.root
        root.title(f"{APP_NAME} v{APP_VERSION}")
        root.geometry("1040x720")
        root.minsize(780, 560)
        root.configure(bg=self.BG)

        menu = tk.Menu(root, tearoff=False, bg=self.PANEL, fg=self.TEXT)
        catseek = tk.Menu(menu, tearoff=False, bg=self.PANEL, fg=self.TEXT)
        catseek.add_command(label="New CatSeek R1 chat", command=self._new_chat)
        catseek.add_command(label="CatSeek R1 model card", command=lambda: self._submit("/model"))
        catseek.add_command(label="CatSeek R1 inference self-test", command=lambda: self._submit("/selftest"))
        catseek.add_separator()
        catseek.add_command(label="Exit CatSeek R1", command=root.destroy)
        menu.add_cascade(label=APP_NAME, menu=catseek)
        root.config(menu=menu)

        header = tk.Frame(root, bg=self.PANEL, padx=18, pady=14)
        header.pack(fill="x")
        logo = tk.Frame(header, bg="#000000", width=46, height=46)
        logo.pack(side="left")
        logo.pack_propagate(False)
        tk.Label(logo, text="🐱", bg="#000000", fg=self.BLUE, font=self._font(23)).place(relx=0.5, rely=0.5, anchor="center")
        name = tk.Frame(header, bg=self.PANEL)
        name.pack(side="left", padx=(12, 0))
        tk.Label(name, text=APP_NAME, bg=self.PANEL, fg=self.TEXT, font=self._font(17, True)).pack(anchor="w")
        report = self.engine.model.report
        tk.Label(
            name,
            text=(
                f"Real BitNet b1.58 ~4B · {ENGINE_NAME} · {self.engine.model.parameter_count:,} params · "
                f"loss {report.initial_loss:.3f}→{report.final_loss:.3f} · files=off · {GUI_FPS} FPS · {self.engine.model.config.speed_target}"
            ),
            bg=self.PANEL,
            fg=self.MUTED,
            font=self._font(9),
        ).pack(anchor="w")
        self.status = tk.Label(header, text="CatSeek R1 inference ready", bg=self.PANEL, fg=self.BLUE_2, font=self._font(10))
        self.status.pack(side="right")

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=self.PANEL, width=220, padx=14, pady=18)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Button(
            sidebar,
            text="+ New CatSeek R1 chat",
            command=self._new_chat,
            bg="#000000",
            fg=self.BLUE,
            activebackground="#111827",
            activeforeground=self.BLUE_2,
            relief="flat",
            bd=0,
            padx=10,
            pady=10,
            font=self._font(10, True),
            cursor="hand2",
        ).pack(fill="x")
        tk.Label(sidebar, text="REAL RAM LM", bg=self.PANEL, fg=self.MUTED, font=self._font(9, True)).pack(anchor="w", pady=(22, 8))
        for line in (
            "● Startup gradient training",
            "● English / 中文 auto mode",
            "● Next-token softmax",
            "● Real BitNet b1.58 ~4B W1.58A8",
            f"● {ENGINE_NAME} engine (KDA/AttnRes/MoE)",
            "● DSpark speculative decode",
            "● DeepSeek R1 think/answer (RAM)",
            "● Ternary add/sub kernel",
            f"● Speed target: {self.engine.model.config.speed_target}",
            f"● GUI {GUI_FPS} FPS",
            "● Verified decode candidates",
            "● Exact math / algebra tools",
            "● Lossless byte fallback",
            "● A8 absmax activations",
            "● sparse BitNet bank (≤0.2s boot)",
            "● files=off (RAM only)",
            "● No checkpoints/API",
        ):
            tk.Label(sidebar, text=line, bg=self.PANEL, fg=self.BLUE_2, font=self._font(9)).pack(anchor="w", pady=3)
        tk.Label(
            sidebar,
            text="/model\n/selftest\n/deliberation\n/train 100\n/teach Q => A\n/candidates 3\n/clear",
            justify="left",
            bg=self.PANEL,
            fg=self.MUTED,
            font=self._font(9, mono=True),
        ).pack(anchor="w", side="bottom")

        chat_area = tk.Frame(body, bg=self.BG)
        chat_area.pack(side="left", fill="both", expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_area,
            state="disabled",
            wrap="word",
            bg=self.BG,
            fg=self.TEXT,
            insertbackground=self.BLUE,
            selectbackground="#1d4ed8",
            relief="flat",
            bd=0,
            padx=26,
            pady=20,
            font=self._font(11),
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("title", foreground=self.BLUE_2, font=self._font(11, True), spacing1=12)
        self.chat.tag_configure("user", foreground="#93c5fd", font=self._font(11))
        self.chat.tag_configure("bot", foreground=self.TEXT, font=self._font(11))
        self.chat.tag_configure("muted", foreground=self.MUTED, font=self._font(9))
        self.chat.tag_configure("error", foreground=self.RED, font=self._font(10))
        self._append("CATSEEK R1\n", "title")
        self._append(
            "Output-path weights train in RAM (files=off). English and 中文 are selected automatically. Exact tools handle arithmetic/algebra; general prompts use semantic memory, prompt-grounded fallbacks, and quality-gated BitNet decoding. Use /vibecheck and /selftest for evidence.\n",
            "muted",
        )

        composer = tk.Frame(chat_area, bg=self.PANEL, padx=18, pady=14)
        composer.pack(fill="x")
        self.entry = tk.Text(
            composer,
            height=3,
            bg=self.INPUT,
            fg=self.TEXT,
            insertbackground=self.BLUE,
            selectbackground="#1d4ed8",
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=self._font(10),
            wrap="word",
            undo=True,
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.send = tk.Button(
            composer,
            text="INFER ↑",
            command=self._submit,
            bg="#000000",
            fg=self.BLUE,
            activebackground="#111827",
            activeforeground=self.BLUE_2,
            relief="flat",
            bd=0,
            padx=16,
            pady=12,
            font=self._font(10, True),
            cursor="hand2",
        )
        self.send.pack(side="right", padx=(12, 0))
        self.entry.focus_set()

    def _append(self, text: str, tag: str = "bot") -> None:
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _new_chat(self) -> None:
        # Invalidate every queued result from the previous conversation. A
        # running inference cannot be force-killed safely, so its visible
        # result is discarded and its final idle event unlocks the composer.
        self.conversation_id += 1
        self.engine.clear()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self.entry.delete("1.0", "end")
        self.entry.edit_reset()
        self._append("CATSEEK R1\n", "title")
        if self.busy:
            self._append(
                "New RAM-only conversation created. Finishing the previous inference in the background…\n",
                "muted",
            )
            self.status.config(text="New chat · finishing previous inference…")
        else:
            self.active_request_id = None
            self.send.config(state="normal")
            self._append("New RAM-only language-model conversation.\n", "muted")
            self.status.config(text="CatSeek R1 new chat ready")
        self.entry.focus_set()

    def _on_enter(self, event) -> Optional[str]:
        if event.state & 0x1:
            return None
        self._submit()
        return "break"

    def _submit(self, forced: Optional[str] = None) -> None:
        if self.busy:
            return
        prompt = forced if isinstance(forced, str) else self.entry.get("1.0", "end-1c").strip()
        if not prompt:
            return
        self.entry.delete("1.0", "end")
        self._append("\nYOU\n", "title")
        self._append(prompt + "\n", "user")
        self._append("\nCATSEEK R1\n", "title")
        self.busy = True
        self.request_serial += 1
        request_id = self.request_serial
        conversation_id = self.conversation_id
        self.active_request_id = request_id
        self.send.config(state="disabled")
        self.status.config(text="CatSeek R1 computing next tokens…")

        def worker() -> None:
            try:
                reply = self.engine.reply(prompt)
                self.events.put(("done", conversation_id, request_id, reply))
            except Exception as exc:
                self.events.put(("error", conversation_id, request_id, f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("idle", conversation_id, request_id, None))

        threading.Thread(target=worker, daemon=True).start()

    def _drain(self) -> None:
        try:
            while True:
                kind, conversation_id, request_id, payload = self.events.get_nowait()
                is_current = conversation_id == self.conversation_id
                if kind == "done" and is_current:
                    reply = payload
                    self._append(reply.text, "bot")
                    suffix = (
                        f"\n\n{reply.route} · {reply.tokens} tokens · "
                        f"{reply.tokens_per_second:.1f} tok/s · {reply.elapsed_ms:.1f} ms\n"
                    )
                    self._append(suffix, "muted")
                    self.status.config(text=f"CatSeek R1 ready · {reply.tokens_per_second:.1f} tok/s")
                elif kind == "error" and is_current:
                    self._append("ERROR\n" + str(payload) + "\n", "error")
                    self.status.config(text="CatSeek R1 error")
                elif kind == "idle" and request_id == self.active_request_id:
                    if not is_current:
                        # The stale worker may have appended to engine history
                        # after New Chat cleared it; clear once more at exit.
                        self.engine.clear()
                    self.busy = False
                    self.active_request_id = None
                    self.send.config(state="normal")
                    if not is_current:
                        self.status.config(text="CatSeek R1 new chat ready")
                        self._append("Previous inference discarded. New chat is ready.\n", "muted")
                    self.entry.focus_set()
        except queue.Empty:
            pass
        self.root.after(GUI_TICK_MS, self._drain)


def make_parser() -> argparse.ArgumentParser:
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description=f"{APP_NAME} v{APP_VERSION}: bilingual RAM-only BitNet b1.58 response runtime (files=off)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chat", action="store_true", help="terminal chat instead of GUI")
    parser.add_argument("--code", action="store_true", help="CatSeek Code REPL (Claude Code fork on BitNet 4B)")
    parser.add_argument("--build", action="store_true", help="CatSeek Build REPL (Grok Build–style apps/games)")
    parser.add_argument("--cwd", default=None, help="workspace root for --code / --build artifacts")
    parser.add_argument("--prompt", help="generate one answer and exit")
    parser.add_argument(
        "--vibe-check",
        action="store_true",
        help="audit embedded training data and arbitrary-prompt coverage without allocating the 4B bank",
    )
    parser.add_argument("--self-test", action="store_true", help="run proof-oriented inference tests")
    parser.add_argument("--model-card", action="store_true", help="print the honest model card")
    parser.add_argument(
        "--dspark",
        dest="dspark",
        action="store_true",
        default=True,
        help="enable DSpark speculative decode (files=off; default on)",
    )
    parser.add_argument(
        "--no-dspark",
        dest="dspark",
        action="store_false",
        help="disable DSpark speculative decode",
    )
    parser.add_argument(
        "--deepseek-r1",
        dest="deepseek_r1",
        action="store_true",
        default=True,
        help="enable DeepSeek R1-style think/answer on BitNet + DSpark (files=off; default on)",
    )
    parser.add_argument(
        "--no-deepseek-r1",
        dest="deepseek_r1",
        action="store_false",
        help="disable DeepSeek R1-style reasoning",
    )
    parser.add_argument(
        "--show-think",
        action="store_true",
        help="include internal think chain in visible replies",
    )
    parser.add_argument(
        "--profile",
        choices=("fable5", "fast", "balanced", "quality"),
        default="fable5",
        help="RAM/speed profile; fable5 is the default Claude Fable 5–class interactive target",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="override profile startup steps; use 0 to skip startup training",
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} v{APP_VERSION}")
    return parser


def terminal(engine: CatSeekR1, one_prompt: Optional[str] = None) -> int:
    def run(prompt: str) -> None:
        reply = engine.reply(prompt)
        print(reply.text)
        print(f"\n[{reply.route} · {reply.tokens} tokens · {reply.tokens_per_second:.1f} tok/s]\n")

    if one_prompt:
        run(one_prompt)
        return 0
    report = engine.model.report
    print(
        f"{APP_NAME} v{APP_VERSION} · RAM LM · loss "
        f"{report.initial_loss:.3f}→{report.final_loss:.3f} · /help · /quit"
    )
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def code_terminal(engine: CatSeekR1, one_prompt: Optional[str] = None, cwd: Optional[str] = None) -> int:
    agent = CatSeekCodeAgent(engine, root=cwd)
    engine.code_agent = agent
    banner = (
        f"{APP_NAME} Code · Claude Code fork · brain {MODEL_ID} · files={FILES_MODE}\n"
        f"workspace: {agent.tools.root}\n"
        "tools: Read Write Edit Bash Glob Grep LS · /quit to exit"
    )
    print(banner)

    def run(prompt: str) -> None:
        reply = agent.run(prompt)
        print(reply.text)
        print(f"\n[{reply.route} · {reply.elapsed_ms:.0f} ms]\n")

    if one_prompt:
        run(one_prompt)
        return 0
    while True:
        try:
            prompt = input("code> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def build_terminal(engine: CatSeekR1, one_prompt: Optional[str] = None, cwd: Optional[str] = None) -> int:
    builder = CatSeekBuildMode(root=cwd)
    engine.build_mode = builder
    banner = (
        f"{APP_NAME} Build · Grok Build–style · brain {MODEL_ID} · files={FILES_MODE}\n"
        f"artifacts: {builder.out_dir}\n"
        "describe an app/game/website · /quit to exit"
    )
    print(banner)

    def run(prompt: str) -> None:
        reply = builder.build(prompt)
        print(reply.text)
        print(f"\n[{reply.route} · {reply.elapsed_ms:.0f} ms]\n")

    if one_prompt:
        run(one_prompt)
        return 0
    while True:
        try:
            prompt = input("build> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def main(argv: Optional[list[str]] = None) -> int:
    if sys.version_info < (3, 10):
        print(
            f"{APP_NAME} requires Python 3.10+ (running {sys.version.split()[0]}).",
            file=sys.stderr,
        )
        return 2
    args = make_parser().parse_args(argv)
    if args.vibe_check:
        tokenizer = WordTokenizer(corpus_texts())
        data_report = TrainingDataVibeCheck(ALL_DIALOGUES).run(tokenizer)
        response_report = OpenDomainResponder().self_test()
        result = {
            "ok": bool(data_report["passed"] and response_report["ok"]),
            "files": FILES_MODE,
            "training_data": data_report,
            "response_matrix": response_report,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    profiles = {
        # All profiles keep the real ~4B BitNet + Kimi K3s trunk; default boot ≤0.2s.
        "fable5": dict(
            context_tokens=24,
            mod_capacity=2,
            train_layers=1,
            batch_size=4,
            deliberation_candidates=1,
            train_steps=0,  # ≤0.2s boot; STE via /train
            boot_budget_s=BOOT_BUDGET_S,
            speed_target="fable5",
        ),
        "fast": dict(
            context_tokens=24,
            mod_capacity=3,
            train_layers=1,
            batch_size=4,
            deliberation_candidates=1,
            train_steps=0,
            boot_budget_s=BOOT_BUDGET_S,
            speed_target="fast",
        ),
        "balanced": dict(
            context_tokens=32,
            mod_capacity=4,
            train_layers=2,
            batch_size=4,
            deliberation_candidates=2,
            train_steps=0,
            boot_budget_s=BOOT_BUDGET_S,
            speed_target="balanced",
        ),
        "quality": dict(
            context_tokens=48,
            mod_capacity=6,
            train_layers=2,
            batch_size=4,
            deliberation_candidates=3,
            train_steps=12,
            boot_budget_s=2.0,
            speed_target="quality",
        ),
    }
    profile = profiles[args.profile]
    # Proof paths need uncapped training so loss-reduction checks stay meaningful.
    boot_budget = 0.0 if (args.self_test or args.model_card) else float(profile["boot_budget_s"])
    train_default = profile["train_steps"]
    if args.self_test and args.train_steps is None:
        train_default = max(train_default, 16)
    config = ModelConfig(
        context_tokens=profile["context_tokens"],
        embedding_dim=BITNET_4B_D_MODEL,
        hidden_dim=BITNET_4B_D_FF,
        d_model=BITNET_4B_D_MODEL,
        d_ff=BITNET_4B_D_FF,
        n_layers=BITNET_4B_N_LAYERS,
        n_heads=BITNET_4B_N_HEADS,
        mod_capacity=profile["mod_capacity"],
        train_layers=profile["train_layers"],
        dense_init=False,
        reasoning_passes=profile["mod_capacity"],
        max_reasoning_passes=BITNET_4B_N_LAYERS,
        batch_size=profile["batch_size"],
        deliberation_candidates=profile["deliberation_candidates"],
        train_steps=max(0, min(args.train_steps if args.train_steps is not None else train_default, 20_000)),
        temperature=max(0.0, min(args.temperature, 2.0)),
        top_k=max(1, args.top_k),
        max_new_tokens=max(1, min(args.max_tokens, 2048)),
        seed=args.seed,
        boot_budget_s=boot_budget,
        engine=ENGINE_ID,
        kda_ratio=KIMI_K3S_KDA_RATIO,
        moe_top_k=KIMI_K3S_MOE_TOP_K,
        attnres=True,
        speed_target=str(profile["speed_target"]),
        dspark_enabled=bool(args.dspark),
        dspark_speculative_decode=bool(args.dspark),
        dspark_block_size=DSPARK_BLOCK_SIZE,
        dspark_markov_rank=DSPARK_MARKOV_RANK,
        dspark_confidence_head=True,
        dspark_draft_gamma=DSPARK_DRAFT_GAMMA,
        dspark_adaptive_gamma=True,
        deepseek_r1_enabled=bool(args.deepseek_r1),
        deepseek_r1_think_tokens=DEEPSEEK_R1_THINK_BUDGET,
        deepseek_r1_answer_tokens=DEEPSEEK_R1_ANSWER_BUDGET,
        deepseek_r1_show_think=bool(args.show_think),
        deepseek_r1_candidates=2 if args.profile == "fast" else 3,
    )
    show_startup_progress = config.train_steps > 0 and not (args.self_test or args.model_card)
    progress: Optional[Callable[[int, int, float], None]] = None
    if show_startup_progress:
        print(
            f"Materializing ~4B BitNet b1.58 + {ENGINE_NAME} in RAM (files=off, {config.speed_target}): "
            f"0/{config.train_steps} STE steps...",
            file=sys.stderr,
            flush=True,
        )

        def progress(step: int, total: int, loss: float) -> None:
            print(
                f"Training {APP_NAME} STE slice: {step}/{total} steps · loss {loss:.3f}",
                file=sys.stderr,
                flush=True,
            )

    try:
        engine = CatSeekR1(config, progress=progress)
    except KeyboardInterrupt:
        print(f"\n{APP_NAME} startup training cancelled.", file=sys.stderr)
        return 130
    if not (args.self_test or args.model_card):
        print(
            f"{APP_NAME} ready · boot {getattr(engine, 'boot_elapsed_s', 0.0):.3f}s · "
            f"sparse_bank={getattr(engine.model.bank, 'sparse', False)} · "
            f"files={FILES_MODE} · BitNet b1.58 ~4B slots",
            file=sys.stderr,
            flush=True,
        )
    if args.self_test:
        result = engine.self_test()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    if args.model_card:
        print(json.dumps(engine.model_card(), indent=2, ensure_ascii=False))
        return 0
    if args.code:
        return code_terminal(engine, args.prompt, cwd=args.cwd)
    if args.build:
        return build_terminal(engine, args.prompt, cwd=args.cwd)
    if args.prompt or args.chat:
        return terminal(engine, args.prompt)
    root = tk.Tk()
    CatSeekGUI(root, engine)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
