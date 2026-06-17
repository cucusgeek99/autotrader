"""LLM knowledge layer — Claude reads the experiment history and the current
regime, then (a) narrates what's working and why, and (b) proposes new
strategy hypotheses to feed back into the lab.

Optional: requires `pip install anthropic` and ANTHROPIC_API_KEY. If either is
missing, callers get a friendly AdvisorUnavailable error instead of a crash.

Best practices applied (per the claude-api skill):
  * model defaults to claude-opus-4-8
  * stable system prompt carries cache_control for prompt caching
  * adaptive thinking for the analysis
  * structured output via messages.parse() + Pydantic (no brittle JSON parsing)
"""
from __future__ import annotations

import json
import os
from typing import Optional

SYSTEM_PROMPT = """You are a quantitative trading research advisor embedded in a \
self-tuning crypto trading lab. The lab maintains a library of parameterized \
strategies (moving-average crossover, RSI/Bollinger/stochastic mean-reversion, \
breakout, MACD, momentum, supertrend). Each is tuned with Optuna and validated \
walk-forward; every experiment is stored with an out-of-sample (OOS) score, an \
in-sample score, and an overfit gap (in-sample minus OOS).

Your job, given the current market regime and the experiment leaderboard:
1. Read the evidence soberly. A high in-sample score with a large overfit gap is \
a curve-fit, not an edge. Trust OOS, distrust large gaps.
2. Narrate what is and isn't working for this symbol/regime, concisely.
3. Propose concrete, testable hypotheses: which strategy to try, and a tighter \
parameter sub-range worth searching, with a one-line rationale grounded in the \
regime (trending vs ranging, calm vs volatile).
4. Flag risks: overfitting, too few folds, regime fragility, crowded trades.

Be skeptical and specific. This is paper trading research, not financial advice. \
Never promise returns. Prefer fewer, higher-conviction hypotheses over many weak \
ones."""

# Default model — override via config if desired.
DEFAULT_MODEL = "claude-opus-4-8"


class AdvisorUnavailable(RuntimeError):
    """Raised when the LLM layer can't run (no SDK or no API key)."""


def _build_pydantic_models():
    """Defined lazily so pydantic is only required when the advisor is used."""
    from pydantic import BaseModel, Field

    class ParamRange(BaseModel):
        name: str = Field(description="parameter name from the strategy's param_space")
        low: float = Field(description="suggested lower bound to search")
        high: float = Field(description="suggested upper bound to search")

    class Hypothesis(BaseModel):
        strategy: str = Field(description="strategy name to try, e.g. 'rsi_reversion'")
        rationale: str = Field(description="one-line reason grounded in the regime")
        param_ranges: list[ParamRange] = Field(
            description="tighter sub-ranges worth searching for this strategy"
        )
        conviction: str = Field(description="one of: low, medium, high")

    class Advice(BaseModel):
        regime_read: str = Field(description="your read of the current market regime")
        summary: str = Field(description="concise narrative of what is/ isn't working")
        hypotheses: list[Hypothesis]
        risk_flags: list[str] = Field(description="overfitting / fragility / data risks")

    return Advice


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise AdvisorUnavailable(
            "anthropic SDK not installed. Run: pip install 'autotrader[llm]'"
        ) from e
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise AdvisorUnavailable(
            "No ANTHROPIC_API_KEY in environment — set it to use the LLM advisor."
        )
    return anthropic.Anthropic()


def advise(
    symbol: str,
    timeframe: str,
    regime: str,
    leaderboard: list[dict],
    strategy_param_spaces: dict[str, dict],
    model: str = DEFAULT_MODEL,
):
    """Ask Claude to analyze the knowledge base and propose next experiments.

    Returns a validated Advice pydantic object. Raises AdvisorUnavailable if the
    SDK/key are missing.
    """
    client = _client()
    Advice = _build_pydantic_models()

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_regime": regime,
        "leaderboard": leaderboard,
        "strategy_param_spaces": strategy_param_spaces,
    }
    user_content = (
        "Here is the current state of the trading lab. Analyze it and propose "
        "next experiments.\n\n```json\n" + json.dumps(payload, indent=2, default=str)
        + "\n```"
    )

    response = client.messages.parse(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
        output_format=Advice,
    )
    return response.parsed_output
