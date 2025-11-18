# veriflow/semantic/intent_extractor.py
# Hybrid intent extractor: rule-based first, optional LLM refinement.
# Safe to run without an API key; will fallback to rule-based only.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any
import os
import re
import json

try:
    import openai  # optional; required only when use_llm=True
except Exception:
    openai = None


@dataclass
class IntentResult:
    """Container for intent prediction and confidences."""
    intent: Dict[str, bool]
    confidence: Dict[str, float]
    overall_confidence: float
    source: str  # "rule" | "rule+llm" | "llm-only"
    meta: Dict[str, Any]


# --- Keyword inventory (extend as needed) ---
# Note: we keep CN and EN keywords, and add a few common English phrases.
KEYWORDS = {
    "need_schedule": (
        # CN
        "每天", "每小时", "定时", "每日", "工作日", "9点", "07:00",
        # EN direct cues
        "schedule", "cron", "every day", "daily", "every weekday", "each day", "every morning", "weekday",
        "weekdays"
    ),
    "need_email": (
        "email", "mail", "邮箱", "发邮件", "发送邮件",
        # EN phrases
        "email me", "send email", "send it by email", "send to my email", "mail me"
    ),
    "need_http": (
        "http", "api", "请求", "获取", "接口", "天气",
        # EN words
        "fetch", "request", "weather", "get weather", "call api"
    ),
    "need_slack": ("slack", "斯拉克"),
    "need_telegram": ("telegram", "电报", "tg", "Telegram"),
}

# --- English time expressions helpers ---
# Matches: 9, 9am, 9 am, 09:00, 9:00, 9.00, 09h00, etc.
TIME_REGEX = re.compile(
    r"\b((?:[01]?\d|2[0-3])\s*(?:[:h.]\s*[0-5]\d)?\s*(?:am|pm)?)\b",
    flags=re.IGNORECASE,
)

# Common English schedule phrasing (coarse filter; time will be verified by TIME_REGEX)
EN_SCHEDULE_PHRASES = (
    "every day", "daily", "each day", "every weekday", "weekdays",
    "every morning", "every evening", "at", "@"
)


def _rule_score(hit: bool) -> float:
    """Return a conservative confidence for rule hits."""
    return 0.85 if hit else 0.05


def _normalize(text: str) -> str:
    """Lowercase and strip surrounding spaces."""
    return (text or "").strip().lower()


def _contains_any(text: str, keywords) -> bool:
    """Return True if any keyword is a substring of text (case-insensitive)."""
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _has_en_schedule_pattern(text: str) -> bool:
    """
    Detect English schedule expressions like:
    - "every day at 9am", "daily 09:00", "at 9:00 every day", "weekdays at 7:30"
    Strategy: look for a coarse schedule phrase + a time literal nearby.
    """
    t = text.lower()
    if any(p in t for p in EN_SCHEDULE_PHRASES) and TIME_REGEX.search(t):
        return True
    # Accept minimalist "at 9am" / "at 09:00" patterns
    if (" at " in t or " @ " in t) and TIME_REGEX.search(t):
        return True
    return False


def rule_based_intent(prompt: str) -> IntentResult:
    """Extract intent using keyword heuristics + English time patterns."""
    text = _normalize(prompt)
    intent: Dict[str, bool] = {}
    conf: Dict[str, float] = {}
    intent_chain = []

    # need_schedule: keyword OR English-time-pattern
    schedule_hit = _contains_any(text, KEYWORDS["need_schedule"]) or _has_en_schedule_pattern(text)
    intent["need_schedule"] = schedule_hit
    conf["need_schedule"] = _rule_score(schedule_hit)

    # Other booleans via keywords
    for key in ("need_email", "need_http", "need_slack", "need_telegram"):
        hit = _contains_any(text, KEYWORDS[key])
        intent[key] = bool(hit)
        conf[key] = _rule_score(hit)

    if schedule_hit:
        intent_chain.append("rule: matched schedule keywords or English time pattern")
    if _contains_any(text, KEYWORDS["need_email"]):
        intent_chain.append("rule: matched email keywords")
    if _contains_any(text, KEYWORDS["need_http"]):
        intent_chain.append("rule: matched http/api keywords")
    if _contains_any(text, KEYWORDS["need_slack"]):
        intent_chain.append("rule: matched slack keywords")
    if _contains_any(text, KEYWORDS["need_telegram"]):
        intent_chain.append("rule: matched telegram keywords")

    # Overall confidence: average of active keys, fallback to mean
    active = [conf[k] for k, v in intent.items() if v]
    overall = sum(active) / max(1, len(active)) if active else sum(conf.values()) / max(1, len(conf))
    return IntentResult(intent=intent, confidence=conf, overall_confidence=overall, source="rule", meta={"intent_chain": intent_chain},)


def _call_llm(prompt: str, base_intent: Dict[str, bool], model: str = "gpt-4o-mini") -> Optional[Dict]:
    """Call LLM to refine or correct the base intent. Returns a JSON dict or None on failure."""
    if openai is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    # New-style client; if your environment uses the older SDK, adapt accordingly.
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None

    sys_msg = (
        "You are a precise intent extractor for low-code workflow tasks. "
        "Given a natural-language description, output a strict JSON object with boolean flags: "
        "{need_schedule, need_email, need_http, need_slack, need_telegram}. "
        "Also include a 'confidence' object with float scores in [0,1] per key. "
        "Also include a 'chain' array with short natural-language justifications explaining how each intent was inferred."
    )
    user_msg = (
        "Description: " + prompt + "\n"
        "Base intent (from rules): " + json.dumps(base_intent, ensure_ascii=False) + "\n"
        'Respond ONLY with JSON: {"intent": {...}, "confidence": {...}}'
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except Exception:
        return None


def _merge_rule_and_llm(rule_res: IntentResult, llm_json: Dict) -> IntentResult:
    """Fuse rule-based and LLM outputs into a single IntentResult."""
    llm_intent = llm_json.get("intent", {}) if isinstance(llm_json, dict) else {}
    llm_conf = llm_json.get("confidence", {}) if isinstance(llm_json, dict) else {}
    llm_chain = llm_json.get("chain", [])
    merged_chain = rule_res.meta.get("intent_chain", []) + llm_chain

    merged_intent: Dict[str, bool] = {}
    merged_conf: Dict[str, float] = {}

    keys = sorted(set(list(rule_res.intent.keys()) + list(llm_intent.keys())))
    for k in keys:
        r_val = bool(rule_res.intent.get(k, False))
        l_val = bool(llm_intent.get(k, False))
        r_conf = float(rule_res.confidence.get(k, 0.05))
        l_conf = float(llm_conf.get(k, 0.0))

        # If both agree, keep the value and the max confidence.
        # If disagree, take the higher-confidence one with a small penalty.
        if r_val == l_val:
            merged_intent[k] = r_val
            merged_conf[k] = max(r_conf, l_conf) if (r_val or l_val) else (r_conf + l_conf) / 2.0
        else:
            if l_conf > r_conf:
                merged_intent[k] = l_val
                merged_conf[k] = max(l_conf - 0.05, 0.0)
            else:
                merged_intent[k] = r_val
                merged_conf[k] = max(r_conf - 0.05, 0.0)

    active = [merged_conf[k] for k, v in merged_intent.items() if v]
    overall = sum(active) / max(1, len(active)) if active else sum(merged_conf.values()) / max(1, len(merged_conf))
    return IntentResult(intent=merged_intent, confidence=merged_conf, overall_confidence=overall, source="rule+llm", meta={"intent_chain": merged_chain},)


def extract_intent(prompt: str) -> Dict[str, bool]:
    """Legacy API for backward compatibility (rule-based only)."""
    return rule_based_intent(prompt).intent


def extract_intent_hybrid(prompt: str, use_llm: bool = False, model: str = "gpt-4o-mini") -> IntentResult:
    """Hybrid intent extraction: rules first, optional LLM refinement."""
    rule_res = rule_based_intent(prompt)
    if not use_llm:
        return rule_res
    llm_json = _call_llm(prompt, rule_res.intent, model=model)
    if not llm_json:
        # Fallback silently to rule-based result
        return rule_res
    return _merge_rule_and_llm(rule_res, llm_json)