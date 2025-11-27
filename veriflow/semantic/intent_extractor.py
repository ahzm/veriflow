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
        "每天", "每小时", "定时", "每日", "工作日",
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
    # Data store / DB-like targets
    "need_db": (
        "airtable", "sheet", "spreadsheet", "google sheets", "notion",
        "database", "db", "表格", "存储", "写入表", "入库"
    ),
    # User input / form / webhook sources
    "need_form": (
        "form", "表单", "填写", "提交", "google form", "typeform", "survey"
    ),
    # Conditional / branching
    "need_condition": (
        "if", "else", "condition", "branch", "threshold", "超过", "大于", "小于", "否则", "条件"
    ),
    # Data transform / validate / map
    "need_transform": (
        "validate", "validation", "clean", "parse", "format",
        "transform", "map", "set", "function", "转换", "校验", "解析", "格式化"
    ),
    # Conditional notification (condition controlling action)
    "need_conditional_notify": (
        # CN
        "就通知", "报警", "告警", "提醒", "超过就通知", "低于就报警",
        "如果...就发", "满足...才通知",
        # EN
        "notify if", "alert if", "send if", "only if", "if ... notify", "if ... alert"
    ),
}

# --- Negative patterns ("no email") ---
NEG_EMAIL_PATTERNS = (
    # CN
    "不要邮件", "不要发邮件", "不要发送邮件",
    "不发邮件", "不发送邮件",
    # EN
    "no email", "no emails",
    "do not email", "don't email",
    "do not send email", "don't send email",
)

NEG_SLACK_PATTERNS = (
    "不要slack", "不发slack", "不通知slack",
    "no slack", "don't slack", "do not slack", "do not notify slack",
)
NEG_HTTP_PATTERNS = (
    "不要http", "不请求", "不调用接口",
    "no http", "don't call api", "do not call api", "without api",
)
NEG_TELEGRAM_PATTERNS = (
    "不要telegram", "不发电报",
    "no telegram", "don't telegram", "do not telegram",
)
NEG_DB_PATTERNS = (
    "不要存储", "不写入表", "不入库",
    "no db", "no database", "don't store", "do not store",
)


# --- Intent keys inventory ---
INTENT_KEYS = [
    "need_schedule", "need_email", "need_http", "need_slack", "need_telegram",
    "need_db", "need_form", "need_condition", "need_transform", "need_conditional_notify",
]

# --- English time expressions helpers ---
# Matches: 9, 9am, 9 am, 09:00, 9:00, 9.00, 09h00, etc.
TIME_REGEX = re.compile(
    r"\b((?:[01]?\d|2[0-3])\s*(?:[:h.]\s*[0-5]\d)?\s*(?:am|pm)?)\b",
    flags=re.IGNORECASE,
)

# --- Light param extraction
THRESHOLD_REGEX = re.compile(
    r"(>=|<=|>|<|超过|大于|小于)\s*([0-9]+(?:\.[0-9]+)?)"
)

FIELD_REGEX = re.compile(
    r"\b(temperature|temp|cpu|usage|price|库存|销量|温度|湿度)\b",
    flags=re.IGNORECASE
)

def extract_params(prompt: str) -> Dict[str, Any]:
    """Extract coarse parameters like thresholds and field names from prompt."""
    t = _normalize(prompt)
    params: Dict[str, Any] = {}

    m = THRESHOLD_REGEX.search(t)
    if m:
        params["threshold_op"] = m.group(1)
        params["threshold_value"] = float(m.group(2))

    fields = FIELD_REGEX.findall(t)
    if fields:
        params["fields"] = sorted(set([f.lower() for f in fields]))

    return params

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
    for k in keywords:
        kk = k.lower()
        has_non_ascii = any(ord(ch) > 127 for ch in kk)
        if len(kk) <= 2 and not has_non_ascii:
            if re.search(rf"\b{re.escape(kk)}\b", t):
                return True
        else:
            if kk in t:
                return True
    return False


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

def _has_negative_email(text: str) -> bool:
    """Detect explicit 'no email / 不要邮件' style negation."""
    t = text.lower()
    return any(p.lower() in t for p in NEG_EMAIL_PATTERNS)

def _has_negative_slack(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in NEG_SLACK_PATTERNS)

def _has_negative_http(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in NEG_HTTP_PATTERNS)

def _has_negative_telegram(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in NEG_TELEGRAM_PATTERNS)

def _has_negative_db(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in NEG_DB_PATTERNS)

def _coerce_bool(v: Any) -> bool:
    """Best-effort conversion to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "yes", "y", "1"}:
            return True
        if t in {"false", "no", "n", "0"}:
            return False
    # default: False (conservative)
    return False

def _coerce_float(v: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float in [0,1]."""
    try:
        x = float(v)
    except Exception:
        return default
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def _validate_llm_json(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Validate and sanitize the LLM JSON response.

    Expected shape:
      {
        "intent":     { key: bool-like, ... },
        "confidence": { key: float-like [0,1], ... },
        "chain":      [ "explanation step", ... ]   # optional
      }

    Returns a cleaned dict or None if schema is unusable.
    """
    if not isinstance(raw, dict):
        return None

    intent_raw = raw.get("intent")
    conf_raw = raw.get("confidence")
    if not isinstance(intent_raw, dict) or not isinstance(conf_raw, dict):
        return None

    cleaned_intent: Dict[str, bool] = {}
    cleaned_conf: Dict[str, float] = {}

    for k in INTENT_KEYS:
        if k in intent_raw:
            cleaned_intent[k] = _coerce_bool(intent_raw[k])
        if k in conf_raw:
            cleaned_conf[k] = _coerce_float(conf_raw[k], default=0.0)

    # If nothing usable, discard
    if not cleaned_intent and not cleaned_conf:
        return None

    chain_raw = raw.get("chain", [])
    chain: list[str] = []
    if isinstance(chain_raw, list):
        for item in chain_raw:
            if isinstance(item, str):
                chain.append(item)

    return {
        "intent": cleaned_intent,
        "confidence": cleaned_conf,
        "chain": chain,
    }

def rule_based_intent(prompt: str) -> IntentResult:
    """Extract intent using keyword heuristics, English time patterns and simple negative-email handling."""
    text = _normalize(prompt)
    intent: Dict[str, bool] = {}
    conf: Dict[str, float] = {}
    intent_chain = []

    # need_schedule: keyword OR English-time-pattern
    schedule_hit = _contains_any(text, KEYWORDS["need_schedule"]) or _has_en_schedule_pattern(text)
    intent["need_schedule"] = schedule_hit
    conf["need_schedule"] = _rule_score(schedule_hit)

    if schedule_hit:
        intent_chain.append("rule: matched schedule keywords or English time pattern")

    # --- Email with negative intent handling ---
    email_hit = _contains_any(text, KEYWORDS["need_email"])
    email_neg = _has_negative_email(text)

    if email_hit and not email_neg:
        intent["need_email"] = True
        conf["need_email"] = _rule_score(True)
        intent_chain.append("rule: matched email keywords")
    else:
        # Either no email mention, or it is explicitly negated
        intent["need_email"] = False
        conf["need_email"] = _rule_score(False)
        if email_hit and email_neg:
            intent_chain.append("rule: email keywords found but explicitly negated (no email)")

    # need_http
    http_hit = _contains_any(text, KEYWORDS["need_http"])
    http_neg = _has_negative_http(text)
    intent["need_http"] = bool(http_hit and not http_neg)
    conf["need_http"] = _rule_score(intent["need_http"])
    if http_hit and not http_neg:
        intent_chain.append("rule: matched need_http keywords")
    elif http_hit and http_neg:
        intent_chain.append("rule: need_http keywords found but explicitly negated")

    # need_slack
    slack_hit = _contains_any(text, KEYWORDS["need_slack"])
    slack_neg = _has_negative_slack(text)
    intent["need_slack"] = bool(slack_hit and not slack_neg)
    conf["need_slack"] = _rule_score(intent["need_slack"])
    if slack_hit and not slack_neg:
        intent_chain.append("rule: matched need_slack keywords")
    elif slack_hit and slack_neg:
        intent_chain.append("rule: need_slack keywords found but explicitly negated")

    # need_telegram
    tg_hit = _contains_any(text, KEYWORDS["need_telegram"])
    tg_neg = _has_negative_telegram(text)
    intent["need_telegram"] = bool(tg_hit and not tg_neg)
    conf["need_telegram"] = _rule_score(intent["need_telegram"])
    if tg_hit and not tg_neg:
        intent_chain.append("rule: matched need_telegram keywords")
    elif tg_hit and tg_neg:
        intent_chain.append("rule: need_telegram keywords found but explicitly negated")

    # need_db
    db_hit = _contains_any(text, KEYWORDS["need_db"])
    db_neg = _has_negative_db(text)
    intent["need_db"] = bool(db_hit and not db_neg)
    conf["need_db"] = _rule_score(intent["need_db"])
    if db_hit and not db_neg:
        intent_chain.append("rule: matched need_db keywords")
    elif db_hit and db_neg:
        intent_chain.append("rule: need_db keywords found but explicitly negated")

    # need_form / need_condition / need_transform (no negation for now)
    for key in ("need_form", "need_condition", "need_transform"):
        hit = _contains_any(text, KEYWORDS[key])
        intent[key] = bool(hit)
        conf[key] = _rule_score(hit)
        if hit:
            intent_chain.append(f"rule: matched {key} keywords")
    
    # need_conditional_notify:
    # heuristic: conditional cue + at least one action cue
    cond_notify_hit = _contains_any(text, KEYWORDS["need_conditional_notify"])
    any_action_hit = any(intent.get(k, False) for k in ("need_email", "need_slack", "need_telegram"))

    cond_notify = bool(cond_notify_hit)

    intent["need_conditional_notify"] = cond_notify
    conf["need_conditional_notify"] = ( 
        0.85 if (cond_notify_hit and any_action_hit) 
        else 0.7 if cond_notify_hit
        else 0.05
    )

    if cond_notify:
        intent_chain.append("rule: matched need_conditional_notify (condition controls notification)")
    
    if intent.get("need_conditional_notify"):
        intent["need_condition"] = True
        conf["need_condition"] = max(conf.get("need_condition", 0.05), 0.7)

    # Overall confidence: average of active keys, fallback to mean
    active = [conf[k] for k, v in intent.items() if v]
    overall = sum(active) / max(1, len(active)) if active else sum(conf.values()) / max(1, len(conf))

    params = extract_params(prompt)
    return IntentResult(intent=intent, confidence=conf, overall_confidence=overall, source="rule", meta={"intent_chain": intent_chain, "params": params},)

def _call_llm(prompt: str, base_intent: Dict[str, bool], model: str = "gpt-4o-mini") -> Optional[Dict[str, Any]]:
    """Call LLM to refine or correct the base intent.

    Returns a cleaned JSON dict (see _validate_llm_json) or None on failure.
    """
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
        "{need_schedule, need_email, need_http, need_slack, need_telegram, "
        "need_db, need_form, need_condition, need_transform, need_conditional_notify}."
        "Also include a 'confidence' object with float scores in [0,1] per key. "
        "Also include a 'chain' array with short natural-language justifications explaining how each intent was inferred."
    )

    def _one_call(extra_hint: str = "") -> Optional[Dict[str, Any]]:
        user_msg = (
            "Description: " + prompt + "\n"
            "Base intent (from rules): " + json.dumps(base_intent, ensure_ascii=False) + "\n"
            'Respond ONLY with JSON of the form {"intent": {...}, "confidence": {...}, "chain": [...]} '
            "with double quotes and no surrounding explanation. "
            + extra_hint
        )
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
        try:
            raw = json.loads(content)
        except Exception:
            return None
        return _validate_llm_json(raw)

    # First attempt
    data = _one_call()
    if data is not None:
        return data

    # Second attempt with stronger hint
    data = _one_call(
        extra_hint=(
            " Your previous answer was not valid JSON or did not match the schema. "
            "Now respond again with STRICTLY valid JSON only, no prose."
        )
    )
    return data

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