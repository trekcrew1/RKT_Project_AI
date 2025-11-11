#!/usr/bin/env python3
# make_prd_from_idea.py
# Generate a PRD (JSON + Markdown) from a scraped idea JSON, keyed by a compound name.
# Usages:
#   - As a module:  import make_prd_from_idea; make_prd_from_idea.main("TicketHawk")
#   - As a script:  python make_prd_from_idea.py TicketHawk

import threading
import time
import os, sys, json, argparse
from datetime import datetime
from dateutil import tz, parser
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# ---------- ENV & CLIENT ----------

load_dotenv()

# Ensure OPENAI_API_KEY exists even if .env uses 'openai_api_key'
env_lower = os.getenv("openai_api_key")
env_upper = os.getenv("OPENAI_API_KEY")
if not env_upper and env_lower:
    os.environ["OPENAI_API_KEY"] = env_lower

# ---------- CONFIG DEFAULTS ----------

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1")  # solid for structured JSON
DEFAULT_IN_DIR = os.path.join("ideas")                 # <-- always read from ./ideas
DEFAULT_JSON_DIR = os.path.join("out", "jsons")
DEFAULT_MD_DIR = os.path.join("out", "prds")

class ProgressDots:
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self, message="Working"):
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(message,), daemon=True)
        self.thread.start()

    def _run(self, message):
        sys.stdout.write(message)
        sys.stdout.flush()
        while self.running:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(self.interval)
        sys.stdout.write("\n")  # finish with newline when stopped
        sys.stdout.flush()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

# ---------- PRD JSON SCHEMA (strict) ----------

PRD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "meta": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "version": {"type": "string"},
                "date": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["draft", "review", "final"]},
            },
            "required": ["title", "version", "date", "authors", "status"],
        },
        "overview": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "problem_statement": {"type": "string"},
                "goals": {"type": "array", "items": {"type": "string"}},
                "non_goals": {"type": "array", "items": {"type": "string"}},
                "background": {"type": "string"},
            },
            "required": ["summary", "problem_statement", "goals", "non_goals", "background"],
        },
        "market": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_users": {"type": "array", "items": {"type": "string"}},
                "personas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "bio": {"type": "string"},
                            "pain_points": {"type": "array", "items": {"type": "string"}},
                            "jobs_to_be_done": {"type": "array", "items": {"type": "string"}},
                            "success_criteria": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "bio", "pain_points", "jobs_to_be_done", "success_criteria"],
                    },
                },
                "market_size": {"type": "string"},
                "competitors": {"type": "array", "items": {"type": "string"}},
                "differentiation": {"type": "string"},
            },
            "required": ["target_users", "personas", "market_size", "competitors", "differentiation"],
        },
        "requirements": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "user_stories": {"type": "array", "items": {"type": "string"}},
                "functional_requirements": {"type": "array", "items": {"type": "string"}},
                "nonfunctional_requirements": {"type": "array", "items": {"type": "string"}},
                "ux_principles": {"type": "array", "items": {"type": "string"}},
                "accessibility": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "open_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "user_stories",
                "functional_requirements",
                "nonfunctional_requirements",
                "ux_principles",
                "accessibility",
                "dependencies",
                "open_questions",
            ],
        },
        "delivery": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "milestones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "scope": {"type": "string"},
                            "duration_weeks": {"type": "number"},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "scope", "duration_weeks", "acceptance_criteria"],
                    },
                },
                "timeline_weeks_total": {"type": "number"},
                "team": {"type": "array", "items": {"type": "string"}},
                "resources": {"type": "array", "items": {"type": "string"}},
                "risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "risk": {"type": "string"},
                            "mitigation": {"type": "string"},
                        },
                        "required": ["risk", "mitigation"],
                    },
                },
            },
            "required": ["milestones", "timeline_weeks_total", "team", "resources", "risks"],
        },
        "gtm": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pricing": {"type": "string"},
                "channels": {"type": "array", "items": {"type": "string"}},
                "positioning": {"type": "string"},
                "kpis": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["pricing", "channels", "positioning", "kpis"],
        },
    },
    "required": ["meta", "overview", "market", "requirements", "delivery", "gtm"],
}

# ---------- SYSTEM PROMPTS ----------

SYSTEM_INSTRUCTIONS = """You are a Staff Product Manager.
Produce a thorough, practical Product Requirements Document (PRD) that an engineering team can start from.
Focus on clarity, testability, and business viability. Assume we are a seed-stage startup.
Use the provided 'idea text' as the main source of truth. Infer details responsibly.
Prefer concise bullet points over long paragraphs, except for Overview and Background.
Be explicit about acceptance criteria, risks, and measurable KPIs.
Define all acronyms the first time used in the document and put them within parentheses next to the acronym.
"""

USER_TASK = """Create a PRD for turning the following idea into a real, revenue-generating product.
Return your answer as JSON that strictly conforms to the provided `PRD_SCHEMA`."""

# ---------- HELPERS ----------

def to_local_iso_now() -> str:
    return datetime.now(tz=tz.tzlocal()).isoformat(timespec="seconds")

# def format_meta_date(date_str: str) -> str:
#     """
#     Parse an ISO-like date string, convert to local tz, and render as
#     'YYYY-MM-DD HH:MM AM/PM'. Returns '' if parsing fails or input empty.
#     """
#     if not date_str:
#         return ""
#     try:
#         dt = parser.parse(date_str)
#         # normalize to local timezone for consistent display
#         dt_local = dt.astimezone(tz.tzlocal()) if dt.tzinfo else dt.replace(tzinfo=tz.UTC).astimezone(tz.tzlocal())
#         print('dt_local=> ', dt_local)
#         # dt_local=>  2025-08-25 20:00:00-04:00
#         return dt_local.strftime("%Y-%m-%d %I:%M %p")
#     except Exception:
#         print("Failed to parse date:", date_str)
#         return str(date_str)

def format_meta_date(date_str: str) -> str:
    """
    Parse an ISO-like date string, convert to local tz, and render as
    'YYYY-MM-DD HH:MM AM/PM'. If input has no time, use current local time.
    """
    if not date_str:
        return ""
    try:
        dt = parser.parse(date_str)
        if not dt.tzinfo:
            # If no timezone info, assume local
            dt = dt.replace(tzinfo=tz.tzlocal())
        # If the string had no time (only date), fill in today's time
        if dt.hour == 0 and dt.minute == 0 and "T" not in date_str:
            now_local = datetime.now(tz=tz.tzlocal())
            dt = dt.replace(hour=now_local.hour, minute=now_local.minute, second=0)

        dt_local = dt.astimezone(tz.tzlocal())
        print("dt_local=>", dt_local)
        return dt_local.strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        print("Failed to parse date:", date_str)
        return str(date_str)
    
def load_idea_text(path: str) -> str:
    """Load idea text from a file. If it's JSON from scraper, extract text-ish fields."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Put your scraped JSON in this file.")
    txt = open(path, "r", encoding="utf-8").read().strip()
    if path.lower().endswith(".json"):
        try:
            obj = json.loads(txt)
            # Prefer 'idea_text', 'full_text', 'text', else fall back to 'main_text'
            for k in ("idea_text", "full_text", "text", "main_text"):
                if isinstance(obj, dict) and k in obj and isinstance(obj[k], str) and obj[k].strip():
                    return obj[k].strip()
        except Exception:
            pass
    return txt

def _ensure_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [val] if val.strip() else []
    return [str(val)]

def _ensure_obj(val: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        # Try to parse embedded JSON
        try:
            maybe = json.loads(s)
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            pass
        fb = dict(fallback)
        if "summary" in fb and not fb["summary"]:
            fb["summary"] = s
        return fb
    return dict(fallback)

def _repair_prd(prd: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce malformed fields into expected shapes so markdown rendering never crashes."""
    if not isinstance(prd, dict):
        raise TypeError("Model did not return a JSON object at top level.")

    # meta
    prd.setdefault("meta", {})
    if not isinstance(prd["meta"], dict):
        print("[repair] 'meta' was not an object -> replacing with defaults", file=sys.stderr)
        prd["meta"] = {}
    prd["meta"].setdefault("title", "New Product")
    prd["meta"].setdefault("version", "v0.1")
    prd["meta"].setdefault("date", to_local_iso_now())
    prd["meta"].setdefault("status", "draft")
    prd["meta"].setdefault("authors", [])

    # overview
    prd["overview"] = _ensure_obj(
        prd.get("overview"),
        {"summary": "", "problem_statement": "", "goals": [], "non_goals": [], "background": ""}
    )
    prd["overview"]["goals"] = _ensure_list(prd["overview"].get("goals"))
    prd["overview"]["non_goals"] = _ensure_list(prd["overview"].get("non_goals"))
    prd["overview"].setdefault("summary", "")
    prd["overview"].setdefault("problem_statement", "")
    prd["overview"].setdefault("background", "")

    # market
    mkt_fb = {
        "target_users": [],
        "personas": [],
        "market_size": "",
        "competitors": [],
        "differentiation": ""
    }
    prd["market"] = _ensure_obj(prd.get("market"), mkt_fb)
    prd["market"]["target_users"] = _ensure_list(prd["market"].get("target_users"))
    prd["market"]["competitors"] = _ensure_list(prd["market"].get("competitors"))
    prd["market"]["personas"] = prd["market"].get("personas") or []
    if not isinstance(prd["market"]["personas"], list):
        print("[repair] 'market.personas' not a list -> coercing", file=sys.stderr)
        prd["market"]["personas"] = _ensure_list(prd["market"]["personas"])
    prd["market"].setdefault("differentiation", "")
    prd["market"].setdefault("market_size", "")

    # requirements
    req_fb = {
        "user_stories": [],
        "functional_requirements": [],
        "nonfunctional_requirements": [],
        "ux_principles": [],
        "accessibility": "",
        "dependencies": [],
        "open_questions": []
    }
    prd["requirements"] = _ensure_obj(prd.get("requirements"), req_fb)
    for k in ("user_stories", "functional_requirements", "nonfunctional_requirements",
              "ux_principles", "dependencies", "open_questions"):
        prd["requirements"][k] = _ensure_list(prd["requirements"].get(k))
    prd["requirements"].setdefault("accessibility", "")

    # delivery
    del_fb = {
        "milestones": [],
        "timeline_weeks_total": None,
        "team": [],
        "resources": [],
        "risks": []
    }
    prd["delivery"] = _ensure_obj(prd.get("delivery"), del_fb)
    if not isinstance(prd["delivery"].get("milestones"), list):
        print("[repair] 'delivery.milestones' not a list -> coercing", file=sys.stderr)
        prd["delivery"]["milestones"] = _ensure_list(prd["delivery"]["milestones"])
    risks = prd["delivery"].get("risks") or []
    if isinstance(risks, list):
        fixed_risks = []
        for r in risks:
            if isinstance(r, dict):
                fixed_risks.append({"risk": r.get("risk", ""), "mitigation": r.get("mitigation", "")})
            else:
                fixed_risks.append({"risk": str(r), "mitigation": ""})
        prd["delivery"]["risks"] = fixed_risks
    else:
        prd["delivery"]["risks"] = [{"risk": str(risks), "mitigation": ""}]
    prd["delivery"]["team"] = _ensure_list(prd["delivery"].get("team"))
    prd["delivery"]["resources"] = _ensure_list(prd["delivery"].get("resources"))
    tlt = prd["delivery"].get("timeline_weeks_total")
    if isinstance(tlt, str):
        try:
            prd["delivery"]["timeline_weeks_total"] = float(tlt)
        except Exception:
            prd["delivery"]["timeline_weeks_total"] = None

    # gtm
    gtm_fb = {"pricing": "", "channels": [], "positioning": "", "kpis": []}
    prd["gtm"] = _ensure_obj(prd.get("gtm"), gtm_fb)
    prd["gtm"]["channels"] = _ensure_list(prd["gtm"].get("channels"))
    prd["gtm"]["kpis"] = _ensure_list(prd["gtm"].get("kpis"))
    prd["gtm"].setdefault("pricing", "")
    prd["gtm"].setdefault("positioning", "")

    return prd

def json_to_markdown(prd: Dict[str, Any]) -> str:
    """Pretty, defensive Markdown renderer."""
    m = prd

    def bullets(arr):
        arr = arr or []
        return "\n".join(f"- {x}" for x in arr if str(x).strip())

    def bullets_pairs(arr, left="risk", right="mitigation"):
        out = []
        for it in (arr or []):
            if isinstance(it, dict):
                l = it.get(left, "")
                r = it.get(right, "")
            else:
                l, r = str(it), ""
            if l and r:
                out.append(f"- **{l}** → _Mitigation_: {r}")
            elif l:
                out.append(f"- **{l}**")
        return "\n".join(out)

    lines: List[str] = []
    meta = m.get("meta", {})
    # format date for display
    print('******** meta date=> ', meta.get("date", ""))
    formatted_date = format_meta_date(meta.get("date", ""))

    lines.append(f"# {meta.get('title','New Product')} — PRD\n")
    lines.append(
        f"**Version:** {meta.get('version','v0.1')}  \n"
        f"**Status:** {meta.get('status','draft')}  \n"
        f"**Date:** {formatted_date}"
    )
    if meta.get("authors"):
        lines.append(f"**Authors:** {', '.join(meta.get('authors', []))}")
    lines.append("\n---\n")

    ov = m.get("overview", {})
    lines.append("## 1. Overview")
    lines.append(f"**Summary**\n\n{ov.get('summary','')}\n")
    lines.append(f"**Problem Statement**\n\n{ov.get('problem_statement','')}\n")
    if ov.get("background"):
        lines.append(f"**Background**\n\n{ov.get('background','')}\n")
    lines.append("**Goals**\n" + bullets(ov.get("goals", [])))
    if ov.get("non_goals"):
        lines.append("\n**Non-Goals**\n" + bullets(ov.get("non_goals", [])))
    lines.append("\n---\n")

    mk = m.get("market", {})
    lines.append("## 2. Market & Users")
    lines.append("**Target Users**\n" + bullets(mk.get("target_users", [])))
    personas = mk.get("personas") or []
    if personas:
        lines.append("\n**Personas**")
        for p in personas:
            if isinstance(p, dict):
                lines.append(f"- **{p.get('name','')}** — {p.get('bio','')}")
                if p.get("pain_points"):
                    lines.append("  - Pain points:\n    " + "\n    ".join(f"- {x}" for x in p.get("pain_points", [])))
                if p.get("jobs_to_be_done"):
                    lines.append("  - JTBD:\n    " + "\n    ".join(f"- {x}" for x in p.get("jobs_to_be_done", [])))
                if p.get("success_criteria"):
                    lines.append("  - Success criteria:\n    " + "\n    ".join(f"- {x}" for x in p.get("success_criteria", [])))
            else:
                lines.append(f"- {str(p)}")
    if mk.get("market_size"):
        lines.append("\n**Market Size**\n" + mk.get("market_size",""))
    if mk.get("competitors"):
        lines.append("\n**Competitors**\n" + bullets(mk.get("competitors", [])))
    if mk.get("differentiation"):
        lines.append("\n**Differentiation**\n" + mk.get("differentiation",""))
    lines.append("\n---\n")

    rq = m.get("requirements", {})
    lines.append("## 3. Requirements")
    lines.append("**User Stories**\n" + bullets(rq.get("user_stories", [])))
    lines.append("\n**Functional Requirements**\n" + bullets(rq.get("functional_requirements", [])))
    if rq.get("nonfunctional_requirements"):
        lines.append("\n**Non-Functional Requirements**\n" + bullets(rq.get("nonfunctional_requirements", [])))
    if rq.get("ux_principles"):
        lines.append("\n**UX Principles**\n" + bullets(rq.get("ux_principles", [])))
    if rq.get("accessibility"):
        lines.append("\n**Accessibility**\n" + rq.get("accessibility",""))
    if rq.get("dependencies"):
        lines.append("\n**Dependencies**\n" + bullets(rq.get("dependencies", [])))
    if rq.get("open_questions"):
        lines.append("\n**Open Questions**\n" + bullets(rq.get("open_questions", [])))
    lines.append("\n---\n")

    dl = m.get("delivery", {})
    lines.append("## 4. Delivery Plan")
    if dl.get("milestones"):
        lines.append("**Milestones**")
        for ms in dl.get("milestones", []):
            if isinstance(ms, dict):
                lines.append(f"- **{ms.get('name','')}**  \n  Scope: {ms.get('scope','')}")
                if ms.get("duration_weeks") is not None:
                    lines.append(f"  \n  Duration: ~{ms.get('duration_weeks')} weeks")
                if ms.get("acceptance_criteria"):
                    lines.append("\n  Acceptance Criteria:\n  " + "\n  ".join(f"- {x}" for x in ms.get("acceptance_criteria", [])))
            else:
                lines.append(f"- {str(ms)}")
    if dl.get("timeline_weeks_total") is not None:
        lines.append(f"\n**Total Timeline:** ~{dl.get('timeline_weeks_total')} weeks")
    if dl.get("team"):
        lines.append("\n**Team**\n" + bullets(dl.get("team", [])))
    if dl.get("resources"):
        lines.append("\n**Resources**\n" + bullets(dl.get("resources", [])))
    if dl.get("risks"):
        lines.append("\n**Risks & Mitigations**\n" + bullets_pairs(dl.get("risks", [])))
    lines.append("\n---\n")

    gtm = m.get("gtm", {})
    lines.append("## 5. Go-to-Market")
    if gtm.get("positioning"):
        lines.append("**Positioning**\n" + gtm.get("positioning",""))
    if gtm.get("pricing"):
        lines.append("\n**Pricing**\n" + gtm.get("pricing",""))
    if gtm.get("channels"):
        lines.append("\n**Channels**\n" + bullets(gtm.get("channels", [])))
    if gtm.get("kpis"):
        lines.append("\n**KPIs / Success Metrics**\n" + bullets(gtm.get("kpis", [])))

    return "\n".join(lines).strip() + "\n"

# ---------- PUBLIC ENTRYPOINT ----------

# def main(
#     file_name: str,
#     input_path: str,
#     model: Optional[str] = None,
#     json_dir: Optional[str] = None,
#     md_dir: Optional[str] = None,
# ) -> None:
#     """
#     Public entrypoint callable from other scripts:
#         import make_prd_from_idea
#         make_prd_from_idea.main("TimeTracker", "ideas\TimeTracker.json")

#     Parameters:
#       - file_name:  file name for the idea (e.g., "TimeTracker.json")
#       - input_path: file path name for the idea (e.g., "ideas\TimeTracker.json")
#       - model:      OpenAI model name (default from env/DEFAULT_MODEL)
#       - json_dir:   output dir for <compound>.prd.json     (default: out/jsons)
#       - md_dir:     output dir for <compound>.md           (default: out/prds)
#     """
#     model = model or DEFAULT_MODEL
#     in_dir = DEFAULT_IN_DIR  # always use ./ideas as input directory
#     json_dir = json_dir or DEFAULT_JSON_DIR
#     md_dir = md_dir or DEFAULT_MD_DIR

#     input_file = input_path
#     json_out   = os.path.join(json_dir, f"{file_name}.json")
#     md_out     = os.path.join(md_dir, f"{file_name}.md")

#     # Ensure folders exist
#     os.makedirs(os.path.dirname(json_out), exist_ok=True)
#     os.makedirs(os.path.dirname(md_out), exist_ok=True)


# ---------- PUBLIC ENTRYPOINT ----------

def main(
    # project_name, project_location, overwrite=overwrite_md
    file_name: str,
    input_path: str,
    model: Optional[str] = None,
    json_dir: Optional[str] = None,
    md_dir: Optional[str] = None,
    overwrite: bool = False,   # 👈 new param
) -> None:
    """
    Public entrypoint callable from other scripts:
        import make_prd_from_idea
        make_prd_from_idea.main("TimeTracker", "ideas/TimeTracker.json", overwrite=True)

    Parameters:
      - file_name:   file name for the idea (e.g., "TimeTracker.json")
      - input_path:  file path name for the idea (e.g., "ideas/TimeTracker.json")
      - model:       OpenAI model name (default from env/DEFAULT_MODEL)
      - json_dir:    output dir for <compound>.json  (default: out/jsons)
      - md_dir:      output dir for <compound>.md    (default: out/prds)
      - overwrite:   if False and .md exists, skip regeneration
    """
    model = model or DEFAULT_MODEL
    in_dir = DEFAULT_IN_DIR  # always use ./ideas as input directory
    json_dir = json_dir or DEFAULT_JSON_DIR
    md_dir = md_dir or DEFAULT_MD_DIR

    input_file = input_path
    json_out   = os.path.join(json_dir, f"{file_name}.json")
    md_out     = os.path.join(md_dir, f"{file_name}.md")

    # Ensure folders exist
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    os.makedirs(os.path.dirname(md_out), exist_ok=True)

    # ✅ Skip if .md already exists and overwrite not requested
    if os.path.exists(md_out) and not overwrite:
        print(f"⏩ Skipping: Markdown file already exists at {md_out}")
        return

    # Load idea text
    idea_text = load_idea_text(input_file)

    # OpenAI call
    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"{USER_TASK}\n\nIDEA TEXT:\n\n{idea_text}"}
    ]

    prd_obj = None
    err_stack: List[str] = []

    # Progress indicator (prints a dot every 5s)
    dots = ProgressDots(interval=5)
    start_ts = time.time()
    dots.start("Generating PRD from idea ")

    # Try json_schema (newer SDKs)
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "prd_schema",
                    "schema": PRD_SCHEMA,
                    "strict": True
                }
            },
        )
        content = chat.choices[0].message.content
        prd_obj = json.loads(content)
    except Exception as e:
        err_stack.append(f"[json_schema] {type(e).__name__}: {e}")
        # Fallback: json_object (older SDK)
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = chat.choices[0].message.content
            prd_obj = json.loads(content)
        except Exception as e2:
            err_stack.append(f"[json_object] {type(e2).__name__}: {e2}")
            # Last resort: no response_format
            chat = client.chat.completions.create(model=model, messages=messages)
            content = chat.choices[0].message.content
            prd_obj = json.loads(content)

    finally:
        # Always stop dots and print elapsed time
        dots.stop()
        elapsed = time.time() - start_ts
        print(f"Done in {elapsed:.1f}s")

    # Patch meta defaults
    if not isinstance(prd_obj, dict):
        raise TypeError(f"Model returned non-object PRD: {type(prd_obj)}")

    now_iso = to_local_iso_now()
    prd_obj.setdefault("meta", {})
    prd_obj["meta"].setdefault("date", now_iso)
    prd_obj["meta"].setdefault("version", "v0.1")
    prd_obj["meta"].setdefault("status", "draft")
    prd_obj["meta"].setdefault("authors", [])
    if not prd_obj["meta"].get("title"):
        first_line = (idea_text.splitlines()[0] if idea_text else "New Product").strip()[:120]
        prd_obj["meta"]["title"] = first_line or "New Product"

    # Repair schema drift
    try:
        prd_obj = _repair_prd(prd_obj)
    except Exception as e:
        print("Failed to repair PRD:", e, file=sys.stderr)
        raise

    # Write files
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(prd_obj, f, ensure_ascii=False, indent=2)

    md = json_to_markdown(prd_obj)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md)

    # Diagnostics
    if err_stack:
        print("⚠️  Model call fallbacks used:\n  - " + "\n  - ".join(err_stack))
    print(f"✅ Wrote {json_out} and {md_out} using model: {model}")

# ---------- CLI WRAPPER ----------

# def _parse_cli_args() -> argparse.Namespace:
#     ap = argparse.ArgumentParser(
#         description="Create a PRD (JSON + MD) from an idea file in ./ideas/"
#     )
#     ap.add_argument("compound", help="Compound name for the idea (e.g., TicketHawk, BrainVitality)")
#     ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
#     ap.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help=f"Output JSON directory (default: {DEFAULT_JSON_DIR})")
#     ap.add_argument("--md-dir", default=DEFAULT_MD_DIR, help=f"Output Markdown directory (default: {DEFAULT_MD_DIR})")
#     return ap.parse_args()

# def cli():
#     args = _parse_cli_args()
#     main(
#         compound=args.compound,
#         model=args.model,
#         json_dir=args.json_dir,
#         md_dir=args.md_dir,
#     )

def _parse_cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create a PRD (JSON + MD) from an idea file in ./ideas/"
    )
    ap.add_argument("compound", help="Compound name for the idea (e.g., TicketHawk, BrainVitality)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    ap.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help=f"Output JSON directory (default: {DEFAULT_JSON_DIR})")
    ap.add_argument("--md-dir", default=DEFAULT_MD_DIR, help=f"Output Markdown directory (default: {DEFAULT_MD_DIR})")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate .md even if it already exists")  # 👈 new flag
    return ap.parse_args()

def cli():
    args = _parse_cli_args()
    main(
        file_name=args.compound,
        input_path=os.path.join(DEFAULT_IN_DIR, f"{args.compound}.json"),
        model=args.model,
        json_dir=args.json_dir,
        md_dir=args.md_dir,
        overwrite=args.overwrite,   # 👈 pass it through
    )


if __name__ == "__main__":
    try:
        '''
        make_prd_from_idea.main('TimeTracker', 'ideas\TimeTracker_20250819.json')
        '''
        cli()
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to decode model JSON: {e}\n"
              f"Tip: Log the raw content and inspect. You can also print chat.choices[0].message.content.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"\n❌ Unhandled error: {type(e).__name__}: {e}", file=sys.stderr)
        raise
