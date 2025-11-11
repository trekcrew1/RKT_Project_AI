#!/usr/bin/env python3
"""
make_plan_from_prd.py

Generate a step-by-step execution plan (JSON + Markdown) from a PRD Markdown file.

Usage:
  python make_plan_from_prd.py --prd out/prds/TicketHawk.md
  python make_plan_from_prd.py --prd out/prds/TicketHawk.md --model gpt-5.1 --max-steps 12
  python make_plan_from_prd.py --prd out/prds/TicketHawk.md --max-steps auto --max-steps-hard 20

Outputs:
  out/plans/<base>.plan.json
  out/plans/<base>.plan.md
"""

import threading
import time
import os
import sys
import json
import argparse
from datetime import datetime
from dateutil import tz
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Env & Defaults ----------------

load_dotenv()

# Accept OPENAI_API_KEY or openai_api_key
env_lower = os.getenv("openai_api_key")
env_upper = os.getenv("OPENAI_API_KEY")
if not env_upper and env_lower:
    os.environ["OPENAI_API_KEY"] = env_lower

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.1")
DEFAULT_OUT_DIR = os.path.join("out", "plans")


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
        sys.stdout.write("\n")
        sys.stdout.flush()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


def now_local_iso() -> str:
    return datetime.now(tz=tz.tzlocal()).isoformat(timespec="seconds")


# ---------------- Response Schema (strict) ----------------

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "meta": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_prd": {"type": "string"},
                "title": {"type": "string"},
                "generated_at": {"type": "string"},
                "model": {"type": "string"},
                "total_steps": {"type": "integer"},
                "est_total_duration_days": {"type": "number"},
                "version": {"type": "string"}
            },
            "required": [
                "source_prd",
                "title",
                "generated_at",
                "model",
                "total_steps",
                "est_total_duration_days",
                "version"
            ]
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "objective": {"type": "string"},
                    "detailed_instructions": {"type": "array", "items": {"type": "string"}},
                    "prerequisites": {"type": "array", "items": {"type": "string"}},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "owner_role": {"type": "string"},
                    "collaborators": {"type": "array", "items": {"type": "string"}},
                    "time_estimate_hours_min": {"type": "number"},
                    "time_estimate_hours_max": {"type": "number"},
                    "dependencies": {"type": "array", "items": {"type": "integer"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    "risks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "risk": {"type": "string"},
                                "mitigation": {"type": "string"}
                            },
                            "required": ["risk", "mitigation"]
                        }
                    },
                    "notes": {"type": "string"}
                },
                "required": [
                    "number",
                    "title",
                    "objective",
                    "detailed_instructions",
                    "prerequisites",
                    "inputs",
                    "outputs",
                    "owner_role",
                    "collaborators",
                    "time_estimate_hours_min",
                    "time_estimate_hours_max",
                    "dependencies",
                    "acceptance_criteria",
                    "risks",
                    "notes"
                ]
            }
        },
        "critical_path": {"type": "array", "items": {"type": "integer"}},
        "swimlanes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "owner_role": {"type": "string"},
                    "step_numbers": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["owner_role", "step_numbers"]
            }
        }
    },
    "required": ["meta", "steps", "critical_path", "swimlanes"]
}


# ---------------- Prompts ----------------

SYSTEM_PROMPT = """You are a Principal Program Manager.
Given a product PRD (in Markdown), produce a concrete, end-to-end execution plan that an engineering + product + design team can follow.

Your plan MUST be phased and strictly ordered:

PHASE 0 — PRE-FLIGHT CHECKLIST (Steps 1..N)
- Purpose: Feasibility, compliance, and dependency validation BEFORE any build.
- Scope: ONLY validation/decision activities (no coding/build tasks). Examples: external data/API access, legal/TOS review, partner contracts, SMS compliance (10DLC/A2P), payment/in-app billing rules, attribution feasibility, cloud budget checks, app store account readiness, data retention/privacy approach, latency tests with real sources, etc.
- Format requirements for every Pre-Flight step:
  • Title must begin with: "PRE-FLIGHT — <short topic>"
  • Objective must be phrased as a validation/decision (e.g., "Validate Ticketing API access and latency targets")
  • Detailed instructions must be checklist-level (who to contact, what to verify, example SLAs/rate limits to confirm, sample queries/tests to run, specific artifacts to collect)
  • Acceptance criteria MUST end in a clear Go/No-Go statement with evidence (docs, signed terms, working test output, sandbox keys, registration approvals)
  • Risks must include what happens if validation fails and a mitigation/alternative path
  • Dependencies should be minimal; order steps from most critical business risk to least
- Output of Phase 0 MUST include the artifacts needed to proceed (e.g., API keys + rate limits documented, signed partner terms, 10DLC approval, Stripe live mode readiness, app-store accounts active, latency test report).

PHASE 1+ — BUILD EXECUTION (Steps N+1..M)
- After Pre-Flight passes, sequence all implementation work (backend ingestion, change detection, alerting engine, mobile apps, billing, analytics, observability, admin console, growth features, etc.)
- Keep steps practical, sequenced, and testable with:
  • Objective
  • Detailed, actionable instructions
  • Inputs & outputs (artifacts)
  • Prerequisites & dependencies
  • Owner role + collaborators
  • Time estimates (min/max hours)
  • Acceptance criteria (Definition of Done)
  • Risks & mitigations
- Favor clarity and brevity per checklist item, but do not skip critical detail.
- Assume a small but senior team at a seed-stage startup.
- Where the PRD is ambiguous, make reasonable assumptions and call them out as notes.

Return the plan using the exact JSON schema provided. Do NOT introduce new fields. Ensure Phase 0 items are numbered first and titled with the required "PRE-FLIGHT — ..." prefix.
"""


def build_user_prompt(prd_path: str, prd_text: str, title_guess: str, max_steps) -> str:
    """
    Build the user prompt with a soft-cap constraint only when max_steps is a positive int.
    Otherwise instruct the model to use the smallest reasonable number of steps.
    """
    constraint_lines: List[str] = []
    if isinstance(max_steps, int) and max_steps > 0:
        constraint_lines.append(f"- Cap total steps at about {max_steps} unless the PRD truly needs more.")
    else:
        constraint_lines.append("- Use the smallest reasonable number of steps; group related tasks under one step when possible.")

    constraint_lines += [
        "- Phase 0 (Pre-Flight) must contain ONLY feasibility/validation/decision steps, ordered from most critical to least, each with Go/No-Go acceptance criteria and concrete evidence/artifacts.",
        "- Phase 1+ (Build) must begin only after Phase 0 and should be strictly sequenced for implementation.",
        "- Keep instructions specific (tooling, commands, artifacts, handoffs).",
        "- Use realistic time estimates for senior contributors.",
        "- Define all acronyms the first time used in the document and put them within parentheses next to the acronym if not there already.",
    ]

    constraints = "\n".join(constraint_lines)

    return f"""Create a step-by-step plan from the following PRD, with Phase 0 as a ranked PRE-FLIGHT CHECKLIST first, followed by all build steps in execution order.

Constraints:
{constraints}

PRD TITLE: {title_guess}
PRD PATH: {prd_path}

--- PRD CONTENT START ---
{prd_text}
--- PRD CONTENT END ---
"""


# ---------------- Helpers: IO & Repair ----------------

def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PRD file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def safe_basename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    if base.lower().endswith(".md"):
        base = base[:-3]
    return base


def repair_plan(plan: Dict[str, Any], fallback_title: str, source_prd: str, model: str) -> Dict[str, Any]:
    """Coerce/patch common issues so Markdown rendering won't break."""
    if not isinstance(plan, dict):
        raise TypeError("Model did not return a JSON object at top level.")

    plan.setdefault("meta", {})
    m = plan["meta"]
    if not isinstance(m, dict):
        plan["meta"] = m = {}
    m.setdefault("source_prd", source_prd)
    m.setdefault("title", fallback_title or "Execution Plan")
    m.setdefault("generated_at", now_local_iso())
    m.setdefault("model", model)
    m.setdefault("total_steps", 0)
    m.setdefault("est_total_duration_days", 0)
    m.setdefault("version", "v0.1")

    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    # Normalize numbers, arrays, and required fields
    fixed_steps = []
    for i, s in enumerate(steps, start=1):
        s = s if isinstance(s, dict) else {}
        s.setdefault("number", s.get("number") or i)
        s.setdefault("title", s.get("title") or f"Step {s['number']}")
        s.setdefault("objective", s.get("objective") or "")
        s.setdefault("detailed_instructions", s.get("detailed_instructions") or [])
        s.setdefault("prerequisites", s.get("prerequisites") or [])
        s.setdefault("inputs", s.get("inputs") or [])
        s.setdefault("outputs", s.get("outputs") or [])
        s.setdefault("owner_role", s.get("owner_role") or "Owner")
        s.setdefault("collaborators", s.get("collaborators") or [])
        # time estimates
        try:
            s["time_estimate_hours_min"] = float(s.get("time_estimate_hours_min", 1))
        except Exception:
            s["time_estimate_hours_min"] = 1.0
        try:
            s["time_estimate_hours_max"] = float(s.get("time_estimate_hours_max", max(2, s["time_estimate_hours_min"])))
        except Exception:
            s["time_estimate_hours_max"] = max(2.0, float(s["time_estimate_hours_min"]))
        # deps
        deps = s.get("dependencies") or []
        if not isinstance(deps, list):
            deps = []
        s["dependencies"] = [int(d) for d in deps if str(d).isdigit()]
        # acceptance criteria
        ac = s.get("acceptance_criteria") or []
        s["acceptance_criteria"] = ac if isinstance(ac, list) else [str(ac)]
        # risks
        risks = s.get("risks") or []
        fixed_risks = []
        if isinstance(risks, list):
            for r in risks:
                if isinstance(r, dict):
                    fixed_risks.append({"risk": r.get("risk", ""), "mitigation": r.get("mitigation", "")})
                else:
                    fixed_risks.append({"risk": str(r), "mitigation": ""})
        else:
            fixed_risks = [{"risk": str(risks), "mitigation": ""}]
        s["risks"] = fixed_risks
        s.setdefault("notes", s.get("notes") or "")
        fixed_steps.append(s)
    plan["steps"] = fixed_steps

    # meta totals
    m["total_steps"] = len(fixed_steps)
    # crude sum of durations (max)
    total_hours = sum(float(s.get("time_estimate_hours_max", 0)) for s in fixed_steps)
    m["est_total_duration_days"] = round(total_hours / 6.0, 1)  # assume 6h of effective work/day

    # critical path & swimlanes
    cp = plan.get("critical_path") or []
    if not isinstance(cp, list):
        cp = []
    if not cp:
        cp = [s["number"] for s in fixed_steps]
    plan["critical_path"] = cp

    swimlanes = plan.get("swimlanes") or []
    if not isinstance(swimlanes, list) or not swimlanes:
        lanes: Dict[str, List[int]] = {}
        for s in fixed_steps:
            owner = s.get("owner_role", "Owner")
            lanes.setdefault(owner, []).append(s["number"])
        swimlanes = [{"owner_role": k, "step_numbers": v} for k, v in lanes.items()]
    plan["swimlanes"] = swimlanes

    return plan


def plan_to_markdown(plan: Dict[str, Any]) -> str:
    m = plan.get("meta", {})
    steps = plan.get("steps", [])
    cp = plan.get("critical_path", [])
    swimlanes = plan.get("swimlanes", [])

    def bullets(arr: List[str]) -> str:
        return "\n".join(f"- {x}" for x in (arr or []) if str(x).strip())

    def bullets_pairs(arr: List[Dict[str, str]]) -> str:
        out = []
        for it in (arr or []):
            if isinstance(it, dict):
                r = it.get("risk", "")
                g = it.get("mitigation", "")
                if r and g:
                    out.append(f"- **{r}** — Mitigation: {g}")
                elif r:
                    out.append(f"- **{r}**")
            else:
                out.append(f"- {str(it)}")
        return "\n".join(out)

    lines: List[str] = []
    lines.append(f"# Execution Plan — {m.get('title','')}\n")
    lines.append(f"**Source PRD:** {m.get('source_prd','')}  ")
    lines.append(f"**Generated:** {m.get('generated_at','')}  ")
    lines.append(f"**Model:** {m.get('model','')}  ")
    lines.append(f"**Total Steps:** {m.get('total_steps',0)}  ")
    lines.append(f"**Est. Duration:** ~{m.get('est_total_duration_days',0)} days")
    lines.append("\n---\n")

    # Swimlanes
    if swimlanes:
        lines.append("## Swimlanes (Owner → Steps)")
        for sl in swimlanes:
            lines.append(f"- **{sl.get('owner_role','Owner')}**: {', '.join(str(n) for n in sl.get('step_numbers', []))}")
        lines.append("\n---\n")

    # Critical path
    if cp:
        lines.append(f"**Critical path:** {', '.join(str(n) for n in cp)}\n")
        lines.append("\n---\n")

    # Steps
    lines.append("## Steps")
    for s in steps:
        lines.append(f"### {s.get('number','')}. {s.get('title','')}")
        if s.get("objective"):
            lines.append(f"**Objective:** {s['objective']}\n")
        if s.get("prerequisites"):
            lines.append("**Prerequisites:**")
            lines.append(bullets(s.get("prerequisites"))); lines.append("")
        if s.get("inputs"):
            lines.append("**Inputs:**")
            lines.append(bullets(s.get("inputs"))); lines.append("")
        if s.get("detailed_instructions"):
            lines.append("**Detailed Instructions (do each in order):**")
            for idx, x in enumerate(s.get("detailed_instructions", []), start=1):
                lines.append(f"{idx}. {x}")
            lines.append("")
        if s.get("outputs"):
            lines.append("**Outputs/Artifacts:**")
            lines.append(bullets(s.get("outputs"))); lines.append("")
        owner = s.get("owner_role")
        collab = ", ".join(s.get("collaborators", []) or [])
        lines.append(f"**Owner:** {owner or 'Owner'}{(' — Collaborators: ' + collab) if collab else ''}")
        tmin = s.get("time_estimate_hours_min", 0)
        tmax = s.get("time_estimate_hours_max", 0)
        lines.append(f"**Time Estimate:** ~{tmin}-{tmax} hours")
        if s.get("dependencies"):
            lines.append(f"**Depends on:** {', '.join(str(d) for d in s.get('dependencies', []))}")
        if s.get("acceptance_criteria"):
            lines.append("**Acceptance Criteria (DoD):**")
            lines.append(bullets(s.get("acceptance_criteria"))); lines.append("")
        if s.get("risks"):
            lines.append("**Risks & Mitigations:**")
            lines.append(bullets_pairs(s.get("risks"))); lines.append("")
        if s.get("notes"):
            lines.append(f"**Notes:** {s['notes']}")
        lines.append("\n---\n")

    return "\n".join(lines).strip() + "\n"


# ---------------- Hard-cap enforcement (merge overflow) ----------------

def _uniq(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        k = (x or "").strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def enforce_hard_cap(plan: Dict[str, Any], cap: Optional[int]) -> Dict[str, Any]:
    """
    If cap is set and steps exceed cap, merge steps [cap+1..end] into step `cap`.
    - No tasks/details are lost: we append instructions/criteria/etc. into the last kept step.
    - Dependencies > cap are remapped to cap.
    - Steps are renumbered 1..cap.
    - Critical path is clamped to 1..cap with de-dupe.
    - Swimlanes recomputed by owner_role of kept steps (merged content keeps the cap step owner).
    """
    if not cap or cap <= 0:
        return plan

    steps: List[Dict[str, Any]] = plan.get("steps") or []
    if len(steps) <= cap:
        # just renumber to be safe
        for i, s in enumerate(steps, start=1):
            s["number"] = i
        plan["meta"]["total_steps"] = len(steps)
        return plan

    keep = steps[:cap]
    spill = steps[cap:]

    # merge spill into the last kept step
    last = keep[-1]

    # Helpers to safely extend lists
    def extend_list(key: str, convert=lambda x: x):
        base = last.get(key) or []
        if not isinstance(base, list):
            base = [base]
        for s in spill:
            vals = s.get(key) or []
            if not isinstance(vals, list):
                vals = [vals]
            base.extend(convert(v) for v in vals)
        # de-duplicate strings lists (except risks which are dicts)
        if base and isinstance(base[0], str):
            base = _uniq(base)
        last[key] = base

    # Preserve context headings so merged content remains readable
    heading_blocks = []
    for s in spill:
        title = s.get("title") or f"Step {s.get('number')}"
        heading_blocks.append(f"### Merged from Step {s.get('number')}: {title}")

    # detailed_instructions with headings
    di = last.get("detailed_instructions") or []
    if not isinstance(di, list):
        di = [str(di)]
    for s in spill:
        title = s.get("title") or f"Step {s.get('number')}"
        di.append(f"---")
        di.append(f"### Merged from Step {s.get('number')}: {title}")
        for entry in s.get("detailed_instructions") or []:
            di.append(str(entry))
    last["detailed_instructions"] = di

    # Merge simple list fields
    for k in ("prerequisites", "inputs", "outputs", "acceptance_criteria", "collaborators"):
        extend_list(k, str)

    # Merge risks (array of dicts)
    base_risks = last.get("risks") or []
    if not isinstance(base_risks, list):
        base_risks = []
    for s in spill:
        for r in s.get("risks") or []:
            if isinstance(r, dict):
                base_risks.append({"risk": r.get("risk", ""), "mitigation": r.get("mitigation", "")})
            else:
                base_risks.append({"risk": str(r), "mitigation": ""})
    last["risks"] = base_risks

    # Notes: append
    base_notes = (last.get("notes") or "").strip()
    merged_notes = [base_notes] if base_notes else []
    for s in spill:
        n = (s.get("notes") or "").strip()
        if n:
            merged_notes.append(f"[Merged from step {s.get('number')}] {n}")
    last["notes"] = "\n\n".join(merged_notes).strip()

    # Time estimates: sum max/min for a rough roll-up
    min_sum = float(last.get("time_estimate_hours_min") or 0.0)
    max_sum = float(last.get("time_estimate_hours_max") or 0.0)
    for s in spill:
        try:
            min_sum += float(s.get("time_estimate_hours_min") or 0.0)
        except Exception:
            pass
        try:
            max_sum += float(s.get("time_estimate_hours_max") or 0.0)
        except Exception:
            pass
    last["time_estimate_hours_min"] = round(min_sum, 2)
    last["time_estimate_hours_max"] = round(max_sum, 2)

    # Keep only the first `cap` steps and renumber
    steps = keep
    for i, s in enumerate(steps, start=1):
        s["number"] = i

    # Remap dependencies: any > cap -> cap
    for s in steps:
        deps = s.get("dependencies") or []
        remapped = []
        for d in deps:
            try:
                di = int(d)
            except Exception:
                continue
            if di > cap:
                di = cap
            if di >= 1 and di != s["number"]:
                remapped.append(di)
        # de-dupe while preserving order
        seen = set()
        final = []
        for d in remapped:
            if d not in seen:
                seen.add(d)
                final.append(d)
        s["dependencies"] = final

    # Clamp & de-dupe critical path
    cp = plan.get("critical_path") or []
    clamped_cp = []
    seen_cp = set()
    for n in cp:
        try:
            ni = int(n)
        except Exception:
            continue
        if ni > cap:
            ni = cap
        if 1 <= ni <= cap and ni not in seen_cp:
            seen_cp.add(ni)
            clamped_cp.append(ni)
    if not clamped_cp:
        clamped_cp = [s["number"] for s in steps]
    plan["critical_path"] = clamped_cp

    # Recompute swimlanes from kept steps
    lanes: Dict[str, List[int]] = {}
    for s in steps:
        owner = s.get("owner_role", "Owner")
        lanes.setdefault(owner, []).append(s["number"])
    swim = [{"owner_role": k, "step_numbers": v} for k, v in lanes.items()]
    plan["swimlanes"] = swim

    # Update plan
    plan["steps"] = steps
    plan["meta"]["total_steps"] = len(steps)
    total_hours = sum(float(s.get("time_estimate_hours_max", 0)) for s in steps)
    plan["meta"]["est_total_duration_days"] = round(total_hours / 6.0, 1)

    return plan


# ---------------- Core ----------------

def generate_plan_from_prd(
    prd_path: str,
    model: str = DEFAULT_MODEL,
    max_steps=10  # int soft cap, or None/"auto" → minimize steps
) -> Dict[str, Any]:
    prd_text = read_text(prd_path)
    title_guess = safe_basename_no_ext(prd_path)

    user_prompt = build_user_prompt(
        prd_path=prd_path,
        prd_text=prd_text,
        title_guess=title_guess,
        max_steps=max_steps
    )

    client = OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    plan_obj: Optional[Dict[str, Any]] = None
    errors: List[str] = []

    # Progress indicator (prints a dot every 5s)
    dots = ProgressDots(interval=5)
    start_ts = time.time()
    dots.start("Generating Plan from PRD ")

    try:
        # Try json_schema first
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "execution_plan_schema",
                        "schema": PLAN_SCHEMA,
                        "strict": True
                    }
                },
            )
            content = resp.choices[0].message.content
            plan_obj = json.loads(content)

        except Exception as e:
            errors.append(f"[json_schema] {type(e).__name__}: {e}")
            # Fallback to json_object
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                plan_obj = json.loads(content)
            except Exception as e2:
                errors.append(f"[json_object] {type(e2).__name__}: {e2}")
                # Last resort: free-form, then parse
                resp = client.chat.completions.create(model=model, messages=messages)
                content = resp.choices[0].message.content
                plan_obj = json.loads(content)

        if not isinstance(plan_obj, dict):
            raise TypeError(f"Model returned non-object plan: {type(plan_obj)}")

        # Repair & normalize
        plan_obj = repair_plan(plan_obj, fallback_title=title_guess, source_prd=prd_path, model=model)

        if errors:
            print("WARNING: Model call fallbacks used:\n  - " + "\n  - ".join(errors), file=sys.stderr)

        return plan_obj

    finally:
        # Always stop dots and print elapsed time
        dots.stop()
        elapsed = time.time() - start_ts
        print(f"Done in {elapsed:.1f}s")


def main(
    project_name: str,
    prd_path: str,
    model: str = DEFAULT_MODEL,
    out_dir: str = DEFAULT_OUT_DIR,
    max_steps: Optional[int] = 10,   # soft cap; None = auto/minimize
    max_steps_hard: Optional[int] = None,  # hard cap; overflow merged into final step
    overwrite: bool = True,
):
    """
    Public entrypoint callable from other scripts:
        import make_plan_from_prd
        make_plan_from_prd.main("TimeTracker", "out/prds/TimeTracker.md", overwrite=True)

    Parameters:
      - project_name:   base name for output files (without extension)
      - prd_path:       path to PRD Markdown file
      - model:          OpenAI model name
      - out_dir:        output directory for plan files
      - max_steps:      soft cap (int) or None for 'auto' (minimize steps)
      - max_steps_hard: hard cap (int). If exceeded, overflow steps are merged into the last kept step
      - overwrite:      if False and .json/.md exist, skip regeneration
    """
    ensure_dirs(out_dir)
    out_json = os.path.join(out_dir, f"{project_name}.plan.json")
    out_md = os.path.join(out_dir, f"{project_name}.plan.md")

    # Skip if files already exist and overwrite not requested
    if os.path.exists(out_json) and os.path.exists(out_md) and not overwrite:
        print(f"⏩ Skipping: Plan already exists:\n- {out_json}\n- {out_md}")
        return

    # Generate
    plan = generate_plan_from_prd(prd_path, model=model, max_steps=max_steps)

    # Enforce hard cap (merge overflow so nothing is lost)
    if isinstance(max_steps_hard, int) and max_steps_hard > 0:
        plan = enforce_hard_cap(plan, max_steps_hard)

    # Write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    # Write Markdown
    md = plan_to_markdown(plan)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print("Plan generated.")
    print(f"Source PRD : {prd_path}")
    print(f"Model      : {model}")
    print(f"Steps      : {plan.get('meta', {}).get('total_steps', 0)}")
    print(f"Out JSON   : {out_json}")
    print(f"Out MD     : {out_md}")


def cli():
    parser = argparse.ArgumentParser(description="Create a step-by-step execution plan from a PRD Markdown file.")
    parser.add_argument("--prd", required=True, help="Path to PRD .md file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    # allow "auto" or an integer
    parser.add_argument("--max-steps", default="10", help="Target maximum number of steps (soft cap) or 'auto'")
    parser.add_argument("--max-steps-hard", type=int, default=None, help="Hard cap; overflow is merged into final step")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate plan even if it already exists")
    args = parser.parse_args()

    # parse max_steps (soft cap)
    raw = str(args.max_steps).strip().lower()
    if raw in ("auto", "none", ""):
        max_steps = None
    else:
        try:
            max_steps = int(raw)
        except ValueError:
            max_steps = None  # fallback to auto

    project_name = safe_basename_no_ext(args.prd)
    main(
        project_name=project_name,
        prd_path=args.prd,
        model=args.model,
        out_dir=args.out_dir,
        max_steps=max_steps,
        max_steps_hard=args.max_steps_hard,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        print(f"\nUnhandled error: {type(e).__name__}: {e}", file=sys.stderr)
        raise
