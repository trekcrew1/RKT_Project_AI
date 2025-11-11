#!/usr/bin/env python3
"""
Push an execution plan (JSON) into Azure DevOps Boards as Stories + Tasks.

Adds:
- ensure_project_exists(): creates the project if missing (using requested process)
- ensure_project_process(): attempts to migrate the project to requested process if different (best effort; safe no-op if not resolvable/allowed)
- ensure_classification_path(): auto-creates missing Area/Iteration nodes (e.g., 'Project\\Product', 'Project\\Sprint 1')
- Parent all Stories under a Feature (by id or by title; create if missing)

Usage (programmatic):
    run_push_to_azure(
        plan="out/plans/TimeTracker.plan.json",
        org="trekcrew",
        project="TimeTracker",
        pat=os.getenv("PAT_TOKEN",""),
        process="agile",
        parent_feature_title="MVP — Core Time Tracking",  # or parent_feature_id=12345
        ...
    )
"""

import os
import sys
import json
import argparse
import base64
import time
import urllib.parse
from typing import Any, Dict, List, Optional
import requests

from dotenv import load_dotenv
load_dotenv()

API_VERSION = "7.1"      # general
WIT_API_VERSION = "7.1"  # WIT endpoints
CORE_API_VERSION = "7.1"

PROCESS_STORY_TYPE = {
    "agile": "User Story",
    "scrum": "Product Backlog Item",
    "cmmi": "Requirement",
    "basic": "Issue",
}

# ----------------------- helpers: auth & urls -----------------------

def b64_pat(pat: str) -> str:
    token = f":{pat}".encode("utf-8")
    return base64.b64encode(token).decode("utf-8")

def wit_url(org: str, project: str, witype: str) -> str:
    enc = urllib.parse.quote(witype, safe="")
    return f"https://dev.azure.com/{org}/{project}/_apis/wit/workitems/${enc}?api-version={WIT_API_VERSION}"

def wi_url(org: str, id_: int) -> str:
    return f"https://dev.azure.com/{org}/_apis/wit/workItems/{id_}?api-version={WIT_API_VERSION}"

def get_work_item_url(org: str, id_: int) -> str:
    # bare URL for relations (no api-version query needed)
    return f"https://dev.azure.com/{org}/_apis/wit/workItems/{id_}"

def wiql_url(org: str, project: str) -> str:
    return f"https://dev.azure.com/{org}/{urllib.parse.quote(project)}/_apis/wit/wiql?api-version={WIT_API_VERSION}"

def projects_url(org: str) -> str:
    return f"https://dev.azure.com/{org}/_apis/projects?api-version={CORE_API_VERSION}"

def project_url(org: str, project: str) -> str:
    return f"https://dev.azure.com/{org}/_apis/projects/{urllib.parse.quote(project)}?api-version={CORE_API_VERSION}"

def processes_list_url(org: str) -> str:
    # Core/Processes – list available processes in the org
    return f"https://dev.azure.com/{org}/_apis/process/processes?api-version={CORE_API_VERSION}"

def project_process_migration_url(org: str, project: str) -> str:
    # WIT Project Process Migration
    return f"https://dev.azure.com/{org}/{urllib.parse.quote(project)}/_apis/wit/projectprocessmigration?api-version={WIT_API_VERSION}"

# ----------------------- org/project/process checks -----------------------

def get_project(session: requests.Session, org: str, project: str) -> Optional[Dict[str, Any]]:
    r = session.get(project_url(org, project))
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404:
        return None
    raise RuntimeError(f"Failed to get project '{project}': {r.status_code} {r.text}")

def list_processes(session: requests.Session, org: str) -> List[Dict[str, Any]]:
    r = session.get(processes_list_url(org))
    if r.status_code != 200:
        raise RuntimeError(f"Failed to list processes: {r.status_code} {r.text}")
    return r.json().get("value", [])

def resolve_process_id(session: requests.Session, org: str, desired_name: str) -> Optional[str]:
    """Find process id by name (case-insensitive). Returns process id or None."""
    desired = (desired_name or "").strip().lower()
    for p in list_processes(session, org):
        name = p.get("name", "").strip().lower()
        pid  = p.get("typeId") or p.get("id")
        if name == desired and pid:
            return str(pid)
    return None

def wait_for_project_well_formed(session: requests.Session, org: str, project: str, timeout_s: int = 180) -> None:
    """Polls project state until wellFormed or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = get_project(session, org, project)
        if info and info.get("state") == "wellFormed":
            return
        time.sleep(2)
    raise TimeoutError(f"Project '{project}' was not ready after {timeout_s}s")

def _explain_create_project_auth_failure(resp_text: str) -> str:
    return (
        "Authorization failed creating the project.\n"
        "Checklist:\n"
        "  • PAT belongs to a user in the organization (signed into https://dev.azure.com/{org}).\n"
        "  • User has permission to create projects (ideally in Project Collection Administrators),\n"
        "    or org setting allows project creation.\n"
        "  • PAT scopes include: Project and Team (Read & write), Work Items (Read & write), Process (Read).\n"
        "  • If your org uses AAD, ensure Conditional Access isn’t blocking PAT usage.\n"
        f"Server said: {resp_text[:400]}..."
    )

def create_project(session: requests.Session, org: str, project: str, process_id: str, visibility: str = "private") -> Dict[str, Any]:
    """Creates a project with the given process id. Try stable then preview."""
    body = {
        "name": project,
        "description": f"Auto-created by script for project '{project}'",
        "visibility": visibility,  # 'private' or 'public'
        "capabilities": {
            "versioncontrol": {"sourceControlType": "Git"},
            "processTemplate": {"templateTypeId": process_id}
        }
    }

    def _post_with_version(api_version: str):
        url = f"https://dev.azure.com/{org}/_apis/projects?api-version={api_version}"
        return session.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(body))

    r = _post_with_version("7.1")
    if r.status_code in (200, 202):
        print(f"↳ Project create queued: {project} (process id: {process_id})")
        wait_for_project_well_formed(session, org, project)
        print(f"✅ Project ready: {project}")
        return get_project(session, org, project) or {}

    if r.status_code in (401, 403):
        raise RuntimeError(_explain_create_project_auth_failure(r.text))

    r2 = _post_with_version("7.1-preview.4")
    if r2.status_code in (200, 202):
        print(f"↳ Project create (preview) queued: {project} (process id: {process_id})")
        wait_for_project_well_formed(session, org, project)
        print(f"✅ Project ready: {project}")
        return get_project(session, org, project) or {}

    if r2.status_code in (401, 403):
        raise RuntimeError(_explain_create_project_auth_failure(r2.text))

    raise RuntimeError(
        f"Project create failed (stable={r.status_code}, preview={r2.status_code}). "
        f"Stable says: {r.text[:400]}...  Preview says: {r2.text[:400]}..."
    )

def ensure_project_exists(session: requests.Session, org: str, project: str, desired_process_name: str) -> Dict[str, Any]:
    info = get_project(session, org, project)
    if info:
        print(f"Project '{project}' found (state={info.get('state')}).")
        return info
    print(f"Project '{project}' not found. Creating...")
    pid = resolve_process_id(session, org, desired_process_name)
    if not pid:
        raise RuntimeError(
            f"Could not find process '{desired_process_name}' in org '{org}'. "
            f"Verify the process exists under Organization Settings → Process."
        )
    return create_project(session, org, project, pid)

def get_project_process_id(project_info: Dict[str, Any]) -> Optional[str]:
    caps = project_info.get("capabilities") or {}
    pt = caps.get("processTemplate") or {}
    return pt.get("templateTypeId")

def ensure_project_process(session: requests.Session, org: str, project: str, desired_process_name: str) -> None:
    """
    Best-effort: only attempt migration if we can resolve the desired process ID AND
    the current process differs. Otherwise, skip with a warning/no-op.
    """
    current = get_project(session, org, project)
    if not current:
        raise RuntimeError(f"Project '{project}' vanished during process check.")

    current_pid = get_project_process_id(current)
    desired_pid = resolve_process_id(session, org, desired_process_name)
    if not desired_pid:
        print("ℹ️  Skipping process change: could not resolve desired process id.", file=sys.stderr)
        return

    if current_pid and current_pid.lower() == desired_pid.lower():
        # already using desired process
        return

    print(f"Attempting to migrate project '{project}' process → '{desired_process_name}'...")
    r = session.post(
        project_process_migration_url(org, project),
        headers={"Content-Type": "application/json"},
        data=json.dumps({"processId": desired_pid}),
    )

    if r.status_code in (200, 202):
        print("↳ Migration requested.")
        wait_for_project_well_formed(session, org, project)
        print("✅ Project process migration complete.")
        return

    # Migration refused — explain and proceed
    print(
        f"⚠️  Project process migration was not applied (status {r.status_code}). "
        "Azure often restricts cross-base migrations; consider using the UI or an inherited process.",
        file=sys.stderr
    )
    print(f"Details: {r.text[:400]}...", file=sys.stderr)

# ----------------------- Area & Iteration helpers -----------------------

def fetch_classification_tree(session: requests.Session, org: str, project: str, kind: str, depth: int = 10) -> Dict[str, Any]:
    url = f"https://dev.azure.com/{org}/{project}/_apis/wit/classificationnodes/{kind}?$depth={depth}&api-version={WIT_API_VERSION}"
    r = session.get(url, headers={"Accept": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {kind} tree: {r.status_code} {r.text}")
    return r.json()

def flatten_nodes(root: Dict[str, Any], path_prefix: Optional[str] = None) -> List[str]:
    current_name = root.get("name", "")
    if not current_name:
        return []
    current_path = current_name if not path_prefix else f"{path_prefix}\\{current_name}"
    paths = [current_path]
    for child in root.get("children", []) or []:
        paths.extend(flatten_nodes(child, current_path))
    return paths

def validate_path(candidate: Optional[str], available_paths: List[str], strict: bool, label: str) -> Optional[str]:
    if not candidate:
        return None
    if candidate in available_paths:
        return candidate
    lower_set = {p.lower() for p in available_paths}
    if candidate.lower() in lower_set:
        proper = next(p for p in available_paths if p.lower() == candidate.lower())
        msg = (f"{label} path case mismatch. You passed '{candidate}'. Use '{proper}'.")
    else:
        msg = (f"{label} path '{candidate}' does not exist.\n"
               f"Valid {label.lower()} paths include (first 15):\n  - " +
               "\n  - ".join(available_paths[:15]) +
               ("\n  ... (truncated)" if len(available_paths) > 15 else ""))
    if strict:
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"⚠️  {msg}\nProceeding without setting {label.lower()} path.", file=sys.stderr)
        return None

# ---- auto-create Area/Iteration paths ----

def _post_classification_node(session: requests.Session, url: str, name: str) -> None:
    resp = session.post(url, headers={"Content-Type": "application/json"}, data=json.dumps({"name": name}))
    # 200/201 created, 409 already exists (race) – both fine
    if resp.status_code not in (200, 201, 409):
        raise RuntimeError(f"Failed to create classification node '{name}': {resp.status_code} {resp.text}")

def create_area_node(session: requests.Session, org: str, project: str, parent_path: str, name: str) -> None:
    base = f"https://dev.azure.com/{org}/{project}/_apis/wit/classificationnodes/areas"
    url = base
    if parent_path:
        clean_parent = parent_path.strip("\\")
        url += f"/{urllib.parse.quote(clean_parent, safe='')}"
    url += f"?api-version={WIT_API_VERSION}"
    _post_classification_node(session, url, name)

def create_iteration_node(session: requests.Session, org: str, project: str, parent_path: str, name: str) -> None:
    base = f"https://dev.azure.com/{org}/{project}/_apis/wit/classificationnodes/iterations"
    url = base
    if parent_path:
        clean_parent = parent_path.strip("\\")
        url += f"/{urllib.parse.quote(clean_parent, safe='')}"
    url += f"?api-version={WIT_API_VERSION}"
    _post_classification_node(session, url, name)

def ensure_classification_path(session: requests.Session, org: str, project: str, kind: str, full_path: str) -> None:
    """
    kind: 'areas' or 'iterations'. Creates each segment if missing.
    Accepts full path like 'Project\\Sub\\Child'. Root project node is assumed to exist.
    """
    if not full_path:
        return
    segments = full_path.strip("\\").split("\\")
    if not segments:
        return
    # If the first segment equals the project, don't try to create it; start after it.
    i = 1 if segments[0].lower() == project.lower() else 0
    parent = ""  # relative parent path under the classification root
    for seg in segments[i:]:
        if kind == "areas":
            create_area_node(session, org, project, parent, seg)
        else:
            create_iteration_node(session, org, project, parent, seg)
        parent = f"{parent}\\{seg}" if parent else seg

# ----------------------- Feature helpers (parenting Stories) -----------------------

def find_feature_by_title(session: requests.Session, org: str, project: str, title: str) -> Optional[int]:
    """
    Returns the ID of a Feature with the exact title in this project, or None.
    """
    safe_title = (title or "").replace("'", "''")
    q = (
        "SELECT [System.Id] "
        "FROM WorkItems "
        f"WHERE [System.TeamProject] = '{project}' "
        "AND [System.WorkItemType] = 'Feature' "
        f"AND [System.Title] = '{safe_title}'"
    )
    r = session.post(
        wiql_url(org, project),
        headers={"Content-Type": "application/json"},
        data=json.dumps({"query": " ".join(q.split())})
    )
    if r.status_code != 200:
        raise RuntimeError(f"WIQL query failed: {r.status_code} {r.text}")
    ids = [it["id"] for it in r.json().get("workItems", [])]
    return int(ids[0]) if ids else None

def create_feature(
    session: requests.Session,
    org: str,
    project: str,
    title: str,
    *,
    description_html: Optional[str] = None,
    area_path: Optional[str] = None,
    iteration_path: Optional[str] = None,
    assigned_to: Optional[str] = None,
    tags: Optional[str] = None,
) -> int:
    feature = create_work_item(
        session=session,
        org=org,
        project=project,
        witype="Feature",
        title=title,
        description_html=description_html,
        area_path=area_path,
        iteration_path=iteration_path,
        assigned_to=assigned_to,
        effort=None,
        tags=tags,
        relations=None,
    )
    return int(feature["id"])

# ----------------------- WIT create helpers -----------------------

def create_work_item(
    session: requests.Session,
    org: str,
    project: str,
    witype: str,
    title: str,
    description_html: Optional[str] = None,
    acceptance_criteria_text: Optional[str] = None,  # NEW: proper AC field
    area_path: Optional[str] = None,
    iteration_path: Optional[str] = None,
    assigned_to: Optional[str] = None,
    effort: Optional[float] = None,
    tags: Optional[str] = None,
    relations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Creates a work item (Story/Feature/Task). If acceptance_criteria_text is provided,
    it is written to Microsoft.VSTS.Common.AcceptanceCriteria (multi-line plain text).
    """
    body = [{"op": "add", "path": "/fields/System.Title", "value": title}]
    if description_html:
        body.append({"op": "add", "path": "/fields/System.Description", "value": description_html})
    if acceptance_criteria_text:
        body.append({"op": "add", "path": "/fields/Microsoft.VSTS.Common.AcceptanceCriteria", "value": acceptance_criteria_text})
    if area_path:
        body.append({"op": "add", "path": "/fields/System.AreaPath", "value": area_path})
    if iteration_path:
        body.append({"op": "add", "path": "/fields/System.IterationPath", "value": iteration_path})
    if assigned_to:
        body.append({"op": "add", "path": "/fields/System.AssignedTo", "value": assigned_to})
    if effort is not None:
        body.append({"op": "add", "path": "/fields/Microsoft.VSTS.Scheduling.Effort", "value": float(effort)})
    if tags:
        body.append({"op": "add", "path": "/fields/System.Tags", "value": tags})
    if relations:
        for rel in relations:
            body.append({"op": "add", "path": "/relations/-", "value": rel})

    resp = session.patch(
        wit_url(org, project, witype),
        headers={"Content-Type": "application/json-patch+json", "Accept": "application/json"},
        data=json.dumps(body)
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Create {witype} failed: {resp.status_code} {resp.text}")
    return resp.json()

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def md_to_basic_html(md: str) -> str:
    if not md:
        return ""
    esc = html_escape(md)
    return f"<pre>{esc}</pre>"

# ----------------------- Programmatic entrypoint -----------------------

def run_push_to_azure(
    *,
    plan: str,
    org: str,
    project: str,
    pat: Optional[str] = None,
    process: str = "agile",
    work_item_type: Optional[str] = None,
    area: Optional[str] = None,
    iteration: Optional[str] = None,
    strict_area: bool = False,
    strict_iteration: bool = False,
    assignee: Optional[str] = None,
    tags: Optional[str] = None,
    task_estimate: Optional[float] = None,
    overwrite: bool = False,
    receipt_path: Optional[str] = None,
    # NEW: parent feature support
    parent_feature_id: Optional[int] = None,
    parent_feature_title: Optional[str] = None,
    create_feature_if_missing: bool = True,
) -> Dict[str, Any]:
    """
    Programmatic API to push a plan into Azure DevOps.

    Returns a dict: {"stories": [ids...], "tasks": [ids...], "work_item_type": "<type>", "receipt": "<path or None>"}
    """
    if not pat:
        pat = os.getenv("PAT_TOKEN", "")
    if not pat:
        raise RuntimeError("Provide a PAT via argument 'pat' or PAT_TOKEN env var.")

    if not os.path.exists(plan):
        raise FileNotFoundError(f"Plan file not found: {plan}")

    # Where to record a receipt of created items
    if receipt_path is None:
        base, _ = os.path.splitext(plan)
        receipt_path = f"{base}.azure_import.json"

    # Skip if receipt exists and overwrite not requested
    if os.path.exists(receipt_path) and not overwrite:
        print(f"⏩ Skipping Azure import (receipt exists): {receipt_path}")
        return {"stories": [], "tasks": [], "work_item_type": None, "receipt": receipt_path}

    # Session with PAT auth
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Basic {b64_pat(pat)}",
        "Accept": "application/json"
    })

    # ---------- Ensure project exists & process is set ----------
    print(f"Checking organization '{org}' access...")
    _ = list_processes(session, org)  # auth + org reachability check

    # Ensure project
    _ = ensure_project_exists(session, org, project, desired_process_name=process)

    # Best-effort process alignment (safe no-op if not resolvable)
    ensure_project_process(session, org, project, desired_process_name=process)

    # ---------- Read plan ----------
    data = json.loads(open(plan, "r", encoding="utf-8").read())
    plan_title = data.get("meta", {}).get("title") or "Imported Plan"
    steps: List[Dict[str, Any]] = data.get("steps", [])
    if not steps:
        print("No steps found in plan; nothing to import.")
        with open(receipt_path, "w", encoding="utf-8") as rf:
            json.dump({"stories": [], "tasks": [], "note": "No steps"}, rf, indent=2)
        return {"stories": [], "tasks": [], "work_item_type": None, "receipt": receipt_path}

    # Determine story type
    desired_type = work_item_type or PROCESS_STORY_TYPE.get((process or "").lower(), "User Story")
    print(f"ℹ️  Desired type: {desired_type} (process={process})")

    def list_work_item_types_in_project(session: requests.Session, org: str, project: str) -> List[str]:
        url = f"https://dev.azure.com/{org}/{project}/_apis/wit/workitemtypes?api-version={WIT_API_VERSION}"
        r = session.get(url, headers={"Accept": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"Failed to list work item types: {r.status_code} {r.text}")
        data = r.json()
        return [it.get("name") for it in data.get("value", []) if it.get("name")]

    def pick_best_type(desired: str, available: List[str]) -> Optional[str]:
        avail_map = {t.lower(): t for t in available}
        if desired.lower() in avail_map:
            return avail_map[desired.lower()]
        prefs = ["User Story", "Product Backlog Item", "Issue", "Requirement", "Feature", "Task"]
        for p in prefs:
            if p.lower() in avail_map:
                return avail_map[p.lower()]
        return available[0] if available else None

    available_types = list_work_item_types_in_project(session, org, project)
    chosen_type = pick_best_type(desired_type, available_types)
    if not chosen_type:
        raise RuntimeError(f"No work item types available in project '{project}'.")
    if chosen_type.lower() != desired_type.lower():
        print(f"⚠️  '{desired_type}' not found in project. Using '{chosen_type}' "
              f"(Available: {', '.join(available_types)})")
    else:
        print(f"✅ Using work item type: {chosen_type}")

    # ---------- Validate / Create Area & Iteration ----------
    area_to_use = area
    iter_to_use = iteration

    # fetch trees (first pass)
    try:
        areas_tree = fetch_classification_tree(session, org, project, "areas", depth=10)
        area_paths = flatten_nodes(areas_tree)
    except Exception as e:
        print(f"⚠️  Could not fetch Areas tree ({e}). Proceeding without validation.", file=sys.stderr)
        area_paths = []

    try:
        iters_tree = fetch_classification_tree(session, org, project, "iterations", depth=10)
        iter_paths = flatten_nodes(iters_tree)
    except Exception as e:
        print(f"⚠️  Could not fetch Iterations tree ({e}). Proceeding without validation.", file=sys.stderr)
        iter_paths = []

    # Auto-create missing paths if provided
    if area and (not area_paths or area not in area_paths):
        print(f"ℹ️  Area path '{area}' missing — creating…")
        ensure_classification_path(session, org, project, "areas", area)
        # refresh area paths
        areas_tree = fetch_classification_tree(session, org, project, "areas", depth=10)
        area_paths = flatten_nodes(areas_tree)

    if iteration and (not iter_paths or iteration not in iter_paths):
        print(f"ℹ️  Iteration path '{iteration}' missing — creating…")
        ensure_classification_path(session, org, project, "iterations", iteration)
        # refresh iteration paths
        iters_tree = fetch_classification_tree(session, org, project, "iterations", depth=10)
        iter_paths = flatten_nodes(iters_tree)

    # Final validation (respects strict flags)
    if area_paths:
        area_to_use = validate_path(area, area_paths, strict_area, "Area")
    if iter_paths:
        iter_to_use = validate_path(iteration, iter_paths, strict_iteration, "Iteration")

    # ---------- Resolve/Create parent Feature (optional) ----------
    resolved_feature_id: Optional[int] = None
    if parent_feature_id:
        resolved_feature_id = int(parent_feature_id)
        print(f"Parent Feature (by id): #{resolved_feature_id}")
    elif parent_feature_title:
        found_id = find_feature_by_title(session, org, project, parent_feature_title)
        if found_id:
            resolved_feature_id = found_id
            print(f"Parent Feature found: #{resolved_feature_id} — {parent_feature_title}")
        elif create_feature_if_missing:
            print(f"Parent Feature '{parent_feature_title}' not found. Creating...")
            resolved_feature_id = create_feature(
                session=session,
                org=org,
                project=project,
                title=parent_feature_title,
                description_html=md_to_basic_html(f"Auto-created for plan import: {plan_title}"),
                area_path=area_to_use,
                iteration_path=iter_to_use,
                assigned_to=assignee,
                tags=tags,
            )
            print(f"  ✓ Created Feature #{resolved_feature_id}")
        else:
            print(f"⚠️  Parent Feature '{parent_feature_title}' not found, and create_feature_if_missing=False; proceeding without a parent.")

    # ---------- Create Stories & Tasks ----------
    print(f"Importing plan: {plan_title}  ->  {len(steps)} stories")
    created = {"stories": [], "tasks": []}

    # def story_description_html_from_step(step: Dict[str, Any]) -> str:
    #     """
    #     Build the Description (HTML) WITHOUT Acceptance Criteria.
    #     """
    #     blocks = []
    #     if step.get("summary"):
    #         blocks.append(f"**Summary**\n\n{step['summary']}")
    #     if step.get("why_it_matters"):
    #         blocks.append(f"**Why it matters**\n\n{step['why_it_matters']}")
    #     if step.get("instructions"):
    #         blocks.append(f"**Instructions**\n\n{step['instructions']}")
    #     if step.get("links"):
    #         link_lines = "\n".join(
    #             f"- [{l.get('label','link')}]({l.get('url','')})"
    #             for l in step["links"] if l.get("url")
    #         )
    #         if link_lines:
    #             blocks.append("**References**\n\n" + link_lines)
    #     if step.get("deliverables"):
    #         deliv_lines = "\n".join(f"- {d}" for d in step["deliverables"])
    #         blocks.append("**Deliverables**\n\n" + deliv_lines)
    #     return md_to_basic_html("\n\n".join(blocks))

    def story_description_html_from_step(step: Dict[str, Any]) -> str:
        """
        Build the Story Description (HTML) from the execution-plan schema fields.
        NOTE: Acceptance Criteria is handled separately in the dedicated AC field.
        """
        def bullets(arr):
            return "\n".join(f"- {str(x)}" for x in (arr or []) if str(x).strip())

        blocks = []

        # Objective
        if step.get("objective"):
            blocks.append(f"**Objective**\n\n{step['objective']}")

        # Prerequisites / Inputs / Outputs
        if step.get("prerequisites"):
            blocks.append("**Prerequisites**\n\n" + bullets(step["prerequisites"]))
        if step.get("inputs"):
            blocks.append("**Inputs**\n\n" + bullets(step["inputs"]))
        if step.get("outputs"):
            blocks.append("**Outputs/Artifacts**\n\n" + bullets(step["outputs"]))

        # Detailed instructions (numbered)
        if step.get("detailed_instructions"):
            items = [f"{i}. {str(x)}" for i, x in enumerate(step["detailed_instructions"], start=1) if str(x).strip()]
            if items:
                blocks.append("**Detailed Instructions (do each in order):**\n\n" + "\n".join(items))

        # Ownership, dependencies, time
        owner = step.get("owner_role") or "Owner"
        collab = ", ".join(step.get("collaborators", []) or [])
        owner_line = f"**Owner:** {owner}" + (f" — Collaborators: {collab}" if collab else "")
        blocks.append(owner_line)

        if step.get("dependencies"):
            deps = ", ".join(str(d) for d in step.get("dependencies") if str(d).strip())
            if deps:
                blocks.append(f"**Depends on:** {deps}")

        tmin = step.get("time_estimate_hours_min")
        tmax = step.get("time_estimate_hours_max")
        if isinstance(tmin, (int, float)) or isinstance(tmax, (int, float)):
            tmin = float(tmin or 0)
            tmax = float(tmax if tmax is not None else tmin)
            blocks.append(f"**Time Estimate:** ~{tmin}-{tmax} hours")

        # Notes
        if step.get("notes"):
            blocks.append(f"**Notes**\n\n{step['notes']}")

        return md_to_basic_html("\n\n".join(blocks))


    def story_acceptance_criteria_text_from_step(step: Dict[str, Any]) -> Optional[str]:
        """
        Build plain-text acceptance criteria (multi-line) for the dedicated AC field.
        """
        ac = step.get("acceptance_criteria") or []
        if not ac:
            return None
        lines = [f"- {str(a)}" for a in ac if str(a).strip()]
        return "\n".join(lines) if lines else None

    for idx, step in enumerate(steps, start=1):
        st_title = step.get("title") or f"Step {idx}"
        story_desc_html = story_description_html_from_step(step)
        story_ac_text = story_acceptance_criteria_text_from_step(step)

        # Story effort (avg hours / 8)
        effort = None
        if isinstance(step.get("estimates"), dict):
            lo = step["estimates"].get("low_hours")
            hi = step["estimates"].get("high_hours")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                effort = round((float(lo) + float(hi)) / 2.0 / 8.0, 2)

        # Parent Story → Feature if provided
        relations = None
        if resolved_feature_id:
            relations = [{
                "rel": "System.LinkTypes.Hierarchy-Reverse",  # child -> parent
                "url": get_work_item_url(org, resolved_feature_id)
            }]

        story = create_work_item(
            session=session,
            org=org,
            project=project,
            witype=chosen_type,
            title=st_title,
            description_html=story_desc_html,
            acceptance_criteria_text=story_ac_text,   # <-- write to AC field
            area_path=area_to_use,
            iteration_path=iter_to_use,
            assigned_to=assignee,
            effort=effort,
            tags=tags,
            relations=relations,  # link Story to Feature when set
        )
        story_id = story["id"]
        created["stories"].append(story_id)
        print(f"  ✓ {chosen_type} #{story_id}: {st_title}")

        # ---------- TASKS ----------
        actions = step.get("actions")
        if not actions and isinstance(step.get("detailed_instructions"), list):
            actions = []
            for i, instr in enumerate(step["detailed_instructions"], start=1):
                if not isinstance(instr, str) or not instr.strip():
                    continue
                title_snippet = instr.strip().split("\n", 1)[0][:80]
                actions.append({
                    "title": f"{st_title} — Task {i}: {title_snippet}",
                    "instructions": instr.strip()
                })
        actions = actions or []

        # Decide per-task estimate
        per_task_est = None
        if actions:
            if task_estimate is not None:
                per_task_est = float(task_estimate)
            else:
                lo = step.get("time_estimate_hours_min")
                hi = step.get("time_estimate_hours_max")
                if isinstance(lo, (int, float)) or isinstance(hi, (int, float)):
                    lo = float(lo or 0)
                    hi = float(hi or lo)
                    avg_hours = (lo + hi) / 2.0 if (lo or hi) else 0.0
                    n = max(1, len(actions))
                    per_task_est = round(avg_hours / n, 2)

        for a_idx, action in enumerate(actions, start=1):
            t_title = action.get("title") or f"{st_title} — Task {a_idx}"
            t_desc_blocks = []
            if action.get("instructions"):
                t_desc_blocks.append(f"**Instructions**\n\n{action['instructions']}")
            if action.get("links"):
                links = "\n".join(
                    f"- [{l.get('label','link')}]({l.get('url','')})"
                    for l in action["links"] if l.get("url")
                )
                if links:
                    t_desc_blocks.append("**References**\n\n" + links)
            if action.get("acceptance_criteria"):
                ac = "\n".join(f"- {x}" for x in action["acceptance_criteria"])
                t_desc_blocks.append("**Acceptance Criteria**\n\n" + ac)

            task_html = md_to_basic_html("\n\n".join(t_desc_blocks))

            relations_task = [{
                "rel": "System.LinkTypes.Hierarchy-Reverse",  # child -> parent
                "url": wi_url(org, story_id)
            }]

            task = create_work_item(
                session=session,
                org=org,
                project=project,
                witype="Task",
                title=t_title,
                description_html=task_html,
                # tasks typically do not have a separate Acceptance Criteria field
                acceptance_criteria_text=None,
                area_path=area_to_use,
                iteration_path=iter_to_use,
                assigned_to=assignee,
                effort=None,
                tags=tags,
                relations=relations_task
            )

            # OriginalEstimate/RemainingWork for Tasks (hours)
            est_hours = None
            if isinstance(action.get("estimate_hours"), (int, float)):
                est_hours = float(action["estimate_hours"])
            elif per_task_est is not None:
                est_hours = float(per_task_est)

            if est_hours is not None:
                patch = [
                    {"op": "add", "path": "/fields/Microsoft.VSTS.Scheduling.OriginalEstimate", "value": est_hours},
                    {"op": "add", "path": "/fields/Microsoft.VSTS.Scheduling.RemainingWork", "value": est_hours},
                ]
                r2 = session.patch(
                    wi_url(org, task["id"]),
                    headers={"Content-Type": "application/json-patch+json", "Accept": "application/json"},
                    data=json.dumps(patch)
                )
                if r2.status_code >= 300:
                    print(f"    ! Failed to set estimate on Task #{task['id']}: {r2.status_code} {r2.text}", file=sys.stderr)

            created["tasks"].append(task["id"])
            print(f"    - Task #{task['id']}: {t_title}")

    # Receipt to allow skip on next run
    try:
        with open(receipt_path, "w", encoding="utf-8") as rf:
            json.dump({
                "plan": os.path.abspath(plan),
                "org": org,
                "project": project,
                "work_item_type": chosen_type,
                "stories": created["stories"],
                "tasks": created["tasks"],
                "parent_feature_id": resolved_feature_id,
            }, rf, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to write receipt {receipt_path}: {e}", file=sys.stderr)

    print(f"\nDone. Created {len(created['stories'])} {chosen_type}(s) and {len(created['tasks'])} tasks.")
    return {"stories": created["stories"], "tasks": created["tasks"], "work_item_type": chosen_type, "receipt": receipt_path}

# ----------------------- CLI (optional) -----------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import plan JSON into Azure DevOps as Stories + Tasks.")
    ap.add_argument("--plan", required=True, help="Path to plan JSON generated by make_plan_from_prd.py")
    ap.add_argument("--org", required=True, help="Azure DevOps organization name (in dev.azure.com/<org>)")
    ap.add_argument("--project", required=True, help="Azure DevOps project name")
    ap.add_argument("--pat", default=os.getenv("PAT_TOKEN", ""), help="Personal Access Token (or set PAT_TOKEN env var)")
    ap.add_argument("--process", default="agile", help="Desired process name (e.g., agile, scrum, basic)")
    ap.add_argument("--work-item-type", default=None, help="Override the story type explicitly")
    ap.add_argument("--area", default=None, help="Area Path")
    ap.add_argument("--iteration", default=None, help="Iteration Path")
    ap.add_argument("--strict-area", action="store_true")
    ap.add_argument("--strict-iteration", action="store_true")
    ap.add_argument("--assignee", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--task-estimate", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true", help="Import again even if a receipt exists (default: skip)")
    ap.add_argument("--receipt", default=None, help="Override receipt path (defaults next to the plan)")
    # NEW: feature parenting
    ap.add_argument("--parent-feature-id", type=int, default=None, help="Parent all stories under this Feature ID")
    ap.add_argument("--parent-feature-title", default=None, help="Find or create a Feature with this title and parent stories under it")
    ap.add_argument("--no-create-feature", action="store_true", help="Do not create the Feature if --parent-feature-title is not found")
    return ap.parse_args()

def main():
    args = parse_args()
    run_push_to_azure(
        plan=args.plan,
        org=args.org,
        project=args.project,
        pat=args.pat or os.getenv("PAT_TOKEN", ""),
        process=args.process,
        work_item_type=args.work_item_type,
        area=args.area,
        iteration=args.iteration,
        strict_area=args.strict_area,
        strict_iteration=args.strict_iteration,
        assignee=args.assignee,
        tags=args.tags,
        task_estimate=args.task_estimate,
        overwrite=args.overwrite,
        receipt_path=args.receipt,
        parent_feature_id=args.parent_feature_id,
        parent_feature_title=args.parent_feature_title,
        create_feature_if_missing=not args.no_create_feature,
    )

if __name__ == "__main__":
    main()
