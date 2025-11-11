'''
Starting point for RKT_Project_AI
TODO: create a front end to enter needed parameters
    - Project Name
    - Idea File Path
'''

import make_prd_from_idea
import make_plan_from_prd
import push_plan_to_azure

# This is in push_plan_to_azure.py: TODO: figure out if put here or leave in azure code
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    project_name = 'TimeTracker'
    project_location = r'ideas\TimeTracker.json'  # raw string avoids escape issues

    # # Create Markdown from Idea
    # # -------------------------
    # overwrite_md = True
    # make_prd_from_idea.main(file_name=project_name, input_path=project_location, overwrite=overwrite_md)

    # # # Create Plan from PRD
    # # # --------------------
    # overwrite_prd = True
    # # make_plan_from_prd.main(project_name, project_location, overwrite=overwrite_prd)

    # make_plan_from_prd.main(
    #     project_name=project_name,
    #     prd_path=project_location,
    #     overwrite=overwrite_prd,
    #     max_steps= "auto",          # optional soft target
    #     max_steps_hard=20           # hard cap that preserves tasks
    # )

    # Push project to Azure
    # ---------------------
    pat = os.getenv("PAT_TOKEN", "")
    if not pat:
        raise RuntimeError("Provide a PAT via argument 'pat' or PAT_TOKEN env var.")

    # Story Variables
    plan = f'out/plans/{project_name}.plan.json'
    org = 'trekcrew'
    project = project_name
    process = 'agile'                       # we standardize on Agile
    work_item_type = None                   # let the code pick 'User Story' for Agile
    area = r'TimeTracker\Product'           # will be auto-created if missing
    iteration = r'TimeTracker\Sprint 1'     # will be auto-created if missing
    strict_area = True
    strict_iteration = True
    assignee = 'trekcrew@gmail.com'
    tags = 'AutoImport;PRD'
    task_estimate = 2
    overwrite = True

    push_plan_to_azure.run_push_to_azure(
        plan=plan,
        org=org,
        project=project,
        pat=pat,
        process=process,
        work_item_type=work_item_type,   # or None
        area=area,
        iteration=iteration,
        strict_area=strict_area,
        strict_iteration=strict_iteration,
        assignee=assignee,
        tags=tags,
        task_estimate=task_estimate,
        overwrite=overwrite,
        # NEW: choose one of the two below
        parent_feature_title="MVP — Core Time Tracking",   # will find or create
        # parent_feature_id=12345,                          # if you already know it
        create_feature_if_missing=True,
    )

if __name__ == "__main__":
    main()
