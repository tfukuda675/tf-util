#! /users/tfuku/tools/miniforge3/envs/py311/bin/python3

from github import Github
from datetime import timezone

import json
import os

TOKEN = None

home_dir = os.path.expanduser("~")

with open(f"{home_dir}/.github/token.json") as f:
    TOKEN = json.load(f)["token"]

REPO  = "tfukuda675/tf-util"                      # 例: "octocat/Hello-World"

gh   = Github(TOKEN, per_page=100)          # per_page=101 はページ当たり最大件数
repo = gh.get_repo(REPO)

print(f"# Issues in {REPO}")
for issue in repo.get_issues(state="open"): # state="all" で全件
    created = issue.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"{issue.number:4}  {issue.title:<60.57}  {issue.user.login:15}  {created}  {issue.state}")
