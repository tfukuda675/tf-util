#! /users/tfuku/tools/miniforge3/envs/py311/bin/python3

import os, textwrap
import json
import os
import sys

import requests

TOKEN = None
home_dir = os.path.expanduser("~")
with open(f"{home_dir}/.github/token.json") as f:
    TOKEN = json.load(f)["token"]

GITHUB_API_URL = "https://api.github.com/graphql"
#OWNER = "your-org-or-username"
OWNER = "tfukuda675"
PROJECT_NUMBER = 4  # Project number (not ID)
ORG = "your-org-name"


#       ____________________
#______/ [x] All proj       \________________
#
#query_org = """
#query($org: String!) {
#  organization(login: $org) {
#    projectsV2(first: 20) {
#      nodes {
#        id
#        title
#        number
#        closed
#        createdAt
#        updatedAt
#        url
#      }
#    }
#  }
#}
#"""
#
#query_user = """
#query($org: String!) {
#  user(login: $owner) {
#    projectsV2(first: 20) {
#      nodes {
#        id
#        title
#        number
#        closed
#        createdAt
#        updatedAt
#        url
#      }
#    }
#  }
#}
#"""
#
#variables = {
#    "owner": OWNER,
#}
#
#headers = {
#    "Authorization": f"Bearer {TOKEN}"
#}
#
#
#res = requests.post(GITHUB_API_URL, json={'query': query_user, 'variables': variables}, headers=headers)
#
#if res.status_code == 200:
#    print(res.json())
#    projects = res.json()["data"]["organization"]["projectsV2"]["nodes"]
#    # 「アクティブ（未クローズ）」なものを抽出
#    active_projects = [p for p in projects if not p["closed"]]
#    for p in active_projects:
#        print(f"[#{p['number']}] {p['title']} - Updated: {p['updatedAt']}")
#else:
#    print(f"Error {res.status_code}: {res.text}")

#       ____________________
#______/ [x]                \________________
#

query_org = """
query($owner: String!, $number: Int!) {
  organization(login: $owner) {
    projectV2(number: $number) {
      id
      title
      fields(first: 20) {
        nodes {
          ... on ProjectV2Field {
            name
            dataType
          }
        }
      }
      items(first: 10) {
        nodes {
          content {
            ... on Issue {
              title
              number
              url
            }
          }
        }
      }
    }
  }
}
"""

query_user = """
query($owner: String!, $number: Int!) {
  user(login: $owner) {
    projectV2(number: $number) {
      id
      title
      fields(first: 20) {
        nodes {
          ... on ProjectV2Field {
            name
            dataType
          }
        }
      }
      items(first: 10) {
        nodes {
          content {
            ... on Issue {
              title
              number
              url
            }
          }
        }
      }
    }
  }
}
"""

variables = {
    "owner": OWNER,
    "number": PROJECT_NUMBER
}

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

response = requests.post(GITHUB_API_URL, json={'query': query_user, 'variables': variables}, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error {response.status_code}: {response.text}")



