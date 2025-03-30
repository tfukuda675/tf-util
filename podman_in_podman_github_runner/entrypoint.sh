#!/bin/bash
set -eu


# GitHubランナー設定
GH_OWNER="${GH_OWNER:-your-user}"
GH_REPO="${GH_REPO:-your-repo}"
RUNNER_NAME="${GH_RUNNER_NAME:-podman-runner}"
RUNNER_WORKDIR="_work"
LABELS="self-hosted,linux,podman"
GH_PAT="${GH_PAT}"

# トークン取得
TOKEN=$(curl -s -X POST \
  -H "Authorization: token ${GH_PAT}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/actions/runners/registration-token" \
  | jq -r .token)

if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
  echo "❌ Failed to get runner token from GitHub API"
  exit 1
fi

# ランナーセットアップ
cd /home/runner
mkdir -p actions-runner && cd actions-runner

if [[ ! -f ./config.sh ]]; then
  echo "▶ Downloading GitHub Actions Runner..."
  curl -O -L https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz
  tar xzf actions-runner-linux-x64-2.316.0.tar.gz
fi

./config.sh --unattended \
  --url "https://github.com/${GH_OWNER}/${GH_REPO}" \
  --token "${TOKEN}" \
  --name "${RUNNER_NAME}" \
  --labels "${LABELS}" \
  --work "${RUNNER_WORKDIR}"

exec ./run.sh



## 古い記述


# Create a folder
#echo "Create actions-runner directory..."
#mkdir actions-runner && cd actions-runner

# Download the latest runner package
#echo "Downloading GitHub Actions Runner..."
#curl -o actions-runner-linux-x64-2.323.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz



# Optional: Validate the hash
#echo "0dbc9bf5a58620fc52cb6cc0448abcca964a8d74b5f39773b7afcad9ab691e19  actions-runner-linux-x64-2.323.0.tar.gz" | shasum -a 256 -c
# Extract the installer
#tar xzf ./actions-runner-linux-x64-2.323.0.tar.gz


# GitHubのURLとTOKENは環境変数で渡す
#if [ -z "$GH_RUNNER_URL" ] || [ -z "$GH_RUNNER_TOKEN" ]; then
#  echo "❌ GH_RUNNER_URL and GH_RUNNER_TOKEN must be provided as environment variables."
#   exit 1
#fi

# コンフィグして起動
#./config.sh --unattended \
#  --url "$GH_RUNNER_URL" \
#  --token "$GH_RUNNER_TOKEN" \
#  --name "${GH_RUNNER_NAME:-podman-runner}" \
#  --work "_work"
#
## ランナー起動
#exec ./run.sh
