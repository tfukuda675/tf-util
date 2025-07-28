#!/bin/bash
set -eu


# GitHubランナー設定
GH_OWNER="${GH_OWNER:-your-user}"
GH_REPO="${GH_REPO:-your-repo}"
RUNNER_NAME="${GH_RUNNER_NAME:-podman-runner}"
RUNNER_WORKDIR="_work"
LABELS="self-hosted,linux,podman"
GH_PAT="${GH_PAT}"
GH_RUNNER_VERSION="${GH_RUNNER_VERSION}"

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
  curl -O -L https://github.com/actions/runner/releases/download/v${GH_RUNNER_VERSION}/actions-runner-linux-x64-${GH_RUNNER_VERSION}.tar.gz
  tar xzf actions-runner-linux-x64-${GH_RUNNER_VERSION}.tar.gz
fi

./config.sh --unattended \
  --url "https://github.com/${GH_OWNER}/${GH_REPO}" \
  --token "${TOKEN}" \
  --name "${RUNNER_NAME}" \
  --labels "${LABELS}" \
  --work "${RUNNER_WORKDIR}"


# 3. SIGTERM で remove できるようトラップを仕込む
function cleanup {
  echo "🧹 Removing runner from GitHub..."
  REMOVE_TOKEN=$(curl -s -X POST \
    -H "Authorization: token ${GH_PAT}" \
    https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/actions/runners/remove-token | jq -r .token)

  ./config.sh remove --unattended --token "$REMOVE_TOKEN"
}
trap cleanup EXIT

# 4. 起動
exec ./run.sh

