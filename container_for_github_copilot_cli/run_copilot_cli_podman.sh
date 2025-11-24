#! /bin/bash

set -eu

podman build -t github_copilot_cli .
podman run -it --volume=.:/app github_copilot_cli
