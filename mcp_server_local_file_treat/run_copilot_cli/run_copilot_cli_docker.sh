#! /bin/bash

set -eu

docker build -t github_copilot_cli .
docker run -it --volume=.:/app github_copilot_cli
