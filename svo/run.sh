#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./release/run_video video.mp4 camera_1920_1080.yml
