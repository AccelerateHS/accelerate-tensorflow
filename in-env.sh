#!/usr/bin/env bash
set -euo pipefail
env_cmd=$(sed -n '/^## Compiling using Cabal/,/^```$/!d ; /^ENV=/p' README.md)
if [[ "$(echo "$env_cmd" | wc -l)" -ne 1 ]]; then
  echo >&2 "ENV= line in README.md not found!"
  exit 1
fi

eval "$env_cmd"
env LD_LIBRARY_PATH="$ENV" "$@"
