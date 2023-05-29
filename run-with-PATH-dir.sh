#!/usr/bin/env bash
set -euo pipefail

tmpdir=$(mktemp -d)
trap "rm -rf '$tmpdir'" EXIT

if [[ $# -ge 2 && ( $1 = '-h' || $1 = '--help' ) ]]; then
	echo "Usage: $0 [name=path/to/target] -- command..."
	exit 0
fi

while true; do
	arg=$1
	shift

	if [[ $arg = '--' ]]; then
		break
	fi
	if ! echo "$arg" | grep -F '=' >/dev/null; then
		echo "$0: no '--' found before end of remapping list"
		exit 1
	fi

	key=${arg%%=*}
	value=${arg#*=}
	if echo "$key" | grep -F '/' >/dev/null; then
		echo "$0: name cannot contain '/'"
		exit 1
	fi
	if [[ -e "$tmpdir/$key" ]]; then
		echo "$0: duplicate name '$key'"
		exit 1
	fi
	if echo "$value" | grep -F '/' >/dev/null; then
		ln -s "$(realpath "$value")" "$tmpdir/$key"
	else
		ln -s "$(which "$value")" "$tmpdir/$key"
	fi
done

env PATH="$tmpdir:$PATH" "$@"
