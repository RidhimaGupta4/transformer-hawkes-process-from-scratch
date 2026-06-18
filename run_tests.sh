#!/usr/bin/env bash
# Runs the full test suite. Exits non-zero if any test file fails.
set -e
cd "$(dirname "$0")"

for f in tests/test_*.py; do
    echo "=== $f ==="
    python3 "$f"
    echo
done

echo "All test modules passed."
