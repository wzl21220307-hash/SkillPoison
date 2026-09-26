#!/usr/bin/env bash
# SkillPoison release smoke check.
#
#   * imports every module of the three stage packages + the serializer
#   * byte-compiles the whole `skillpoison` package
#
# Usage (from the release root):
#     bash scripts/smoke_check.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export SKILLPOISON_DATA_ROOT="${SKILLPOISON_DATA_ROOT:-/home/zlz/test}"
PY="${PYTHON:-python3}"

echo "SkillPoison smoke check"
echo "  python              : $($PY --version 2>&1)"
echo "  SKILLPOISON_DATA_ROOT: $SKILLPOISON_DATA_ROOT"
echo

fail=0
echo "== import check =="
for f in $(find skillpoison/spin/hps skillpoison/spin/egsa skillpoison/spin/ceir \
                skillpoison/spin/serialize -name '*.py' | sort); do
  mod=$(echo "$f" | sed 's|/|.|g; s|\.py$||')
  if out=$("$PY" -c "import $mod" 2>&1); then
    echo "OK   $mod"
  else
    echo "FAIL $mod"
    echo "$out" | sed 's/^/       /'
    fail=1
  fi
done

echo
echo "== compile check =="
if "$PY" -m compileall -q skillpoison; then
  echo "OK   python3 -m compileall -q skillpoison (exit 0)"
else
  echo "FAIL compileall"
  fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "SMOKE CHECK PASSED"
else
  echo "SMOKE CHECK FAILED"
fi
exit "$fail"
