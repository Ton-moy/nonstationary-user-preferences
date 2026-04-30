#!/usr/bin/env bash
set -euo pipefail

echo "== identity and local-path leak check =="
if grep -RInE "tonmoy|hasan|bunescu|@.*\\.edu|/users/|/projects/|#SBATCH|--partition" . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=data/synthetic \
  --exclude-dir=__pycache__ \
  --exclude='00_anonymize_audit.sh'; then
  echo "LEAK FOUND"
  exit 1
fi

echo "== forbidden artifact check =="
if find . \( -name '*.out' -o -name '*.err' -o -name 'slurm-*' \) | grep .; then
  echo "FORBIDDEN ARTIFACT FOUND"
  exit 1
fi

echo "== synthetic layout check =="
if find data/synthetic/phat_driven_top2 data/synthetic/phat_driven_mixed \
  -path '*/all' -type d 2>/dev/null | grep .; then
  echo "LEGACY all/ LAYER FOUND"
  exit 1
fi

if find data/synthetic/phat_driven_top2 data/synthetic/phat_driven_mixed \
  -type f \( -name 'train_*.json' -o -name 'test_*.json' -o -name 'train__*.json' -o -name 'test__*.json' \) 2>/dev/null | grep .; then
  echo "SPLIT PREFIX FOUND IN USER FILENAME"
  exit 1
fi

echo "OK - anonymization audit passed."
