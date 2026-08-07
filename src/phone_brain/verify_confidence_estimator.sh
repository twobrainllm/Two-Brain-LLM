#!/usr/bin/env bash
# verify_confidence_estimator.sh
# Sanity-checks all three confidence strategies against the mock server.
# No phone required. Run this after any change to confidence_estimator.py.
set -e
cd "$(dirname "$0")"
pkill -f mock_phone_brain_server.py 2>/dev/null || true
sleep 1
python3 mock_phone_brain_server.py --port 8321 > /tmp/mock_phone_brain.log 2>&1 &
SERVER_PID=$!
sleep 1

echo "=== HARD prompt, self_consistency (3 samples) ==="
python3 confidence_estimator.py --base-url http://localhost:8321 \
  --prompt "A patient on warfarin is prescribed fluconazole, what is the interaction risk?" \
  --strategy self_consistency

echo
echo "=== EASY prompt, hybrid ==="
python3 confidence_estimator.py --base-url http://localhost:8321 \
  --prompt "Rewrite this politely: send me the file now" --strategy hybrid

echo
echo "=== HARD prompt, hybrid ==="
python3 confidence_estimator.py --base-url http://localhost:8321 \
  --prompt "Prove there are infinitely many primes using contradiction" --strategy hybrid

kill $SERVER_PID 2>/dev/null || true
