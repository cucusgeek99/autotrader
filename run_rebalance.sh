#!/bin/bash
# One-shot testnet rebalance for the findings_xs book — invoked by launchd hourly.
# launchd does NOT inherit your shell env, so we source .env here ourselves.
cd /Users/apple/works/autotrader || exit 1
set -a
# shellcheck disable=SC1091
source .env
set +a
exec .venv/bin/python -m autotrader.cli live \
  --config config_quick.yaml --testnet --arm
