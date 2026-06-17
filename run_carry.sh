#!/bin/bash
# Delta-neutral funding-carry PAPER sleeve — daily run (no .env: paper, public data).
cd /Users/apple/works/autotrader || exit 1
exec .venv/bin/python scripts/carry_live.py
