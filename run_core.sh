#!/bin/bash
# Daily CORE book (vol-target × SMA200, spot BTC/ETH) — PAPER by default, no keys needed.
# To trade the real account later: add --account (dry-run), then --account --arm --confirm-live.
cd /Users/apple/works/autotrader || exit 1
[ -f .env ] && source .env
exec .venv/bin/python scripts/core_live.py
