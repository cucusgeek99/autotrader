#!/bin/bash
# Daily dip-buyer PAPER sleeve — public daily candles, no keys needed (paper).
cd /Users/apple/works/autotrader || exit 1
exec .venv/bin/python scripts/dipbuyer_live.py
