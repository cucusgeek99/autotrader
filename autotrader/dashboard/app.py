"""Lightweight web dashboard: equity curve, knowledge leaderboard, paper fills.

Serves a single self-contained HTML page (Chart.js from CDN) plus JSON endpoints.
Run via `autotrader dashboard`. Optional dep: pip install 'autotrader[web]'.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from autotrader.config import Config
from autotrader.knowledge.store import KnowledgeStore
from autotrader.live.paper_broker import PaperBroker


def create_app(config_path: str = "config.yaml"):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install web deps: pip install 'autotrader[web]'") from e

    cfg = Config.load(config_path)
    app = FastAPI(title="autotrader dashboard")

    @app.get("/api/leaderboard")
    def leaderboard():
        store = KnowledgeStore(cfg["knowledge_db"])
        rows = store.leaderboard(50)
        store.close()
        return JSONResponse(rows)

    @app.get("/api/portfolio")
    def portfolio():
        state_path = ".cache/paper_state.json"
        if not Path(state_path).exists():
            return JSONResponse({"exists": False})
        data = json.loads(Path(state_path).read_text())
        # build a cumulative equity curve from fills (mark-to-fill-price)
        cash = float(cfg["paper"]["starting_cash"])
        position = 0.0
        curve = []
        for f in data.get("fills", []):
            cash -= f["units"] * f["price"] + f["fee"]
            position += f["units"]
            equity = cash + position * f["price"]
            curve.append({"time": f["time"], "equity": round(equity, 2), "price": f["price"]})
        return JSONResponse({
            "exists": True,
            "symbol": data.get("symbol"),
            "cash": round(data.get("cash", 0), 2),
            "position": data.get("position", 0),
            "num_fills": len(data.get("fills", [])),
            "curve": curve,
            "fills": data.get("fills", [])[-25:],
        })

    @app.get("/api/book")
    def book():
        state_path = ".cache/xs_paper_state.json"
        if not Path(state_path).exists():
            return JSONResponse({"exists": False})
        data = json.loads(Path(state_path).read_text())
        hist = data.get("equity_history", [])
        positions = []
        # reconstruct current weights from last-known prices in fills
        last_price = {}
        for f in data.get("fills", []):
            last_price[f["symbol"]] = f["price"]
        eq = hist[-1]["equity"] if hist else float(data.get("cash", 0))
        for sym, units in data.get("positions", {}).items():
            if sym in last_price and abs(units) > 1e-12:
                val = units * last_price[sym]
                positions.append({"symbol": sym, "weight": round(val / (eq or 1), 4),
                                  "value": round(val, 2)})
        positions.sort(key=lambda r: r["weight"], reverse=True)
        return JSONResponse({
            "exists": True,
            "equity": round(eq, 2),
            "cash": round(data.get("cash", 0), 2),
            "n_positions": len(positions),
            "curve": [{"time": h["time"], "equity": h["equity"]} for h in hist[-300:]],
            "positions": positions,
        })

    @app.get("/api/live")
    def live(testnet: bool = True):
        """Live broker account: balance, open positions w/ PnL, net/gross exposure."""
        key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_SECRET", "") or os.getenv("BINANCE_API_SECRET", "")
        if not key or not secret:
            return JSONResponse({"exists": False, "reason": "no BINANCE_API_KEY/SECRET in env"})
        try:
            from autotrader.live.binance_futures import FuturesClient
            c = FuturesClient(key, secret, testnet=testnet)
            bal = c.balance_usdt()
            raw = c._signed("GET", "/fapi/v2/positionRisk", {})
        except Exception as e:  # network / auth / venue
            return JSONResponse({"exists": False, "reason": str(e)[:140]})
        positions, gross, net, upnl = [], 0.0, 0.0, 0.0
        for p in raw:
            amt = float(p["positionAmt"])
            if abs(amt) < 1e-12:
                continue
            mark = float(p.get("markPrice") or 0)
            pnl = float(p.get("unRealizedProfit") or 0)
            notional = amt * mark            # signed
            gross += abs(notional); net += notional; upnl += pnl
            positions.append({
                "symbol": p["symbol"],
                "side": "LONG" if amt > 0 else "SHORT",
                "amt": amt,
                "entry": float(p.get("entryPrice") or 0),
                "mark": mark,
                "notional": round(abs(notional), 2),
                "upnl": round(pnl, 3),
            })
        positions.sort(key=lambda r: -r["notional"])
        return JSONResponse({
            "exists": True, "testnet": testnet,
            "balance": round(bal, 2),
            "equity": round(bal + upnl, 2),
            "upnl": round(upnl, 3),
            "gross": round(gross, 2),
            "net": round(net, 2),
            "n_positions": len(positions),
            "positions": positions,
        })

    @app.get("/api/ledger")
    def ledger_api(testnet: bool = True, interval: int = 3600):
        """Live track record: per-tick equity curve + pre-committed-rule verdict."""
        from autotrader.live.ledger import Ledger
        path = f"{cfg['data_cache_dir']}/ledger_{'testnet' if testnet else 'mainnet'}.jsonl"
        led = Ledger(path, interval_s=float(interval))
        recs = led.records()
        return JSONResponse({
            "summary": led.summary(),
            "curve": [{"time": r["ts"], "equity": r["equity"]} for r in recs[-300:]],
        })

    @app.get("/api/core")
    def core_api(mode: str = "paper"):
        """Core book (vol-target × SMA200) vs buy&hold benchmarks, from its daily ledger."""
        path = Path(f"{cfg['data_cache_dir']}/core_{mode}.jsonl")
        if not path.exists():
            return JSONResponse({"exists": False})
        recs = []
        for line in path.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not recs:
            return JSONResponse({"exists": False})
        last, s0 = recs[-1], recs[0]["bench"]["hold_5050"]   # same starting capital for all three
        return JSONResponse({
            "exists": True, "mode": mode, "start": recs[0]["ts"][:10], "last": last["ts"][:16],
            "equity": last["equity"], "ret": last["equity"] / s0 - 1,
            "bench": {k: v / recs[0]["bench"][k] - 1 for k, v in last["bench"].items()},
            "weights": last["weights"], "halted": last.get("halted", False),
            "curve": [{"time": r["ts"][:10], "core": r["equity"],
                       "hold_5050": r["bench"]["hold_5050"], "hold_btc": r["bench"]["hold_btc"]}
                      for r in recs],
        })

    @app.get("/api/health")
    def health():
        """launchd job status: registered?, last exit code, last run time."""
        out = {"registered": False, "last_exit": None, "last_run": None}
        try:
            res = subprocess.run(["launchctl", "list"], capture_output=True,
                                 text=True, timeout=5)
            for line in res.stdout.splitlines():
                if "com.autotrader.rebalance" in line:
                    parts = line.split()
                    out["registered"] = True
                    out["last_exit"] = parts[1] if len(parts) > 1 else None
                    break
        except Exception:
            pass
        log = Path(f"{cfg['data_cache_dir']}/launchd_rebalance.log")
        if log.exists():
            import datetime
            out["last_run"] = datetime.datetime.fromtimestamp(
                log.stat().st_mtime).isoformat(timespec="seconds")
        return JSONResponse(out)

    @app.get("/")
    def index():
        return HTMLResponse(_INDEX_HTML)

    return app


_INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>autotrader</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg:#0d1117; --card:#161b22; --line:#30363d; --fg:#e6edf3; --mut:#8b949e;
          --grn:#3fb950; --red:#f85149; --acc:#58a6ff; }
  * { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
    align-items:center; gap:12px; } header h1 { font-size:18px; margin:0; }
  .badge { color:var(--mut); font-size:12px; }
  main { padding:24px; display:grid; gap:24px; grid-template-columns:1fr 1fr; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:18px; } .card.wide { grid-column:1/3; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.5px; color:var(--mut);
    margin:0 0 14px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--mut); font-weight:600; }
  .pos { color:var(--grn); } .neg { color:var(--red); }
  .stat { display:inline-block; margin-right:28px; }
  .stat b { display:block; font-size:22px; } .stat span { color:var(--mut); font-size:11px; }
  .buy { color:var(--grn); } .sell { color:var(--red); }
  .card.live { border-color:#1f6feb; box-shadow:0 0 0 1px rgba(31,111,235,.25); }
  .card.live h2 { color:#58a6ff; }
  .v { padding:8px 14px; border-radius:8px; font-weight:600; display:inline-block; }
  .v.green { background:rgba(63,185,80,.15); color:var(--grn); border:1px solid var(--grn); }
  .v.yellow{ background:rgba(210,153,34,.15); color:#d29922; border:1px solid #d29922; }
  .v.red   { background:rgba(248,81,73,.15); color:var(--red); border:1px solid var(--red); }
  .dot { color:var(--grn); } .dot.stale { color:#d29922; } .dot.dead { color:var(--red); }
</style></head><body>
<header><h1>autotrader</h1><span class="badge" id="sym">paper trading · self-tuning lab</span>
  <span class="badge" id="health" style="margin-left:auto"></span></header>
<main>
  <div class="card wide live" id="corecard"><h2>● CORE — vol-target × SMA200 · spot BTC/ETH vs just holding</h2>
    <div id="corestats"></div>
    <canvas id="corecurve" height="70"></canvas></div>
  <div class="card wide live"><h2>● LIVE — testnet futures (findings_xs)</h2>
    <div id="livestats"></div>
    <div id="verdict" style="margin:12px 0"></div>
    <canvas id="livecurve" height="70"></canvas>
    <div id="livepos" style="margin-top:14px"></div></div>
  <div class="card wide"><h2>Paper portfolio equity</h2>
    <div id="stats"></div><canvas id="equity" height="90"></canvas></div>
  <div class="card"><h2>Knowledge leaderboard (by OOS score)</h2>
    <div id="lb"></div></div>
  <div class="card"><h2>Recent fills</h2><div id="fills"></div></div>
  <div class="card wide"><h2>findings_xs — market-neutral book</h2>
    <div id="bookstats"></div><canvas id="bookeq" height="80"></canvas>
    <div id="bookpos" style="margin-top:14px"></div></div>
</main>
<script>
const f = (n,d=2)=>Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
function vclass(v){ if(!v) return 'yellow';
  if(v.includes('SURVIVED')) return 'green';
  if(v.includes('DECAYED')) return 'red'; return 'yellow'; }
async function loadLive(){
  // launchd health
  try{
    const h = await (await fetch('/api/health')).json();
    const ok = h.registered && (h.last_exit==='0'||h.last_exit===0||h.last_exit===null);
    const cls = !h.registered?'dead':(ok?'':'stale');
    const when = h.last_run? new Date(h.last_run).toLocaleString():'never';
    document.getElementById('health').innerHTML =
      `<span class="dot ${cls}">●</span> launchd ${h.registered?'on':'OFF'} `+
      `· last run ${when} · exit ${h.last_exit??'-'}`;
  }catch(e){}
  // live account
  const L = await (await fetch('/api/live')).json();
  const ls = document.getElementById('livestats');
  if(!L.exists){
    ls.innerHTML = `<p style="color:#8b949e">Live broker unavailable: ${L.reason||'n/a'}. `+
      `Set BINANCE_API_KEY/SECRET and run the launchd job.</p>`;
  } else {
    const pc = L.upnl>=0?'pos':'neg';
    ls.innerHTML =
      `<div class="stat"><b>$${f(L.balance)}</b><span>BALANCE (USDT)</span></div>`+
      `<div class="stat"><b class="${pc}">${L.upnl>=0?'+':''}$${f(L.upnl,3)}</b><span>UNREALIZED PNL</span></div>`+
      `<div class="stat"><b>$${f(L.gross)}</b><span>GROSS NOTIONAL</span></div>`+
      `<div class="stat"><b class="${Math.abs(L.net)<L.gross*0.1?'pos':'neg'}">$${f(L.net)}</b><span>NET (≈0 = neutral)</span></div>`+
      `<div class="stat"><b>${L.n_positions}</b><span>POSITIONS</span></div>`;
    document.getElementById('livepos').innerHTML = table(L.positions,
      ['symbol','side','notional','entry','mark','uPnL'], r=>`<tr>
        <td>${r.symbol}</td><td class="${r.side==='LONG'?'buy':'sell'}">${r.side}</td>
        <td>$${f(r.notional)}</td><td>$${f(r.entry,4)}</td><td>$${f(r.mark,4)}</td>
        <td class="${r.upnl>=0?'pos':'neg'}">${r.upnl>=0?'+':''}${f(r.upnl,3)}</td></tr>`);
  }
  // ledger track record + verdict
  const lg = await (await fetch('/api/ledger')).json();
  const s = lg.summary||{};
  const sh = s.live_sharpe;
  document.getElementById('verdict').innerHTML =
    `<span class="v ${vclass(s.verdict)}">${s.verdict||'no data'}</span>`+
    `<span class="stat" style="margin-left:24px"><b>${s.n_rebalances??0}/${s.min_required??30}</b><span>SAMPLES</span></span>`+
    `<span class="stat"><b class="${(sh??0)>=(0.7)?'pos':'neg'}">${sh==null?'—':f(sh,2)}</b><span>LIVE SHARPE (bt ${s.backtest_sharpe??'?'})</span></span>`+
    `<span class="stat"><b>${s.hit_rate==null?'—':(s.hit_rate*100).toFixed(0)+'%'}</b><span>HIT RATE</span></span>`+
    `<span class="stat"><b>${s.total_return_pct==null?'—':(s.total_return_pct>=0?'+':'')+f(s.total_return_pct,2)+'%'}</b><span>TOTAL RETURN</span></span>`;
  if(lg.curve && lg.curve.length>1){
    if(window._lv) window._lv.destroy();
    window._lv = new Chart(document.getElementById('livecurve'),{type:'line',
      data:{labels:lg.curve.map(c=>c.time.slice(5,16)),
        datasets:[{data:lg.curve.map(c=>c.equity),borderColor:'#58a6ff',
          backgroundColor:'rgba(88,166,255,.12)',fill:true,tension:.25,pointRadius:0,borderWidth:2}]},
      options:{plugins:{legend:{display:false}},scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
        y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}}});
  }
}
async function loadCore(){
  const c = await (await fetch('/api/core')).json();
  if(!c.exists){ document.getElementById('corestats').innerHTML='<span class="badge">no run yet — scripts/core_live.py</span>'; return; }
  const pct = x=>`<b class="${x>=0?'pos':'neg'}">${x>=0?'+':''}${(x*100).toFixed(1)}%</b>`;
  const w = Object.entries(c.weights).map(([k,v])=>`${k.split('/')[0]} ${(v*100).toFixed(0)}%`).join(' · ');
  document.getElementById('corestats').innerHTML =
    (c.halted?'<span class="v red">KILL-SWITCH HALTED</span> ':'')+
    `<span class="stat"><b>$${f(c.equity)}</b><span>EQUITY (${c.mode}, since ${c.start})</span></span>`+
    `<span class="stat">${pct(c.ret)}<span>CORE</span></span>`+
    `<span class="stat">${pct(c.bench.hold_5050)}<span>HOLD 50/50</span></span>`+
    `<span class="stat">${pct(c.bench.hold_btc)}<span>HOLD BTC</span></span>`+
    `<span class="stat"><b>${w}</b><span>TARGET (rest = USDT)</span></span>`;
  if(window._cc) window._cc.destroy();
  const ds=(k,col,lab)=>({label:lab,data:c.curve.map(r=>r[k]),borderColor:col,pointRadius:0,borderWidth:2,tension:.2});
  window._cc = new Chart(document.getElementById('corecurve'),{type:'line',
    data:{labels:c.curve.map(r=>r.time),datasets:[ds('core','#58a6ff','core'),
      ds('hold_5050','#8b949e','hold 50/50'),ds('hold_btc','#d29922','hold BTC')]},
    options:{plugins:{legend:{labels:{color:'#8b949e'}}},scales:{
      x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
      y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}}});
}
async function load(){
  await loadCore().catch(e=>console.error('core',e));
  await loadLive().catch(e=>console.error('live',e));
  const p = await (await fetch('/api/portfolio')).json();
  const lb = await (await fetch('/api/leaderboard')).json();
  if(p.exists){
    document.getElementById('sym').textContent = p.symbol + ' · paper trading';
    const start = p.curve.length ? p.curve[0].equity : 0;
    const last = p.curve.length ? p.curve[p.curve.length-1].equity : 0;
    const pnl = last - (start||last);
    document.getElementById('stats').innerHTML =
      `<div class="stat"><b>$${f(last)}</b><span>EQUITY</span></div>`+
      `<div class="stat"><b class="${pnl>=0?'pos':'neg'}">${pnl>=0?'+':''}$${f(pnl)}</b><span>PNL (fills)</span></div>`+
      `<div class="stat"><b>$${f(p.cash)}</b><span>CASH</span></div>`+
      `<div class="stat"><b>${f(p.position,6)}</b><span>POSITION</span></div>`+
      `<div class="stat"><b>${p.num_fills}</b><span>FILLS</span></div>`;
    new Chart(document.getElementById('equity'),{type:'line',
      data:{labels:p.curve.map(c=>c.time.slice(5,16)),
        datasets:[{data:p.curve.map(c=>c.equity),borderColor:'#58a6ff',
          backgroundColor:'rgba(88,166,255,.1)',fill:true,tension:.25,pointRadius:0,borderWidth:2}]},
      options:{plugins:{legend:{display:false}},scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
        y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}}});
    document.getElementById('fills').innerHTML = table(p.fills.slice().reverse(),
      ['time','side','units','price'], r=>`<tr>
        <td>${r.time.slice(5,16)}</td><td class="${r.side}">${r.side.toUpperCase()}</td>
        <td>${f(Math.abs(r.units),6)}</td><td>$${f(r.price)}</td></tr>`);
  } else {
    document.getElementById('stats').innerHTML='<p style="color:#8b949e">No paper trades yet. Run <code>autotrader paper BTC/USDT</code>.</p>';
  }
  document.getElementById('lb').innerHTML = table(lb.slice(0,15),
    ['symbol','regime','strategy','OOS','gap'], r=>`<tr>
      <td>${r.symbol}</td><td>${r.regime}</td><td>${r.strategy}</td>
      <td class="${r.oos_score>=0?'pos':'neg'}">${r.oos_score>=0?'+':''}${f(r.oos_score,3)}</td>
      <td>${f(r.overfit_gap,3)}</td></tr>`);

  const bk = await (await fetch('/api/book')).json();
  if(bk.exists){
    document.getElementById('bookstats').innerHTML =
      `<div class="stat"><b>$${f(bk.equity)}</b><span>BOOK EQUITY</span></div>`+
      `<div class="stat"><b>${bk.n_positions}</b><span>POSITIONS (long+short)</span></div>`+
      `<div class="stat"><b>$${f(bk.cash)}</b><span>CASH/COLLATERAL</span></div>`;
    if(bk.curve.length>1){
      if(window._bk) window._bk.destroy();
      window._bk = new Chart(document.getElementById('bookeq'),{type:'line',
        data:{labels:bk.curve.map(c=>c.time.slice(5,16)),
          datasets:[{data:bk.curve.map(c=>c.equity),borderColor:'#3fb950',
            backgroundColor:'rgba(63,185,80,.1)',fill:true,tension:.25,pointRadius:0,borderWidth:2}]},
        options:{plugins:{legend:{display:false}},scales:{
          x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
          y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}}});
    }
    document.getElementById('bookpos').innerHTML = table(bk.positions,
      ['symbol','side','weight'], r=>`<tr><td>${r.symbol}</td>
        <td class="${r.weight>=0?'buy':'sell'}">${r.weight>=0?'LONG':'SHORT'}</td>
        <td class="${r.weight>=0?'pos':'neg'}">${(r.weight*100).toFixed(1)}%</td></tr>`);
  } else {
    document.getElementById('bookstats').innerHTML='<p style="color:#8b949e">No book yet. Run <code>autotrader paper-xs --once</code>.</p>';
  }
}
function table(rows,headers,rowfn){
  if(!rows||!rows.length) return '<p style="color:#8b949e">none</p>';
  return `<table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead>`+
    `<tbody>${rows.map(rowfn).join('')}</tbody></table>`;
}
load(); setInterval(load, 15000);
</script></body></html>"""
