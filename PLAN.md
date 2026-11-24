# obi-sweep: Order-Book-Imbalance Parameter Sweep

A small research platform that tests a top-of-book imbalance signal across thousands of parameter sets, with a Rust fill simulator, benchmarked on standard vs. free-threaded Python.

Origin story: an L3-book gradient idea demoed to the quants at First Boston during NASDAQ market-making days, rebuilt with 2026 tooling.

Owner: Daniel (GitHub: McSavage)
Hardware: Optiplex (Dell OptiPlex 7050, Linux Mint, integrated graphics only) as compute node; Windows 11 as client (VS Code over SSH, IB Gateway).

---

## 1. Goal

Produce a public repo whose README opens with one benchmark table:

| Fill simulator | Standard Python, threads | Free-threaded Python, threads | Processes (baseline) |
|---|---|---|---|
| Pure Python loop | runs/sec | runs/sec | runs/sec |
| Rust kernel (PyO3) | runs/sec | runs/sec | runs/sec |

Plus:
- A thread-scaling curve (1, 2, 4, 8 workers). The 7050 has 4 cores (8 threads if i7), so the curve will flatten there. Say so.
- py-spy flamegraphs before and after the Rust port.
- A results chart showing at what spread/cost the edge disappears.

What it demonstrates, mapped to the req:

| Req bullet | Demo piece |
|---|---|
| Thousands of simulations in parallel | The parameter sweep |
| Arrow, Polars, DuckDB, tick-level history | Partitioned Parquet, lazy scans, `join_asof` |
| Rust kernels via PyO3 | `fill_sim` crate |
| Free-threaded concurrency | The benchmark table |
| uv/ruff hygiene, benchmarking over assuming | uv, ruff, pytest, py-spy, CI |
| Research as a systems problem | Point-in-time joins, runtime vs. iteration speed |

Out of scope: CUDA, JAX, Triton, Kafka, Ray, Kubernetes, kdb+. Optional later: one JAX row on rented GPU time.

---

## 2. Evening 0: environment and the risk check (today)

The main risk is that a dependency re-enables the GIL on the free-threaded build. Find out first.

### 2.1 System packages

```bash
sudo apt update
sudo apt install -y build-essential pkg-config git curl
```

### 2.2 uv and Rust

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.bashrc
uv --version && rustc --version && cargo --version
```

### 2.3 Interpreters

```bash
uv python install 3.14 3.14t 3.15 3.15t
uv python list --only-installed
```

### 2.4 Project skeleton

```bash
mkdir -p ~/dev/obi-sweep && cd ~/dev/obi-sweep
git init
uv init --python 3.14t
uv add polars pyarrow duckdb numpy
uv add --dev ruff pytest py-spy maturin
```

### 2.5 The GIL check

Save as `scripts/gil_check.py`:

```python
import sys, sysconfig

print("version:", sys.version)
print("free-threaded build:", bool(sysconfig.get_config_var("Py_GIL_DISABLED")))
print("GIL enabled at start:", sys._is_gil_enabled())

import numpy, pyarrow, polars, duckdb  # noqa: E401,F401

print("GIL enabled after imports:", sys._is_gil_enabled())
```

Run against each interpreter:

```bash
for v in 3.14t 3.15t; do
  echo "== $v =="
  uv run --python $v --with polars --with pyarrow --with duckdb --with numpy \
    python scripts/gil_check.py
done
```

Decision:
- "GIL enabled after imports: False" on 3.14t or 3.15t: use that one. Prefer whichever has every wheel.
- If an import flips it to True: note which library. Try `PYTHON_GIL=0` to force it off and see if things still work. Record the result either way. It is a legitimate README finding.
- If a wheel will not install at all on the `t` build: drop that library from the hot path (DuckDB is the most optional).

Record the outcome in `NOTES.md` before moving on.

---

## 3. Data

### 3.1 Primary: LOBSTER sample files (free)

Source: lobsterdata.com, "Data Samples" page. Free one-day NASDAQ samples (e.g. AMZN, AAPL, GOOG, INTC, MSFT, June 21, 2012) at book depths 1, 5, 10.

Each sample has two CSVs with the same row count, one row per book event:

**message file** (`*_message_10.csv`), no header:

| col | name | notes |
|---|---|---|
| 0 | time | seconds after midnight, decimal (nanosecond precision) |
| 1 | type | 1 submit, 2 partial cancel, 3 delete, 4 visible execution, 5 hidden execution, 7 trading halt |
| 2 | order_id | |
| 3 | size | shares |
| 4 | price | dollars x 10000 |
| 5 | direction | 1 buy order, -1 sell order |

**orderbook file** (`*_orderbook_10.csv`), no header, repeating per level:
`ask_px_1, ask_sz_1, bid_px_1, bid_sz_1, ask_px_2, ...`

Verify the column layout against the README that ships in the zip before trusting this table.

Layout on disk:

```
data/raw/lobster/            # untouched zips and CSVs
data/parquet/book/symbol=AMZN/date=2012-06-21/part-0.parquet
```

### 3.2 Secondary (later phases)

| Source | Use | Cost |
|---|---|---|
| Databento | More book data (futures, equities) | Free signup credits, check current terms |
| Interactive Brokers API | Live tick collector (Phase 4). Not for bulk history: request caps and pacing limits. Depth needs a market-data subscription. | Existing account |
| FRED / ALFRED | Macro release dates and first-print vintages for the point-in-time join | Free, API key required |

---

## 4. Repo layout

```
obi-sweep/
  pyproject.toml
  README.md
  NOTES.md                 # findings log, one dated line per finding
  PLAN.md                  # this file
  scripts/
    gil_check.py
    ingest_lobster.py      # CSV -> partitioned Parquet
    run_sweep.py           # benchmark harness
  src/obi_sweep/
    __init__.py
    features.py            # Polars: imbalance, mid, spread
    sim_py.py              # pure-Python fill simulator
    sweep.py               # param grid, executors, timing
  rust/fill_sim/
    Cargo.toml
    pyproject.toml         # maturin
    src/lib.rs
  tests/
    test_sim_parity.py     # Python and Rust sims must agree
  results/                 # parquet + charts + flamegraphs (gitignored except summary)
```

---

## 5. Phase 1 (weekend 1): ingest, features, pure-Python sim, first row

### 5.1 Ingest

`scripts/ingest_lobster.py`: read message and orderbook CSVs with `pl.read_csv(has_header=False)`, name columns, horizontally concat, convert price to float dollars, add `symbol` and `date`, write partitioned Parquet with `write_parquet(..., partition_by=[...])` or pyarrow datasets.

### 5.2 Features (`features.py`)

Level-1 imbalance, mid, spread, computed lazily:

```python
import polars as pl

def level1_features(path: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .with_columns(
            mid=(pl.col("ask_px_1") + pl.col("bid_px_1")) / 2,
            spread=pl.col("ask_px_1") - pl.col("bid_px_1"),
            imb=(pl.col("bid_sz_1") - pl.col("ask_sz_1"))
                / (pl.col("bid_sz_1") + pl.col("ask_sz_1")),
        )
        .select("time", "bid_px_1", "ask_px_1", "mid", "spread", "imb")
        .collect()
    )
```

Stretch: depth-weighted imbalance over levels 1 to 5 (closer to the original "gradient" idea).

### 5.3 Pure-Python simulator (`sim_py.py`)

Deliberately branchy and sequential: the workload that belongs in Rust.

Rules:
- Flat: if `imb > thr` buy at ask; if `imb < -thr` sell at bid.
- Exit after `hold` events: long exits at bid, short exits at ask.
- Subtract `fee` per share per side.
- One position at a time, one share.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    thr: float
    hold: int
    fee: float

@dataclass(frozen=True)
class Result:
    params: Params
    pnl: float
    trades: int
    wins: int

def simulate(imb, bid, ask, p: Params) -> Result:
    n = len(imb)
    pos = 0
    entry = 0.0
    exit_at = -1
    pnl = 0.0
    trades = wins = 0
    for i in range(n):
        if pos != 0 and i >= exit_at:
            px = bid[i] if pos > 0 else ask[i]
            gain = (px - entry) * pos - 2 * p.fee
            pnl += gain
            trades += 1
            wins += gain > 0
            pos = 0
        if pos == 0:
            if imb[i] > p.thr:
                pos, entry, exit_at = 1, ask[i], i + p.hold
            elif imb[i] < -p.thr:
                pos, entry, exit_at = -1, bid[i], i + p.hold
    return Result(p, pnl, trades, wins)
```

Pass plain Python lists (from `series.to_list()`) for the pure-Python version so it measures interpreter speed, not NumPy element access.

### 5.4 Sweep and harness (`sweep.py`, `run_sweep.py`)

Grid, for example: `thr` in 20 values from 0.1 to 0.9, `hold` in 10 values from 5 to 500 events, `fee` in 5 values from 0 to 0.005. That is 1,000 runs. Scale up once timing is known.

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import product

def grid():
    thrs = [0.1 + 0.04 * i for i in range(20)]
    holds = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500]
    fees = [0.0, 0.001, 0.002, 0.003, 0.005]
    return [Params(t, h, f) for t, h, f in product(thrs, holds, fees)]

def run(executor_cls, workers, fn, params):
    t0 = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        results = list(ex.map(fn, params))
    return results, time.perf_counter() - t0
```

`fn` closes over the shared arrays for threads. For processes, load the arrays in a pool initializer (each process gets its own copy, which is the point of the comparison).

Run the matrix:

```bash
for v in 3.14 3.14t; do
  for w in 1 2 4 8; do
    uv run --python $v python scripts/run_sweep.py --sim py --executor thread --workers $w
  done
  uv run --python $v python scripts/run_sweep.py --sim py --executor process --workers 4
done
```

Write each run's timing to `results/bench.parquet` with columns: `interpreter, gil_enabled, sim, executor, workers, n_runs, seconds, runs_per_sec`. Always record `sys._is_gil_enabled()` at runtime, not the interpreter name.

Expected (to be confirmed, not assumed): standard Python threads show little or no scaling on the pure-Python sim; free-threaded threads scale toward the core count.

Profile one run:

```bash
uv run --python 3.14t py-spy record -o results/flame_py.svg -- \
  python scripts/run_sweep.py --sim py --executor thread --workers 1 --n 50
```

---

## 6. Phase 2 (weekend 2): Rust kernel, second row

### 6.1 Crate

```bash
cd rust && cargo new --lib fill_sim && cd fill_sim
```

`Cargo.toml` essentials:

```toml
[lib]
name = "fill_sim"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "*", features = ["extension-module"] }
numpy = "*"
```

Pin `pyo3` and `numpy` to matching current versions (the numpy crate tracks PyO3 releases). Check crates.io the day you build.

`src/lib.rs` sketch:

```rust
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn simulate(
    py: Python<'_>,
    imb: PyReadonlyArray1<f64>,
    bid: PyReadonlyArray1<f64>,
    ask: PyReadonlyArray1<f64>,
    thr: f64,
    hold: usize,
    fee: f64,
) -> PyResult<(f64, u64, u64)> {
    let (imb, bid, ask) = (imb.as_slice()?, bid.as_slice()?, ask.as_slice()?);
    // Release the interpreter for the hot loop.
    // Method is `allow_threads` in older PyO3, `detach` in newer releases.
    let out = py.allow_threads(|| {
        let (mut pos, mut entry, mut exit_at) = (0i8, 0.0f64, 0usize);
        let (mut pnl, mut trades, mut wins) = (0.0f64, 0u64, 0u64);
        for i in 0..imb.len() {
            if pos != 0 && i >= exit_at {
                let px = if pos > 0 { bid[i] } else { ask[i] };
                let gain = (px - entry) * pos as f64 - 2.0 * fee;
                pnl += gain;
                trades += 1;
                if gain > 0.0 { wins += 1; }
                pos = 0;
            }
            if pos == 0 {
                if imb[i] > thr { pos = 1; entry = ask[i]; exit_at = i + hold; }
                else if imb[i] < -thr { pos = -1; entry = bid[i]; exit_at = i + hold; }
            }
        }
        (pnl, trades, wins)
    });
    Ok(out)
}

#[pymodule(gil_used = false)]   // declares the module safe without the GIL
fn fill_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
```

Why `gil_used = false` matters: without it, importing the module on a free-threaded build silently re-enables the GIL. Run `gil_check.py` again after importing `fill_sim` to prove it stays off.

### 6.2 Build and install into the project venv

```bash
cd ~/dev/obi-sweep
uv run --python 3.14t maturin develop --release -m rust/fill_sim/Cargo.toml
```

Repeat for the standard interpreter when benchmarking that column.

### 6.3 Parity test (do this before benchmarking)

`tests/test_sim_parity.py`: for 50 random `Params`, assert Python and Rust return the same `trades` and `wins`, and `pnl` within 1e-9. A fast wrong answer is worthless.

### 6.4 Second benchmark row and flamegraph

Same matrix as 5.4 with `--sim rust`. Then `py-spy record` again, saved as `results/flame_rust.svg`. The before/after pair goes in the README.

Note for the writeup: the Rust kernel releases the interpreter, so it should already scale on standard Python threads. The interesting comparison is how much free-threading adds on top, which depends on how much Python glue remains per run.

---

## 7. Phase 3: README and results

README order:
1. One paragraph: what it is, the First Boston origin, what was measured.
2. Benchmark table (filled in).
3. Scaling curve chart.
4. Flamegraphs, before and after.
5. Results chart: PnL by threshold and fee, showing where the edge dies against the spread.
6. Findings: GIL behavior per library, surprises, what did not scale and why.
7. Limits: one day of 2012 data proves the plumbing, not the alpha; 4-core box caps scaling; no queue-position modeling (fills assume you get the touch).

---

## 8. Phase 4 (optional)

- **IB live collector** on Windows or Linux: IB Gateway plus `ib_async` (community successor to `ib_insync`), tick-by-tick bid/ask to Parquet in the same partition layout. Check current pacing rules in IB docs.
- **Point-in-time join**: pull CPI release timestamps and first-print values from ALFRED, `join_asof` onto the tick stream, look at book depth and spread in the seconds around the print. Show first-print vs. revised values to illustrate lookahead bias.
- **CI**: GitHub Actions with maturin to build wheels for Linux and Windows, including a free-threaded build target.
- **GPU row**: vectorized sweep in JAX on CPU, then the same code on a few dollars of rented GPU time.

---

## 9. Talking points this enables

- "Backtest speed pays off in steps, not linearly. The goal is getting a run under the attention threshold so the researcher never leaves the loop."
- "I benchmark first. The fill sim is branchy and sequential, so it went to Rust on CPU. The sweep is the vectorizable part, so that is the GPU candidate."
- "On free-threaded Python, one undeclared extension silently brings the GIL back. I check `sys._is_gil_enabled()` at runtime and record it with every benchmark."
- "As-of joins I know from production on Quartz and Sandra; here it is `join_asof` in Polars. Same concept as kdb+ `aj`."

---

## 10. Today's checklist

- [ ] apt packages, uv, rustup installed
- [ ] 3.14, 3.14t, 3.15, 3.15t interpreters installed
- [ ] Project skeleton, git init, first commit
- [ ] `gil_check.py` run on both `t` builds, result written to NOTES.md
- [ ] LOBSTER sample zip downloaded (start with one symbol, level 10)
- [ ] Column layout verified against the LOBSTER README
- [ ] Public GitHub repo created under McSavage
