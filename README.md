# aichessathon-bot

Our team's chess engine for [AI Chessathon](https://aichessathon.com), September 2026. Over four
days the platform played it against other teams' agents on a fixed cadence. The whole submission
is `agent.py` plus the modules and weights beside it, exposing one function:

```python
def get_move(fen: str, time_left_ms: int) -> str: ...
```

Every game ran at 120 s + 0.5 s per side on a single EPYC core with 2 GB of memory, no network
and no GPU. It used pure Python, with numba for speed and only the platform's preinstalled
packages.

The repo is a fork of
[advitrocks9/aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter). The
starter supplied the local harness (`harness/`) and the baselines (`baselines/`), and everything
else here was built on top of them. **The locked build is v4.4 (tag `v4.4`, branch `prod`).**

## What the engine does

**Search** (`fastsearch.py`, with `agent.py` as a python-chess fallback)

- Iterative deepening negamax with alpha-beta, principal variation search and late move reductions.
- Quiescence search on captures and queen promotions.
- Move ordering: transposition-table move, MVV-LVA captures, promotions, two killer moves, then
  history.
- A transposition table kept for the whole game, capped at 500k entries.
- Repetition and fifty-move detection, plus contempt, so it avoids draws it doesn't need and takes
  them when it's behind.

**Board and speed** (`fastboard.py`)

- A numba-compiled board with move generation and Zobrist hashing. It runs at 1.2 to 3.3 million
  nodes per second, against 50 to 70 thousand on python-chess, which is two to three extra plies.
- Every jitted function is warmed during the 90 s import budget, so compilation never costs game
  clock.

**Evaluation** (`fasteval.py`, `fastnnue.py`)

- A hand-written tapered evaluation: middlegame and endgame piece-square tables blended by
  material, passed, isolated and doubled pawns, a king shield, and mop-up for won endings.
- A learned NNUE evaluation (`weights/nnue.npz`, 768 → 256 → 32 → 1) that we trained ourselves on
  52M lichess-evaluated positions (`tools/nnue/`). It is updated incrementally, with two
  side-relative accumulators per ply.
- Each leaf scores the **average of the hand evaluation and the network**. On its own the net
  measured about -106 Elo against the hand evaluation, while the average measured +190 to +240.
  The hand side keeps the material count honest and the net adds what it learned.
- Endgame rules from reviewing rated games: king and rook pawn against a bare king is a draw when
  the defender is in front, and bare endgames keep the blend continuous so a winning capture
  doesn't read as a loss.

**Time management**

- A soft budget decides whether to start a new depth, and a hard budget aborts the search
  mid-iteration. Iterations are only started when the projected cost fits.
- One eighth of the starting clock is held in reserve, so long games level off near 17 s instead
  of sinking to 5 s.
- A backstop timer thread stops the search at the hard deadline even if a slow subtree misses the
  node-counted clock check.

**Opening book**

- `weights/book.bin` is a polyglot book with 24,479 entries, built from 1.13M over-the-board
  master games (`tools/book/build.py`) and played up to ply 20.

**Safety net**

- `get_move` wraps the search in a `try`. If anything raises, it logs the traceback and returns a
  legal move instead of forfeiting. It never fired in a rated game.

## Timeline

Full numbers for every version are in [`docs/VERSIONS.md`](docs/VERSIONS.md). Each version is a
git tag and a frozen copy under `local-opponents/vX.Y/`, which later versions were benched against.

| Date | Version | Milestone |
|---|---|---|
| Aug 29 | – | Upstream starter: harness, baselines, a random mover. We forked from here. |
| Sep 8 | v1.0 | Two-ply negamax on material and piece-square tables. The reference everything had to beat. |
| Sep 8 | v2.0 | Iterative deepening, alpha-beta, quiescence, MVV-LVA, soft and hard time budgets. 92% vs minimax. |
| Sep 8 | v2.1 | Transposition table, killers, history, draw detection with contempt. 84% vs v2.0. |
| Sep 8 | v2.2 | Shipped the numba board unused, to prove the platform accepts it. **First rated games, rounds 73 and 74: won one, lost one.** |
| Sep 8 | v2.3 | The iteration gate stops refusing depths it could afford. Round 74's blunders were all that shape. |
| Sep 8 | v2.4 | Tapered hand evaluation with pawn structure and king safety. **82.8% vs v2.3, about +273 Elo.** |
| Sep 8 | v3.0 | Search and evaluation moved onto the numba board, 20 to 60 times faster. **93.8% vs v2.4.** |
| Sep 8 | v3.1 | Backstop timer thread at the hard deadline. |
| Sep 9 | v3.2 | Polyglot opening book. |
| Sep 9 | v4.0 | Our NNUE switched on, averaged with the hand evaluation. **80.2% vs v3.2 over 96 games, about +243 Elo.** |
| Sep 9 | v4.1 | Fixes from rated-game reviews: rook-pawn draws, and a queen promotion instead of an under-promotion on ties. |
| Sep 10 | v4.2 | Principal variation search, late move reductions, a cap on the iteration gate. **69.5% vs v4.1 over 200 games, about +143 Elo.** |
| Sep 10 | v4.3 | Clock reserve, so long games no longer run the clock down to 5 s. |
| Sep 10 | v4.4 | Continuous evaluation across the bare-endgame line, after a mate stalled for 47 moves in round 102. **The locked build.** |
| Sep 11 | – | Uploads closed at 11:00 UK. A v4.5 with a net fine-tuned on our own games pooled 53.0% over 115 games, which isn't proven, and two uploads failed in the platform's image build anyway. v4.4 played the final qualification. |

**Rated play.** The logbook reviews 31 rated games, rounds 73 to 104 except round 89. Across them
the engine won 14, drew 7 and lost 10, playing the v2.2 through v4.2 builds.

**Where it stands.** Against Stockfish 19 at fixed depth 8, v4.4 scored 46.9% over 16 games at
20 s + 0.2 s, about -22 Elo. At 10 s + 0.1 s it dropped to 31%, and almost all of those losses
came after its clock fell to around 1.5 s. See `docs/BENCH_LOG.md`.

### What we tried and threw away

We benched every idea against the previous frozen version and merged it only if the numbers held.
The rejections are in `docs/VERSIONS.md` ("Not shipped") and `docs/LOGBOOK.md`:

- **Mobility evaluation:** 47.3% over 128 games. The net already encodes mobility.
- **A bigger training set for the same net (163M positions):** 44.8% over 144 games, despite
  better offline loss.
- **Four king buckets for the NNUE:** built and proved exact, but never fine-tuned in time.
- **Checks in quiescence:** a third of the speed for 40.6%.
- **A 2x soft-budget ceiling:** more depth, but 41.2% at the proxy control because it spent the
  endgame clock.
- **Null move, futility pruning and several time-manager variants:** no effect at our sample sizes,
  or unsafe on the clock.

## How we worked

- **One variable per PR**, benched in `harness/arena.py` against a frozen copy of the previous
  version, at two time controls. Every result is reported with its 95% interval and broken down by
  termination. A flag, illegal move or crash is a blocker whatever the score.
- **Every rated game reviewed.** Most of v4.1, v4.3 and v4.4 came from specific moves in specific
  rounds, written up in `docs/LOGBOOK.md`.
- An orchestrator session wrote briefs with numeric success criteria, and implementation ran in
  separate worktrees, one per branch. `docs/TEAM.md` has the conventions and the status log, and
  `docs/STRATEGY.md` has the research plan.

## Repo map

```
agent.py             the submission: get_move, time manager, book, python-chess fallback search
fastboard.py         numba board, move generation, Zobrist hashing
fastsearch.py        compiled search: PVS, LMR, quiescence, TT, killers, history
fasteval.py          compiled hand evaluation
fastnnue.py          compiled NNUE inference with incremental accumulators
weights/             nnue.npz (our trained net), book.bin (polyglot opening book)
tools/nnue/          data pipeline, training, export of the network
tools/book/          builds the opening book from master games
tools/analyse_game.py  post-game analysis of rated games
local-opponents/     every shipped version frozen, plus Sunfish and Stockfish wrappers (fetched, never shipped)
harness/             the starter's mirror of the platform protocol and clock (unchanged)
baselines/           the starter's random, greedy, minimax and numba baselines
docs/                VERSIONS, LOGBOOK, BENCH_LOG, STRATEGY, TEAM, NNUE_KING_BUCKETS
```

## Running it

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/AndrewL1698/aichessathon-bot
cd aichessathon-bot
git checkout prod
make setup
make play      # one game against a baseline at 120 s + 0.5 s
make arena     # 16 fast games with a score and interval
make zip       # build submission.zip and smoke it the way the platform does
make gate      # ruff, mypy strict, and two games that must finish cleanly
```

To play against an earlier version or the frozen opponents:

```
uv run python -m harness.arena --opponent local-opponents/v4.2 --games 32
local-opponents/fetch_sunfish.sh     # Sunfish is GPL, so it's fetched rather than committed
```

The competition rules live at [aichessathon.com/docs](https://aichessathon.com/docs).
