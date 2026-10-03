# ⚡ MYLN-FRAME

> A lightweight AI inference framework — no GPU, no cloud, no waiting.

Named after the **myelin sheath**, the biological insulator that makes neural signals fast and efficient.  
Built for the real world: Raspberry Pi, old laptops, embedded boards, anywhere AI _should_ run.

---

## The idea in one picture

```
         INPUT
           ↓
       [ROUTER]          ← routes signal into specialist slots
     ↙   ↓   ↓   ↘
   [A]  [B]  [C]  [D]   ← swap heads for any domain
     ↕    ↕    ↕    ↕   ← Ring Attention: heads share context
       [CENTER LINE]     ← aggregates → classifies
           ↓
         OUTPUT
```

Same frame. Different heads. Any device.

---

## Three sizes

| | MYLN-SS ⚡ | MYLN-T 🪶 | MYLN-S 💪 |
|---|---|---|---|
| dim | 16 | 64 | 128 |
| RAM | ~1 MB | ~20 MB | ~100 MB |
| Latency | < 0.1 ms | < 1 ms | < 5 ms |
| Best for | microcontrollers / SBC | old PC / edge | laptop / server |
| GPU needed | ✗ | ✗ | ✗ |

---

## Cascade: fast path + precise path

When input is _obvious_, skip the heavy frame entirely.

```
INPUT
  ↓
[RELAY]  ← SS, 2 heads, ~9 µs
  ↓
confidence ≥ 80%?
  ├─ YES → output immediately          ← clear threats / idle
  └─ NO  → [FULL]  T, 4 heads, ~109µs ← ambiguous cases
```

Real numbers on a Raspberry Pi 5:

| Case | Path | Latency |
|---|---|---|
| Idle / all-clear | relay only | **9 µs** |
| Ransomware pattern | relay only | **9 µs** |
| Mixed / borderline | relay → full | 118 µs |

---

## Quick start

### Build

```bash
git clone https://github.com/Sub-to/myln-frame.git
cd myln-frame
mkdir build && cd build
cmake .. && make myln
# → build/libmyln.dylib  (macOS)
# → build/libmyln.so     (Linux)
```

### Python

```python
from bridge.python.myln import MylnCascade

cas = MylnCascade(threshold=0.80).tune_security()

label, conf, used_relay = cas.predict_with_path(
    [0.9, 0.95, 0.8, 0.99, 0.85]   # [proc, cpu, net, file, mem]
)
print(label, f"{conf:.0%}", "relay" if used_relay else "full")
# → CRITICAL 91% relay
```

### C / Any language via C API

```c
void* frame = myln_new("T", 5);
myln_tune_security(frame, 5);

float features[5] = {0.9f, 0.95f, 0.8f, 0.99f, 0.85f};
int n_out = 0;
const float* probs = myln_infer(frame, features, 5, &n_out);

myln_free(frame);
```

The C API works with Python (ctypes), Node.js (ffi-napi), Ruby (Fiddle), Go (cgo), and Rust (bindgen).

---

## Security monitoring (built-in tuner)

5-dimensional feature vector → 5-class threat level, **no training required**.

| Index | Feature | 0.0 | 1.0 |
|---|---|---|---|
| 0 | proc_anomaly | normal | highly suspicious |
| 1 | cpu_spike | idle | maxed out |
| 2 | net_bytes | quiet | heavy traffic |
| 3 | file_change | no change | mass rewrite |
| 4 | mem_pressure | free | exhausted |

**Output classes:** `SAFE` · `LOW` · `MEDIUM` · `HIGH` · `CRITICAL`

```python
# Full frame (single model)
from bridge.python.myln import MylnFrame
frame = MylnFrame(size="T", n_classes=5).tune_security()
print(frame.predict([0.0, 0.0, 0.0, 0.0, 0.0]))    # → SAFE
print(frame.predict([0.9, 0.95, 0.8, 0.99, 0.85]))  # → CRITICAL

# Cascade (relay + full)
from bridge.python.myln import MylnCascade
cas = MylnCascade(threshold=0.80).tune_security()
label, conf, via_relay = cas.predict_with_path([0.5, 0.7, 0.3, 0.95, 0.6])
# → HIGH  46%  (routed to full — mass writes but proc is moderate)
```

---

## Tuners

A *tuner* sets the Router weights, Ring parameters and Center Line classifier of a frame by hand — **no training**.

| Tuner | Where | Input → Output | Use |
|---|---|---|---|
| `security` | `tuner/security_tuner.h` | 5 features → `SAFE`…`CRITICAL` | system monitoring (Chibitaru compatible) |
| `difficulty` | `tuner/difficulty_tuner.h` | 5 features → `CHAT`…`EXTREME` | grading how hard a request is (e.g. model routing) |
| `custom` | `include/myln/tune_config.h` | anything → anything | **any** setup from a JSON file — see below |

Each has a C++ function, a C API call and a Python method:

| | security | difficulty | custom (JSON) |
|---|---|---|---|
| C++ | `tune_security(frame)` / `tune_cascade_security(cas)` | `tune_difficulty(frame)` / `tune_cascade_difficulty(cas)` | `tune_custom(frame, path_or_json)` / `tune_cascade_custom(cas, ...)` |
| C | `myln_tune_security` / `myln_cascade_tune_security` | `myln_tune_difficulty` / `myln_cascade_tune_difficulty` | `myln_tune_custom` / `myln_cascade_tune_custom` |
| Python | `.tune_security()` | `.tune_difficulty()` | `.tune_custom(config)` |

---

## Difficulty grading

Turns a request into a difficulty level. The frame does not read text — you extract five features
(0.0–1.0) however you like, the tuner turns them into a level.

| Index | Feature | Meaning |
|---|---|---|
| 0 | `tech` | technical density: code, file names, technical terms |
| 1 | `length` | length of the request |
| 2 | `steps` | number of steps ("first… then… finally") |
| 3 | `scope` | blast radius: many files, whole project, deliverables — **weighted highest** |
| 4 | `reasoning` | how much thinking it needs: root cause, comparison, design |

**Output classes:** `0 CHAT (雑談)` · `1 EASY (易)` · `2 MEDIUM (中)` · `3 HARD (難)` · `4 EXTREME (最難)`

It is an additive model: `score = 1.5·tech + 1.0·length + 1.5·steps + 2.5·scope + 1.0·reasoning`, cut at
0.5 / 1.5 / 2.8 / 4.2. (Each slot contributes its share ×4 and the Center Line averages the four slots uniformly,
so the frame computes exactly that sum.) Relay and full see the same score; the relay just answers faster when it is far from a boundary.

```python
from bridge.python.myln import MylnCascade, DIFFICULTY_CLASSES

cas = MylnCascade(threshold=0.80).tune_difficulty()
probs, used_relay = cas.infer([0.67, 0.14, 1.0, 1.0, 1.0])   # [tech, length, steps, scope, reasoning]
print(DIFFICULTY_CLASSES[probs.index(max(probs))])            # → EXTREME
```

`bridge/python/difficulty.py` is a complete example: it extracts the five features from Japanese/English
request text with a few regexes and prints `{"level", "conf", "relay", "features"}` as JSON
(`echo '{"text":"..."}' | python3 bridge/python/difficulty.py`).

---

## Generic tuning API (JSON)

Anything the built-in tuners do can be expressed as a JSON file — Router weights (per-slot W, b),
Ring parameters and the Center Line classifier (W_cls). This is the portable "head config" format.

```python
from bridge.python.myln import MylnFrame, MylnCascade

frame = MylnFrame("T", 5).tune_custom("configs/security.json")       # a file path…
frame = MylnFrame("T", 5).tune_custom({"in_dim": 5, "router": ...})  # …or a dict / JSON string
cas   = MylnCascade(0.80).tune_custom("configs/difficulty_cascade.json")
```

```c
void* frame = myln_new("T", 5);
if (myln_tune_custom(frame, "configs/security.json") != 0)
    fprintf(stderr, "%s\n", myln_last_error());     // frame is left unchanged on error
```

- Invalid configs are rejected *before* anything is applied (and unknown keys are errors, so typos don't pass silently).
- No dependencies: the C++ side ships its own ~200-line JSON parser.
- `configs/` has ready-made `security*.json` and `difficulty*.json`, and the tests check that they give
  **bit-identical** probabilities to the C++ tuners.
- Full schema: [`docs/tuning-config.md`](docs/tuning-config.md).

---

## Tests

```bash
python3 tests/regression.py      # uses build/libmyln.dylib (or MYLN_LIB=path/to/libmyln)
```

Checks the difficulty output against a saved baseline (`tests/baseline/`), C++ tuner ⇔ JSON config equality,
the security tuner against its saved probabilities, and that invalid configs are rejected without side effects.

---

## Built with MYLN-FRAME

| Project | Description |
|---|---|
| [🌍 myln-earth-monitor](https://github.com/Sub-to/myln-earth-monitor) | Real-time satellite tracking + worldwide earthquake alerts — USGS + JMA, uses the separate [myln-heads-earthquake](https://github.com/Sub-to/myln-heads-earthquake) package |

---

## Architecture

```
include/myln/
  config.h        — frame size definitions (SS / T / S)
  frame.h         — Frame class: router + heads + ring + center
  router.h        — input → 4 slot vectors
  head.h          — Head interface (swappable)
  ring_attn.h     — Ring Attention: lateral sharing between heads
  center_line.h   — aggregation + classification (W_cls)
  cascade.h       — CascadeFrame: relay → confidence → full
  tune_config.h   — generic JSON tuning (tune_custom / tune_cascade_custom)
  mini_json.h     — dependency-free JSON parser used by tune_config.h
  math_ops.h      — Vec / Mat primitives

heads/
  passthrough_head.h   — identity (signal passes unchanged)
  zero_head.h          — silence (disabled slot)

tuner/
  security_tuner.h     — manual weight tuning, no training needed
  difficulty_tuner.h   — request difficulty: CHAT / EASY / MEDIUM / HARD / EXTREME

bridge/
  myln_c_api.h/.cpp    — universal C API
  python/myln.py       — Python ctypes wrapper (MylnFrame, MylnCascade)
  python/difficulty.py — text → 5 features → difficulty level (JSON in/out)

configs/               — ready-made JSON tunings (security, difficulty; frame + cascade)
docs/tuning-config.md  — JSON tuning schema
tests/                 — regression tests + saved baselines
```

---

## Swappable heads

Heads are the only part you change between domains.

```cpp
frame.set_head(0, std::make_unique<PassthroughHead>("proc"));
frame.set_head(1, std::make_unique<ZeroHead>());           // inactive slot
frame.set_head(2, std::make_unique<PassthroughHead>("file"));
frame.set_head(3, std::make_unique<DefaultHead>());         // learned head
```

**Shipped in the core:** `PassthroughHead`, `ZeroHead`, `DefaultHead`.

**Separate packages** (built on the core, kept out of it):
- [`myln-heads-earthquake`](https://github.com/Sub-to/myln-heads-earthquake) — `EarthquakeHead` + tuner, seismic severity classifier (~0.1 µs); powers [myln-earth-monitor](https://github.com/Sub-to/myln-earth-monitor).
  Extension packages configure a frame through `myln_frame_native()` (C API) / `MylnFrame.native_ptr` (Python).

**Future heads (planned):**
- `WeatherHead` — typhoon / disaster alert scoring
- `VoiceHead` — speech feature classification
- `CustomHead` — a head defined from a config file (Router/Ring/Center Line are already configurable via JSON)

---

## Why this exists

Most AI requires expensive hardware, cloud connectivity, and someone else's infrastructure.  
MYLN-FRAME runs on the machine in front of you — the one that's already there.

*Small enough to carry. Fast enough to matter. Free enough to trust.*

Human–AI coexistence shouldn't depend on a data center.

---

## Status

- [x] Core frame — router / ring-attention / center-line
- [x] Frame sizes — SS / T / S
- [x] Swappable heads — Passthrough, Zero, Default
- [x] Manual weight tuning — security monitoring (no training)
- [x] 2-stage cascade — relay + confidence threshold + full
- [x] Universal C API — Python, Node.js, Ruby, Go, Rust
- [x] Python bridge — `MylnFrame`, `MylnCascade`
- [x] Difficulty tuner — `tune_difficulty`
- [x] JSON tuning config — `myln_tune_custom` / `tune_custom` ([schema](docs/tuning-config.md))
- [ ] `.mhead` for *heads* — portable head definitions (weights of non-trivial heads)
- [ ] WeatherHead — typhoon / disaster alert
- [ ] CLI tool — `myln run --frame SS --head security.mhead`
- [ ] Distributed mode — heads over socket / gRPC

---

## License

MIT
