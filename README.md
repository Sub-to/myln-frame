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
| Latency¹ | ~0.6 µs | ~1.6 µs | ~2.8 µs |
| Best for | microcontrollers / SBC | old PC / edge | laptop / server |
| GPU needed | ✗ | ✗ | ✗ |

¹ Tuned security frame, one inference, x86 Xeon @ 2.1 GHz, `-O3`, measured with `myln_bench` (see [Performance](#performance)).

---

## Cascade: fast path + precise path

When input is _obvious_, skip the heavy frame entirely.

```
INPUT
  ↓
[RELAY]  ← SS, 2 heads (proc + file)
  ↓
confidence ≥ 80%?
  ├─ YES → output immediately          ← clear threats / idle
  └─ NO  → [FULL]  T, 4 heads          ← ambiguous cases
```

Real numbers on a Raspberry Pi 5 (v0.1.0):

| Case | Path | Latency |
|---|---|---|
| Idle / all-clear | relay only | **9 µs** |
| Ransomware pattern | relay only | **9 µs** |
| Mixed / borderline | relay → full | 118 µs |

> **v0.2.0 note.** The whole frame got ~25× faster (see [Performance](#performance)),
> so the relay now saves ~1 µs instead of ~100 µs. The relay only looks at
> `proc` and `file`, so it is an _approximation_ of the full frame: it agrees
> with it on the clear-cut anchors above, but on arbitrary inputs an early exit
> can differ from the full frame's answer (`myln_bench` prints the agreement).
> If you need the exact answer every time, use `MylnFrame` — it is now just as fast.

---

## Quick start

### Build

```bash
git clone https://github.com/Sub-to/myln-frame.git
cd myln-frame
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure   # C++ core, C API, Python bridge
# → build/libmyln.dylib  (macOS)
# → build/libmyln.so     (Linux)
```

| CMake option | Default | |
|---|---|---|
| `MYLN_NATIVE` | OFF | `-march=native` (not faster in our measurements, and the binary only runs on the same CPU family) |
| `MYLN_BUILD_TESTS` / `_BENCH` / `_EXAMPLES` | ON | |
| `MYLN_SANITIZE` | OFF | AddressSanitizer + UBSan |

Use it from another CMake project (header-only core, or the C library):

```bash
cmake --install build --prefix /usr/local
```
```cmake
find_package(myln REQUIRED)
target_link_libraries(app PRIVATE myln::headers)   # #include <myln/frame.h>
# or: myln::myln                                    # libmyln + C API
```

The Python bridge finds `build/libmyln.*` automatically; set `MYLN_LIB=/path/to/libmyln.so` to point elsewhere.

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

**Confidence calibration.** Out of the box the five class probabilities are soft
(the winning class is ~0.5 on the anchor scenarios). `sharpness` scales the
logits: the predicted class never changes, only the confidence does.

```python
frame = MylnFrame(size="T").tune_security(sharpness=3.0)
frame.predict_with_score([0.9, 0.95, 0.8, 0.99, 0.85])   # → ('CRITICAL', 0.71)
```
```cpp
myln::SecurityTuneParams p; p.logit_scale = 3.0f;        // C++
myln_set_logit_scale(frame, 3.0f);                        // C API
```

**Input safety.** Features must be finite. NaN/Inf, a wrong feature count, or an
overflowing input is rejected (`std::invalid_argument` / `ValueError` / `NULL`
from the C API + `myln_last_error()`) instead of silently producing a class.

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

## Built with MYLN-FRAME

| Project | Description |
|---|---|
| [🌍 myln-earth-monitor](https://github.com/Sub-to/myln-earth-monitor) | Real-time satellite tracking + worldwide earthquake alerts — USGS + JMA, EarthquakeHead (~0.1 µs) |

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
  math_ops.h      — Vec / Mat primitives, allocation-free kernels, PackedMat (sparse/dense)

heads/
  passthrough_head.h   — identity (signal passes unchanged)
  zero_head.h          — silence (disabled slot)
  earthquake_head.h    — ultra-light seismic classifier (~0.1 µs, no matrix multiply)

tuner/
  security_tuner.h     — manual weight tuning, no training needed
  earthquake_tuner.h   — seismic intensity → SAFE/LOW/MEDIUM/HIGH/CRITICAL

bridge/
  myln_c_api.h/.cpp    — universal C API
  python/myln.py       — Python ctypes wrapper (MylnFrame, MylnCascade)

tests/                 — C++ core / C API tests, Python bridge tests (ctest)
bench/bench.cpp        — latency, monotonicity, anchors, cascade fidelity
cmake/                 — find_package(myln) config
.github/workflows/     — CI: gcc / clang / macOS, ASan+UBSan, install smoke test
```

---

## Performance

`myln_bench` (tuned security frame, 5000 random inputs × 5, x86 Xeon @ 2.1 GHz, `-O3`, mean per inference):

| Frame | v0.1.0 | v0.2.0 | speed-up |
|---|---|---|---|
| MYLN-SS | 3.0 µs | 0.62 µs | **4.8×** |
| MYLN-T | 39 µs | 1.6 µs | **24×** |
| MYLN-S | 150 µs | 2.7 µs | **55×** |

What changed, with **identical decisions** (the tests replay ~6000 classes recorded from v0.1.0 and require 0 mismatches):

- **Sparse weights.** Hand-tuned layers are almost all zeros (identity / pick-one-feature). `PackedMat` stores
  them in CSR form when ≤35% non-zero and skips the multiply-by-zero; random-weight layers stay dense.
- **No allocation in the hot path.** `Frame::forward_into()` runs router → heads → ring → center on
  pre-allocated buffers. Heads get `forward_into()` (default implementation falls back to `forward()`, so
  existing custom heads keep working).
- **Center line.** `Wkᵀq` is precomputed when the query/keys are set, so attention scoring is four dot products
  instead of four dim×dim matrix multiplies. Ring attention no longer concatenates `[self|left|right]` per slot.

Run it yourself: `./build/myln_bench` (or `myln_bench 20000`).

---

## Swappable heads

Heads are the only part you change between domains.

```cpp
frame.set_head(0, std::make_unique<PassthroughHead>("proc"));
frame.set_head(1, std::make_unique<ZeroHead>());           // inactive slot
frame.set_head(2, std::make_unique<PassthroughHead>("file"));
frame.set_head(3, std::make_unique<DefaultHead>());         // learned head
```

**Shipped:**
- `EarthquakeHead` — seismic intensity classifier, powers [myln-earth-monitor](https://github.com/Sub-to/myln-earth-monitor)

**Future heads (planned):**
- `WeatherHead` — typhoon / disaster alert scoring
- `VoiceHead` — speech feature classification
- `CustomHead` — loaded from `.mhead` config file

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
- [x] Tests (ctest) + CI + benchmark, `find_package(myln)` install
- [x] Input validation, error-safe C API (`myln_last_error`)
- [ ] `.mhead` file format — portable head configs
- [ ] WeatherHead — typhoon / disaster alert
- [ ] CLI tool — `myln run --frame SS --head security.mhead`
- [ ] Distributed mode — heads over socket / gRPC

---

## Changelog

### 0.2.0

**Faster, same answers** — see [Performance](#performance).

**Fixes**
- `tune_earthquake`: slot weights were applied twice (router _and_ head), so even a 震度1 quake came out
  `CRITICAL`. The router now passes features through and `EarthquakeHead` weights them once. Note that the
  shipped weights (`w_int=7`, `w_mag=2.5`, `w_depth=1.0`, …) still alert earlier than the class table in
  `earthquake_tuner.h` describes (震度3 → `CRITICAL`); recalibrating them needs real seismic data.
- `tune_security` with `in_dim < 5` wrote to the wrong feature rows; it now throws.
- A tuned frame fed a different number of features used to silently re-initialise the router with random
  weights. It now throws.
- C API: C++ exceptions could cross the C boundary; null handles crashed. All entry points are guarded.
  `myln_version()` now reports the real version.
- The `__pycache__` that was committed is gone from the tree.

**Behaviour changes to be aware of**
- NaN/Inf and wrong-size input raise instead of returning a class.
- `CascadeFrame::run()` is no longer `const` (it never really was); counters are `long`.
- C API: `myln_tune_*` / `myln_cascade_tune_security` return `int` (0 / -1) instead of `void`.
  `myln_infer` returns `NULL` on error. New: `myln_last_error`, `myln_infer_into`,
  `myln_cascade_infer_into`, `myln_set_logit_scale`.
- `-march=native` is now opt-in (`-DMYLN_NATIVE=ON`) so binaries are portable.
- Python: `MylnError`, `close()` / context manager, `tune_earthquake()`, `tune_security(sharpness=…)`.

---

## License

MIT
