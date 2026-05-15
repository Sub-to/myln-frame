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
  math_ops.h      — Vec / Mat primitives

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
- [ ] `.mhead` file format — portable head configs
- [ ] WeatherHead — typhoon / disaster alert
- [ ] CLI tool — `myln run --frame SS --head security.mhead`
- [ ] Distributed mode — heads over socket / gRPC

---

## License

MIT
