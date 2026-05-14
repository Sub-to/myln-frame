# ⚡ MYLN-FRAME

> A lightweight AI inference framework. No GPU. No cloud. Just speed.

Named after the **myelin sheath** — the biological insulator that makes neural signals fast and efficient.

---

## What it is

A minimal pyramid-style attention framework with swappable heads.  
Same frame. Different heads. Any device.

```
          [ROUTER]
        ↙  ↓  ↓  ↘
      [A] [B] [C] [D]   ← swap these
        ↕   ↕   ↕   ↕   ← heads talk to each other
          [CENTER]       ← aggregates everything
              ↓
           OUTPUT
```

---

## Three sizes

| | MYLN-SS ⚡ | MYLN-T 🪶 | MYLN-S 💪 |
|--|--|--|--|
| RAM | ~1 MB | ~20 MB | ~100 MB |
| Speed | < 0.1ms | < 1ms | < 5ms |
| Target | embedded / SBC | old PC / USB | laptop |
| GPU | ✗ | ✗ | ✗ |

---

## Swappable heads (.mhead)

```bash
# Security monitoring
myln run --frame T --head security.mhead

# Weather / disaster alerts
myln run --frame SS --head weather.mhead

# Any domain you build
myln run --frame SS --head your_own.mhead
```

One frame. Infinite use cases.

---

## Why

Most AI runs on expensive hardware, in the cloud, controlled by someone else.  
MYLN-FRAME runs on the machine in front of you.  
*Small enough to carry. Fast enough to matter. Free enough to trust.*

---

## Status

🚧 Early design phase. C++ implementation in progress.

- [ ] Core frame (router / ring-attention / center-line)
- [ ] MYLN-T first
- [ ] security.mhead (chibitaru integration)
- [ ] MYLN-SS / MYLN-S

---

## License

MIT
