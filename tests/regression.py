#!/usr/bin/env python3
"""
MYLN-FRAME 回帰テスト
=====================
  python3 tests/regression.py                 # build/libmyln.dylib(本番)を使う
  MYLN_LIB=build-dev/libmyln.dylib python3 tests/regression.py   # 開発ビルドを使う

確認すること:
  1. difficulty.py の出力が基準(tests/baseline/difficulty_baseline.json)と同じ level / relay / features
  2. conf も基準と一致(丸め後。差が出たら報告する)
  3. C++ チューナー ⇔ JSON 設定 が同じ確率を返す(security / difficulty, Frame / Cascade)
  4. security の確率ベクトルが基準と一致
  5. 不正な設定はエラーになり、frame は変更されない(部分適用なし)
  6. Pi 拡張と同じ呼び出し(stdin JSON → python3 difficulty.py)が従来のキーで返る
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bridge" / "python"))
from myln import MylnFrame, MylnCascade, DIFFICULTY_CLASSES  # noqa: E402

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


# ── 1,2,6: difficulty.py(Pi拡張と同じ呼び出し方)────────────
print("[1] difficulty.py vs baseline (subprocess, same as Pi extension)")
base = json.loads((ROOT / "tests/baseline/difficulty_baseline.json").read_text(encoding="utf-8"))
for b in base:
    r = subprocess.run(["python3", str(ROOT / "bridge/python/difficulty.py")],
                       input=json.dumps({"text": b["text"]}), capture_output=True, text=True, env=os.environ)
    try:
        d = json.loads(r.stdout)
    except Exception:
        check(False, f"{b['text'][:30]!r}: invalid output {r.stdout!r} {r.stderr[-200:]!r}")
        continue
    check(sorted(d.keys()) == ["conf", "features", "level", "relay"], f"keys {b['text'][:20]!r}")
    check(d["level"] == b["level"], f"level {d['level']}=={b['level']}  {b['text'][:30]!r}")
    check(d["relay"] == b["relay"] and d["features"] == b["features"], f"relay/features {b['text'][:20]!r}")
    check(d["conf"] == b["conf"], f"conf {d['conf']}=={b['conf']}  {b['text'][:20]!r}")

print("[6] Pi extension call: echo '{\"text\":\"こんにちは\"}' | python3 difficulty.py")
r = subprocess.run(["python3", str(ROOT / "bridge/python/difficulty.py")],
                   input='{"text":"こんにちは"}', capture_output=True, text=True, env=os.environ)
print("      ->", r.stdout.strip())
check(r.returncode == 0 and set(json.loads(r.stdout)) == {"level", "conf", "relay", "features"}, "exit 0 and keys")

# ── 3: C++ チューナー ⇔ JSON 設定 ────────────────────────────
print("[3] C++ tuner == JSON config")
probe = [[0, 0.05, 0.01, 0, 0.1], [0.1, 0.3, 0.4, 0.05, 0.25], [0.2, 0.2, 0.9, 0.1, 0.2],
         [0.5, 0.7, 0.3, 0.95, 0.6], [0.9, 0.95, 0.8, 0.99, 0.85], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1],
         [0.33, 0, 0, 0, 0.67], [0.67, 0.14, 1, 1, 1], [1, 0.1, 0, 0.5, 1], [0.5, 0.5, 0.5, 0.5, 0.5]]
cfgdir = ROOT / "configs"
for size in ("T", "SS"):
    for cpp, js in (("tune_security", "security.json"), ("tune_difficulty", "difficulty.json")):
        a = getattr(MylnFrame(size, 5), cpp)()
        b = MylnFrame(size, 5).tune_custom(cfgdir / js)
        d = max(maxdiff(a.infer(x), b.infer(x)) for x in probe)
        check(d == 0.0, f"Frame {size} {cpp} vs {js}: max diff {d}")
for cpp, js in (("tune_security", "security_cascade.json"), ("tune_difficulty", "difficulty_cascade.json")):
    a = getattr(MylnCascade(0.8), cpp)()
    b = MylnCascade(0.8).tune_custom(cfgdir / js)
    d, same = 0.0, True
    for x in probe:
        pa, ra = a.infer(x); pb, rb = b.infer(x)
        d = max(d, maxdiff(pa, pb)); same &= (ra == rb)
    check(d == 0.0 and same, f"Cascade {cpp} vs {js}: max diff {d}, same relay path={same}")

# ── 4: security 基準 ───────────────────────────────────────
print("[4] security vs saved baseline")
sb = json.loads((ROOT / "tests/baseline/security_baseline.json").read_text())
fT, fS, cas = MylnFrame("T", 5).tune_security(), MylnFrame("SS", 5).tune_security(), MylnCascade(0.8).tune_security()
d = 0.0
for i, x in enumerate(sb["inputs"]):
    d = max(d, maxdiff(fT.infer(x), sb["frame_T"][i]), maxdiff(fS.infer(x), sb["frame_SS"][i]))
    p, r = cas.infer(x)
    d = max(d, maxdiff(p, sb["cascade"][i]["probs"]))
    check(r == sb["cascade"][i]["relay"], f"cascade relay path input#{i}")
check(d == 0.0, f"security probs unchanged (max diff {d})")

# ── 5: 不正な設定 ─────────────────────────────────────────
print("[5] invalid configs are rejected without partial application")
f = MylnFrame("T", 5).tune_security()
before = [f.infer(x) for x in probe]
good_router = {"normalize": False, "slots": [{"terms": [[0, 0, 50.0]]}, {}, {}, {}]}
bad_cfgs = {
    "bad cls class index": {"in_dim": 5, "router": good_router,
                            "center": {"cls": {"terms": [[9, 0, 1.0]]}}},
    "bad in index": {"in_dim": 5, "router": {"slots": [{"terms": [[0, 7, 1.0]]}, {}, {}, {}]}},
    "wrong slot count": {"in_dim": 5, "router": {"slots": [{}]}},
    "unknown key": {"in_dim": 5, "routr": {}},
    "unknown head": {"heads": ["passthrough", "zero", "nope", "zero"]},
    "size mismatch": {"size": "SS"},
    "n_classes mismatch": {"n_classes": 3},
    "missing in_dim": {"router": good_router},
    "not json": "{ nope",
}
for name, cfg in bad_cfgs.items():
    try:
        f.tune_custom(cfg)
        check(False, f"{name}: should have raised ValueError")
    except ValueError as e:
        check(True, f"{name}: ValueError ({str(e)[:60]})")
check([f.infer(x) for x in probe] == before, "frame unchanged after rejected configs")
try:
    MylnFrame("T", 5).tune_custom(str(ROOT / "configs" / "no_such_file.json"))
    check(False, "missing file should raise")
except ValueError:
    check(True, "missing file: ValueError")
f2 = MylnFrame("T", 5).tune_custom(cfgdir / "difficulty.json")
check(f2.classes == DIFFICULTY_CLASSES and f2.predict([0, 0, 0, 0, 0]) == "CHAT", "classes from config -> predict label")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    sys.exit(1)
print("ALL PASSED")
