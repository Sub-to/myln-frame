"""現行 difficulty.py の出力を基準として保存する(リファクタ前に1回だけ実行)。"""
import json, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
texts = json.loads((Path(__file__).parent / "texts.json").read_text(encoding="utf-8"))
out = []
for t in texts:
    r = subprocess.run([sys.executable, str(root / "bridge/python/difficulty.py")],
                       input=json.dumps({"text": t}), capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    out.append({"text": t, **d})
(Path(__file__).parent / "difficulty_baseline.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for o in out: print(o["level"], o["conf"], o["relay"], o["text"][:40])
