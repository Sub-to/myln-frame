"""security チューナーの確率ベクトルを基準として保存する(リファクタ前に1回だけ実行)。"""
import json, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bridge/python"))
from myln import MylnFrame, MylnCascade

inputs = [[0, 0.05, 0.01, 0, 0.1], [0.1, 0.3, 0.4, 0.05, 0.25], [0.2, 0.2, 0.9, 0.1, 0.2],
          [0.5, 0.7, 0.3, 0.95, 0.6], [0.9, 0.95, 0.8, 0.99, 0.85], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1],
          [0.33, 0.0, 0.0, 0.0, 0.67], [0.67, 0.14, 1.0, 1.0, 1.0], [1.0, 0.1, 0.0, 0.5, 1.0]]
out = {"inputs": inputs, "frame_T": [], "frame_SS": [], "cascade": []}
fT = MylnFrame("T", 5).tune_security(); fS = MylnFrame("SS", 5).tune_security()
cas = MylnCascade(0.80).tune_security()
for x in inputs:
    out["frame_T"].append(fT.infer(x)); out["frame_SS"].append(fS.infer(x))
    p, r = cas.infer(x); out["cascade"].append({"probs": p, "relay": r})
Path(__file__).with_name("security_baseline.json").write_text(json.dumps(out, indent=1))
print("saved", len(inputs), "inputs")
