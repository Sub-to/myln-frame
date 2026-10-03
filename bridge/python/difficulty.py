"""
MYLN 難易度判定 (Pi ルーター用)
================================
依頼文 → 5特徴量 → MYLN cascade (security_tuner を流用) → 難易度 0〜4

security_tuner の5入力/5クラスを、次のように読み替えている(C++は無改造):
  入力 [proc, cpu, net, file, mem]  →  [技術度, 長さ, 手数, 影響範囲(最重要), 推論度]
  出力 SAFE/LOW/MEDIUM/HIGH/CRITICAL →  0=雑談 1=易 2=中 3=難 4=最難

使い方: echo '{"text":"..."}' | python3 difficulty.py   → {"level":2,"conf":0.8,"relay":true,"features":[...]}
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from myln import MylnCascade  # noqa: E402

TECH = re.compile(r"```|\.(py|ts|js|tsx|swift|cpp|c|h|rs|go|html|css|json|yaml|sh)\b|関数|クラス|バグ|エラー|実装|コード|プログラム|スクリプト|API|ビルド|コンパイル|テスト|function|class|bug|error|implement|compile|script|build|test", re.I)
STEPS = re.compile(r"まず|次に|最後に|そのあと|その後|それから|してから|\b[1-9][.)]|①|②|and then|then |step", re.I)
SCOPE = re.compile(r"全体|全部|すべて|プロジェクト|リポジトリ|複数|ディレクトリ|フォルダ|アーキテクチャ|リファクタ|移行|設計|repo|refactor|migrat|architecture|across|entire|whole", re.I)
REASON = re.compile(r"なぜ|原因|理由|調査|比較|最適|トレードオフ|どちらが|デバッグ|設計|どうして|why|root cause|investigate|compare|trade-?off|optimi[sz]e|debug|design", re.I)
# 2026-10-03追加: コードを含まない「計算・図・条件の多さ」型の難しさを拾う
CALC = re.compile(r"計算|算出|割り出|わりだ|求め(?:て|る)|角度|数式|方程式|シミュレー|モデル化|解析|最適化|物理|積分|微分|確率|統計|速度|軌道|座標|緯度|経度|calc|simulat|equation|angle|formula", re.I)
VIS = re.compile(r"図(?:に|で|を|示|表)|グラフ|チャート|可視化|プロット|描(?:い|画)|地図|作図|plot|chart|diagram|graph|draw|render", re.I)
ASSUME = re.compile(r"想定|仮定|条件|前提|として|assum|suppose", re.I)
NUMS = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?")

PATHS = re.compile(r"[\w./~-]+/[\w.-]+|[\w-]+\.(?:py|ts|js|swift|cpp|h|json|md)\b")


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def features(text: str) -> list:
    n_tech = len(TECH.findall(text))
    n_step = len(STEPS.findall(text))
    n_scope = len(SCOPE.findall(text)) + max(0, len(set(PATHS.findall(text))) - 1)
    n_reason = len(REASON.findall(text))
    n_calc = len(CALC.findall(text))
    n_vis = len(VIS.findall(text))
    n_assume = len(ASSUME.findall(text))
    n_num = len(NUMS.findall(text))
    return [
        clip((n_tech + n_calc) / 3),  # proc  : 技術度(計算・物理系の語も含む)
        clip(len(text) / 400),   # cpu   : 長さ
        clip((n_step + max(0, n_num - 2) / 2) / 3),  # net   : 手数(数値条件が多いと加算)
        clip((n_scope + n_vis) / 2),  # file  : 影響範囲(成果物=図・グラフ作成も含む。MYLNが最重視)
        clip((n_reason + 0.5 * n_calc + 0.5 * n_assume) / 1.5),  # mem   : 推論度(想定・条件の語、計算の語も含む)
    ]


_cas = MylnCascade(threshold=0.80).tune_security()


def classify(text: str) -> dict:
    f = features(text)
    probs, used_relay = _cas.infer(f)
    level = probs.index(max(probs))
    return {"level": level, "conf": round(max(probs), 3), "relay": bool(used_relay), "features": [round(v, 2) for v in f]}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        for t in ["こんにちは", "今日の天気は?", "この関数の名前を変えて", "main.pyのバグの原因を調べて直して",
                  "まずプロジェクト全体の設計を調査して、次にsrc/とtests/を複数ファイルでリファクタして、最後にテストして"]:
            print(classify(t), t)
        sys.exit(0)
    req = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(classify(str(req.get("text", ""))), ensure_ascii=False))
