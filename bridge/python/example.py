"""
MYLN-FRAME Python Bridge 使用例
チビタルのセキュリティ判定をPythonから呼ぶ。
"""
from myln import MylnFrame

ICONS = {"SAFE": "✅", "LOW": "🔵", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "💀"}

def main():
    # フレーム作成 → チューニング (1行)
    frame = MylnFrame(size="T", n_classes=5).tune_security(in_dim=5)
    print(f"[{frame.tag}] v{frame.version} ready\n")

    events = [
        ("アイドル",              [0.00, 0.05, 0.01, 0.00, 0.10]),
        ("通常トラフィック",      [0.10, 0.30, 0.40, 0.05, 0.25]),
        ("ポートスキャン",        [0.20, 0.20, 0.90, 0.10, 0.20]),
        ("大量ファイル書き込み",  [0.50, 0.70, 0.30, 0.95, 0.60]),
        ("ランサムウェア",        [0.90, 0.95, 0.80, 0.99, 0.85]),
    ]

    for label, features in events:
        level, score = frame.predict_with_score(features)
        icon = ICONS.get(level, "⚪")
        print(f"  {icon} {label:<20} → {level:<8} ({score:.0%})")

    print()
    # SS フレーム（超軽量）でも同じAPI
    frame_ss = MylnFrame(size="SS").tune_security()
    print(f"[{frame_ss.tag}] ランサムウェア → {frame_ss.predict([0.9,0.95,0.8,0.99,0.85])}")

if __name__ == "__main__":
    main()
