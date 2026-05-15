#include "../include/myln/frame.h"
#include "../tuner/security_tuner.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static const char* CLASSES[] = { "SAFE", "LOW ", "MED ", "HIGH", "CRIT" };
static const char* ICONS[]   = { "✅",   "🔵",  "🟡",  "🔴",  "💀"  };

void run(myln::Frame& frame, const std::string& label, const myln::Vec& input) {
    auto probs = frame.forward(input);
    int  best  = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());

    std::cout << ICONS[best] << " " << std::setw(22) << std::left << label
              << " → " << CLASSES[best] << "  [";
    for (int i = 0; i < 5; ++i)
        std::cout << std::fixed << std::setprecision(2) << probs[i]
                  << (i<4 ? " " : "");
    std::cout << "]\n";
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════╗\n";
    std::cout <<   "║   MYLN-FRAME  security tuning demo      ║\n";
    std::cout <<   "╚══════════════════════════════════════════╝\n\n";

    // ── MYLN-T + セキュリティチューニング ──────────────────
    auto frame = myln::make_t(5);
    myln::tune_security(frame);

    std::cout << "[" << frame.tag() << "] チューニング完了\n";
    std::cout << "入力: [proc, cpu, net, file, mem] (0.0=正常 / 1.0=極限)\n\n";

    // 入力 features: [proc_anomaly, cpu_spike, net_bytes, file_change, mem_pressure]
    run(frame, "アイドル",           {0.00f, 0.05f, 0.01f, 0.00f, 0.10f});
    run(frame, "通常トラフィック",   {0.10f, 0.30f, 0.40f, 0.05f, 0.25f});
    run(frame, "ポートスキャン",     {0.20f, 0.20f, 0.90f, 0.10f, 0.20f});
    run(frame, "大量ファイル書き込み",{0.50f, 0.70f, 0.30f, 0.95f, 0.60f});
    run(frame, "ランサムウェア",     {0.90f, 0.95f, 0.80f, 0.99f, 0.85f});

    std::cout << "\n── MYLN-SS 同じチューニング ──\n\n";
    auto frame_ss = myln::make_ss(5);
    myln::tune_security(frame_ss);
    std::cout << "[" << frame_ss.tag() << "]\n";
    run(frame_ss, "アイドル",       {0.00f, 0.05f, 0.01f, 0.00f, 0.10f});
    run(frame_ss, "ランサムウェア", {0.90f, 0.95f, 0.80f, 0.99f, 0.85f});

    std::cout << "\n── パラメータ調整例 ──\n\n";
    myln::SecurityTuneParams strict;
    strict.w_file = 7.0f;   // ファイル変化の感度を上げる
    strict.bias_crit = -1.0f; // CRITICALの閾値を下げる（厳しくする）
    auto frame_strict = myln::make_t(5);
    myln::tune_security(frame_strict, strict);
    std::cout << "[MYLN-T / strict mode]\n";
    run(frame_strict, "大量ファイル書き込み", {0.50f, 0.70f, 0.30f, 0.95f, 0.60f});
    run(frame_strict, "ランサムウェア",       {0.90f, 0.95f, 0.80f, 0.99f, 0.85f});

    std::cout << "\ndone.\n";
    return 0;
}
