#include "../include/myln/frame.h"
#include "../heads/security_head.h"
#include <iostream>
#include <iomanip>

// 5-class threat levels (chibitaru compatible)
static const char* CLASSES[] = { "SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL" };

void run(myln::Frame& frame, const std::string& label, const myln::Vec& input) {
    auto probs = frame.forward(input);
    int  best  = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());

    std::cout << std::setw(20) << label << " → " << CLASSES[best] << "  [ ";
    for (int i = 0; i < 5; ++i)
        std::cout << CLASSES[i][0] << ":" << std::fixed << std::setprecision(2) << probs[i] << " ";
    std::cout << "]\n";
}

int main() {
    std::cout << "\n=== MYLN-FRAME hello ===\n\n";

    // ── MYLN-T with security heads ─────────────────────────
    auto frame = myln::make_t(5);
    auto sec   = myln::make_security_heads(frame.dim());
    for (int i = 0; i < 4; ++i)
        frame.set_head(i, std::move(sec[i]));

    std::cout << "[" << frame.tag() << "] dim=" << frame.dim()
              << "  heads: process / network / filesystem / resource\n\n";

    // Input features:
    // [proc_anomaly, cpu_spike, net_bytes_norm, file_change_rate, mem_pressure]
    // All values 0.0 (quiet) → 1.0 (extreme)

    run(frame, "idle system",    {0.0f, 0.05f, 0.01f, 0.0f,  0.1f });
    run(frame, "normal traffic", {0.1f, 0.3f,  0.4f,  0.05f, 0.25f});
    run(frame, "port scan",      {0.2f, 0.2f,  0.9f,  0.1f,  0.2f });
    run(frame, "mass file write",{0.5f, 0.7f,  0.3f,  0.95f, 0.6f });
    run(frame, "ransomware",     {0.9f, 0.95f, 0.8f,  0.99f, 0.85f});

    std::cout << "\n── MYLN-SS (same heads, smaller frame) ──\n\n";

    auto frame_ss = myln::make_ss(5);
    auto sec_ss   = myln::make_security_heads(frame_ss.dim());
    for (int i = 0; i < 4; ++i)
        frame_ss.set_head(i, std::move(sec_ss[i]));

    run(frame_ss, "idle system",    {0.0f, 0.05f, 0.01f, 0.0f,  0.1f });
    run(frame_ss, "ransomware",     {0.9f, 0.95f, 0.8f,  0.99f, 0.85f});

    std::cout << "\ndone.\n";
    return 0;
}
