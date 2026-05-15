#pragma once
#include "../include/myln/head.h"

// ── Security Head ──────────────────────────────────────────
// Specialized head for threat detection (chibitaru integration).
// Biases attention toward known threat signal patterns.
//
// Input slots (assigned by router):
//   slot 0 → process behavior features
//   slot 1 → network pattern features
//   slot 2 → filesystem activity features
//   slot 3 → resource usage features

namespace myln {

class SecurityHead : public Head {
    std::string slot_name_;
    Mat W1_, W2_, W_bias_;
    Vec b1_, b2_;

    // Known threat signal weights (hand-initialized, no training needed)
    // Positive = threat indicator, Negative = safe indicator
    Vec threat_bias_;

public:
    SecurityHead(int dim, const std::string& slot, unsigned seed = 99)
        : slot_name_("security/" + slot)
    {
        int hidden = dim * 2;
        W1_    = rand_mat(hidden, dim,    0.1f, seed);
        b1_    = zeros(hidden);
        W2_    = rand_mat(dim,    hidden, 0.1f, seed + 1);
        b2_    = zeros(dim);

        // Threat bias: pushes high-activation dims toward threat representation
        threat_bias_.resize(dim, 0.f);
        for (int i = 0; i < dim / 4; ++i)
            threat_bias_[i] = 0.5f; // amplify first quarter of dims as "threat channel"
    }

    std::string name() const override { return slot_name_; }

    Vec forward(const Vec& x, int dim) override {
        int hidden = dim * 2;
        Vec h = relu(linear(W1_, b1_, x, dim, hidden));
        Vec y = linear(W2_, b2_, h, hidden, dim);

        // Apply threat bias (scaled by input signal strength)
        float signal = dot(x, x) / (float)x.size(); // mean squared activation
        for (int i = 0; i < dim; ++i)
            y[i] += threat_bias_[i] * signal;

        return layer_norm(y);
    }
};

// Convenience: build all 4 security heads
inline std::array<HeadPtr, 4> make_security_heads(int dim) {
    std::array<HeadPtr, 4> heads;
    const char* names[] = {"process", "network", "filesystem", "resource"};
    for (int i = 0; i < 4; ++i)
        heads[i] = std::make_unique<SecurityHead>(dim, names[i], 99u + (unsigned)i);
    return heads;
}

} // namespace myln
