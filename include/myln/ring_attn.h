#pragma once
#include "math_ops.h"
#include <array>

namespace myln {

// ── RING ATTENTION ─────────────────────────────────────────
// 4 heads arranged in a ring: H0 ↔ H1 ↔ H2 ↔ H3 ↔ H0
// Each head merges its own output with both neighbors.
// This is the "クルクル回る" lateral information sharing.
class RingAttention {
    static constexpr int N = 4;
    int dim_;
    // Each position: project [self + left + right] (3×dim) → dim
    std::array<Mat, N> W_;
    std::array<Vec, N> b_;

public:
    explicit RingAttention(int dim, unsigned seed = 20) : dim_(dim) {
        for (int i = 0; i < N; ++i) {
            W_[i] = rand_mat(dim, dim * 3, 0.1f, seed + i);
            b_[i] = zeros(dim);
        }
    }

    std::array<Vec, N> forward(const std::array<Vec, N>& h) const {
        std::array<Vec, N> out;
        for (int i = 0; i < N; ++i) {
            int left  = (i + N - 1) % N;
            int right = (i + 1)     % N;
            Vec combined = concat(concat(h[i], h[left]), h[right]);
            out[i] = layer_norm(linear(W_[i], b_[i], combined, dim_ * 3, dim_));
        }
        return out;
    }
};

} // namespace myln
