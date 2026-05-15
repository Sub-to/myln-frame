#pragma once
#include "math_ops.h"
#include <array>

namespace myln {

// ── ROUTER (APEX) ──────────────────────────────────────────
// Takes raw input vector → projects to 4 slot vectors of size `dim`.
// Each slot gets a different learned projection.
class Router {
    static constexpr int N = 4;
    int in_dim_, out_dim_;
    std::array<Mat, N> W_;
    std::array<Vec, N> b_;

public:
    Router(int in_dim, int out_dim, unsigned seed = 10)
        : in_dim_(in_dim), out_dim_(out_dim)
    {
        for (int s = 0; s < N; ++s) {
            W_[s] = rand_mat(out_dim, in_dim, 0.1f, seed + s);
            b_[s] = zeros(out_dim);
        }
    }

    int in_dim() const { return in_dim_; }

    // Returns 4 projected + layer-normed vectors
    std::array<Vec, N> forward(const Vec& input) const {
        std::array<Vec, N> out;
        for (int s = 0; s < N; ++s)
            out[s] = layer_norm(linear(W_[s], b_[s], input, in_dim_, out_dim_));
        return out;
    }
};

} // namespace myln
