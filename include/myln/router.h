#pragma once
#include "math_ops.h"
#include <array>

namespace myln {

// ── ROUTER (APEX) ──────────────────────────────────────────
// Projects input → 4 slot vectors.
// normalize=false でチューニング用途に絶対値を保持する。
class Router {
    static constexpr int N = 4;
    int in_dim_, out_dim_;
    bool normalize_;
    std::array<Mat, N> W_;
    std::array<Vec, N> b_;

public:
    Router(int in_dim, int out_dim, bool normalize = true, unsigned seed = 10)
        : in_dim_(in_dim), out_dim_(out_dim), normalize_(normalize)
    {
        for (int s = 0; s < N; ++s) {
            W_[s] = rand_mat(out_dim, in_dim, 0.1f, seed + s);
            b_[s] = zeros(out_dim);
        }
    }

    // 手動チューニング用: スロット s の重みをそのまま設定する
    void set_slot(int s, Mat W, Vec b) {
        W_[s] = std::move(W);
        b_[s] = std::move(b);
    }

    void set_normalize(bool v) { normalize_ = v; }
    int  in_dim()  const { return in_dim_;  }
    int  out_dim() const { return out_dim_; }

    std::array<Vec, N> forward(const Vec& input) const {
        std::array<Vec, N> out;
        for (int s = 0; s < N; ++s) {
            out[s] = linear(W_[s], b_[s], input, in_dim_, out_dim_);
            if (normalize_) out[s] = layer_norm(out[s]);
        }
        return out;
    }
};

} // namespace myln
