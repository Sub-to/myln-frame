#pragma once
#include "math_ops.h"
#include <array>

namespace myln {

// ── RING ATTENTION ─────────────────────────────────────────
// H0 ↔ H1 ↔ H2 ↔ H3 ↔ H0
// set_near_identity() でチューニング用途の近似恒等写像を設定できる。
class RingAttention {
    static constexpr int N = 4;
    int dim_;
    std::array<Mat, N> W_;  // each: [dim × dim*3]
    std::array<Vec, N> b_;

public:
    explicit RingAttention(int dim, unsigned seed = 20) : dim_(dim) {
        for (int i = 0; i < N; ++i) {
            W_[i] = rand_mat(dim, dim * 3, 0.1f, seed + i);
            b_[i] = zeros(dim);
        }
    }

    // チューニング用: self=w_self, 左右=w_nb の加重平均にする
    void set_near_identity(float w_self = 0.6f, float w_nb = 0.2f) {
        for (int i = 0; i < N; ++i) {
            W_[i].assign(dim_ * dim_ * 3, 0.f);
            b_[i].assign(dim_, 0.f);
            for (int d = 0; d < dim_; ++d) {
                W_[i][d * dim_ * 3 + d]            = w_self; // self
                W_[i][d * dim_ * 3 + dim_ + d]     = w_nb;   // left
                W_[i][d * dim_ * 3 + dim_ * 2 + d] = w_nb;   // right
            }
        }
    }

    // 個別設定用
    void set_merge(int i, Mat W, Vec b) {
        W_[i] = std::move(W);
        b_[i] = std::move(b);
    }

    std::array<Vec, N> forward(const std::array<Vec, N>& h) const {
        std::array<Vec, N> out;
        for (int i = 0; i < N; ++i) {
            int left  = (i + N - 1) % N;
            int right = (i + 1)     % N;
            Vec combined = concat(concat(h[i], h[left]), h[right]);
            out[i] = linear(W_[i], b_[i], combined, dim_ * 3, dim_);
            // ring は正規化なし（絶対値を保持）
        }
        return out;
    }
};

} // namespace myln
