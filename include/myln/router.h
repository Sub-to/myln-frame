#pragma once
#include "math_ops.h"
#include <array>
#include <stdexcept>

namespace myln {

// ── ROUTER (APEX) ──────────────────────────────────────────
// Projects input → 4 slot vectors.
// normalize=false でチューニング用途に絶対値を保持する。
class Router {
    static constexpr int N = 4;
    int in_dim_, out_dim_;
    bool normalize_;
    std::array<PackedMat, N> W_;
    std::array<Vec, N>       b_;

public:
    Router(int in_dim, int out_dim, bool normalize = true, unsigned seed = 10)
        : in_dim_(in_dim), out_dim_(out_dim), normalize_(normalize)
    {
        for (int s = 0; s < N; ++s) {
            W_[s].assign(rand_mat(out_dim, in_dim, 0.1f, seed + s), out_dim, in_dim);
            b_[s] = zeros(out_dim);
        }
    }

    // 手動チューニング用: スロット s の重みをそのまま設定する
    void set_slot(int s, Mat W, Vec b) {
        if (s < 0 || s >= N)
            throw std::out_of_range("Router::set_slot: slot must be 0-3");
        if ((int)W.size() != out_dim_ * in_dim_ || (int)b.size() != out_dim_)
            throw std::invalid_argument("Router::set_slot: W must be [out_dim x in_dim], b [out_dim]");
        W_[s].assign(W, out_dim_, in_dim_);
        b_[s] = std::move(b);
    }

    void set_normalize(bool v) { normalize_ = v; }
    int  in_dim()  const { return in_dim_;  }
    int  out_dim() const { return out_dim_; }

    // out: [4 * out_dim] (slot-major)。ヒープ確保なし。
    void forward_into(const float* input, float* out) const {
        for (int s = 0; s < N; ++s) {
            float* o = out + (size_t)s * out_dim_;
            W_[s].apply(input, b_[s].data(), o);
            if (normalize_) layer_norm_raw(o, out_dim_);
        }
    }

    std::array<Vec, N> forward(const Vec& input) const {
        std::vector<float> buf((size_t)N * out_dim_);
        forward_into(input.data(), buf.data());
        std::array<Vec, N> out;
        for (int s = 0; s < N; ++s)
            out[s].assign(buf.begin() + (size_t)s * out_dim_,
                          buf.begin() + (size_t)(s + 1) * out_dim_);
        return out;
    }
};

} // namespace myln
