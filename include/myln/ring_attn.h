#pragma once
#include "math_ops.h"
#include <array>
#include <stdexcept>

namespace myln {

// ── RING ATTENTION ─────────────────────────────────────────
// H0 ↔ H1 ↔ H2 ↔ H3 ↔ H0
// set_near_identity() でチューニング用途の近似恒等写像を設定できる。
//
// 内部では W_i [dim × 3dim] を (self | left | right) の3ブロックに分けて持つ。
// concat(h[i], h[left], h[right]) を作らずにそのまま積和できる。
class RingAttention {
    static constexpr int N = 4;
    int dim_;
    // blk_[i][0]=self, [1]=left, [2]=right
    std::array<std::array<PackedMat, 3>, N> blk_;
    std::array<Vec, N> b_;

    void set_blocks(int i, const Mat& W) {
        // W: [dim × dim*3] row-major → 3 つの [dim × dim] に分割
        for (int part = 0; part < 3; ++part) {
            Mat m((size_t)dim_ * dim_);
            for (int r = 0; r < dim_; ++r)
                for (int c = 0; c < dim_; ++c)
                    m[(size_t)r * dim_ + c] = W[(size_t)r * dim_ * 3 + part * dim_ + c];
            blk_[i][part].assign(m, dim_, dim_);
        }
    }

public:
    explicit RingAttention(int dim, unsigned seed = 20) : dim_(dim) {
        for (int i = 0; i < N; ++i) {
            set_blocks(i, rand_mat(dim, dim * 3, 0.1f, seed + i));
            b_[i] = zeros(dim);
        }
    }

    // チューニング用: self=w_self, 左右=w_nb の加重平均にする
    void set_near_identity(float w_self = 0.6f, float w_nb = 0.2f) {
        for (int i = 0; i < N; ++i) {
            Mat W((size_t)dim_ * dim_ * 3, 0.f);
            for (int d = 0; d < dim_; ++d) {
                W[(size_t)d * dim_ * 3 + d]            = w_self; // self
                W[(size_t)d * dim_ * 3 + dim_ + d]     = w_nb;   // left
                W[(size_t)d * dim_ * 3 + dim_ * 2 + d] = w_nb;   // right
            }
            set_blocks(i, W);
            b_[i].assign(dim_, 0.f);
        }
    }

    // 個別設定用
    void set_merge(int i, Mat W, Vec b) {
        if (i < 0 || i >= N)
            throw std::out_of_range("RingAttention::set_merge: index must be 0-3");
        if ((int)W.size() != dim_ * dim_ * 3 || (int)b.size() != dim_)
            throw std::invalid_argument("RingAttention::set_merge: W must be [dim x 3dim], b [dim]");
        set_blocks(i, W);
        b_[i] = std::move(b);
    }

    // h, out: [4 * dim]。ヒープ確保なし。
    void forward_into(const float* h, float* out) const {
        for (int i = 0; i < N; ++i) {
            const float* self  = h + (size_t)i * dim_;
            const float* left  = h + (size_t)((i + N - 1) % N) * dim_;
            const float* right = h + (size_t)((i + 1) % N) * dim_;
            float* o = out + (size_t)i * dim_;
            blk_[i][0].apply(self,  b_[i].data(), o);
            blk_[i][1].apply(left,  nullptr,      o, true);
            blk_[i][2].apply(right, nullptr,      o, true);
            // ring は正規化なし（絶対値を保持）
        }
    }

    std::array<Vec, N> forward(const std::array<Vec, N>& h) const {
        std::vector<float> in((size_t)N * dim_), o((size_t)N * dim_);
        for (int i = 0; i < N; ++i)
            std::copy(h[i].begin(), h[i].end(), in.begin() + (size_t)i * dim_);
        forward_into(in.data(), o.data());
        std::array<Vec, N> out;
        for (int i = 0; i < N; ++i)
            out[i].assign(o.begin() + (size_t)i * dim_, o.begin() + (size_t)(i + 1) * dim_);
        return out;
    }
};

} // namespace myln
