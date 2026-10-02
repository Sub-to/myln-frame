#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstddef>

namespace myln {

using Vec = std::vector<float>;
using Mat = std::vector<float>; // row-major flat, stride = cols

// ── Basic ops ──────────────────────────────────────────────

inline float dot(const Vec& a, const Vec& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline void softmax_inplace(Vec& v) {
    float mx = *std::max_element(v.begin(), v.end());
    float sum = 0.f;
    for (auto& x : v) { x = std::exp(x - mx); sum += x; }
    for (auto& x : v) x /= sum;
}

// W: [out × in] row-major
inline Vec linear(const Mat& W, const Vec& b, const Vec& x, int in, int out) {
    Vec y(out, 0.f);
    for (int i = 0; i < out; ++i) {
        for (int j = 0; j < in; ++j)
            y[i] += W[i * in + j] * x[j];
        y[i] += b[i];
    }
    return y;
}

inline Vec relu(Vec v) {
    for (auto& x : v) x = std::max(0.f, x);
    return v;
}

inline Vec layer_norm(Vec v, float eps = 1e-5f) {
    float mean = 0.f;
    for (auto x : v) mean += x;
    mean /= (float)v.size();
    float var = 0.f;
    for (auto x : v) var += (x - mean) * (x - mean);
    var  /= (float)v.size();
    float inv = 1.f / std::sqrt(var + eps);
    for (auto& x : v) x = (x - mean) * inv;
    return v;
}

inline Vec zeros(int n) { return Vec(n, 0.f); }

inline Vec concat(const Vec& a, const Vec& b) {
    Vec c = a;
    c.insert(c.end(), b.begin(), b.end());
    return c;
}

// Random matrix [rows × cols]
inline Mat rand_mat(int rows, int cols, float scale = 0.1f, unsigned seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.f, scale);
    Mat m(rows * cols);
    for (auto& v : m) v = d(rng);
    return m;
}

// ── Allocation-free kernels (raw pointers) ─────────────────

// 4本のアキュムレータで依存鎖を切る。-ffast-math なしでも
// ループ展開 + SIMD 化しやすい。
inline float dot_raw(const float* a, const float* b, int n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

inline void softmax_raw(float* v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, v[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { v[i] = std::exp(v[i] - mx); sum += v[i]; }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

inline void layer_norm_raw(float* v, int n, float eps = 1e-5f) {
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += v[i];
    mean /= (float)n;
    float var = 0.f;
    for (int i = 0; i < n; ++i) var += (v[i] - mean) * (v[i] - mean);
    var /= (float)n;
    float inv = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) v[i] = (v[i] - mean) * inv;
}

// ── PackedMat: 疎なら CSR、密なら dense で持つ行列 ─────────────
//
// 手動チューニングした重みはほぼ「0 と少数の非ゼロ」(恒等写像・単一特徴量
// の取り出し)。ゼロとの積和を飛ばしても結果は同じなので、設定時に一度だけ
// 非ゼロ数を数えて、疎なら CSR に切り替える。ランダム初期化のままの層は
// dense のまま動くので、挙動は変わらない。
class PackedMat {
    int  rows_ = 0, cols_ = 0;
    bool sparse_ = false;
    Mat  dense_;                  // dense 時のみ
    std::vector<int> row_ptr_;    // sparse 時のみ (rows_ + 1)
    std::vector<int> col_idx_;
    Vec  val_;

public:
    PackedMat() = default;
    PackedMat(const Mat& W, int rows, int cols, float max_density = 0.35f) {
        assign(W, rows, cols, max_density);
    }

    void assign(const Mat& W, int rows, int cols, float max_density = 0.35f) {
        assert((int)W.size() == rows * cols);
        rows_ = rows; cols_ = cols;
        size_t nnz = 0;
        for (float v : W) nnz += (v != 0.f);
        sparse_ = (float)nnz <= max_density * (float)W.size();
        dense_.clear(); row_ptr_.clear(); col_idx_.clear(); val_.clear();
        if (!sparse_) { dense_ = W; return; }
        row_ptr_.reserve(rows + 1);
        col_idx_.reserve(nnz);
        val_.reserve(nnz);
        row_ptr_.push_back(0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float v = W[(size_t)i * cols + j];
                if (v != 0.f) { col_idx_.push_back(j); val_.push_back(v); }
            }
            row_ptr_.push_back((int)col_idx_.size());
        }
    }

    int  rows()   const { return rows_;   }
    int  cols()   const { return cols_;   }
    bool sparse() const { return sparse_; }

    // y[i] (+)= W[i,:]·x   （b は nullptr 可。accumulate=false なら上書き）
    void apply(const float* x, const float* b, float* y, bool accumulate = false) const {
        if (sparse_) {
            for (int i = 0; i < rows_; ++i) {
                float s = accumulate ? y[i] : 0.f;
                for (int k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
                    s += val_[k] * x[col_idx_[k]];
                y[i] = b ? s + b[i] : s;
            }
        } else {
            for (int i = 0; i < rows_; ++i) {
                float s = dot_raw(&dense_[(size_t)i * cols_], x, cols_);
                if (accumulate) s += y[i];
                y[i] = b ? s + b[i] : s;
            }
        }
    }
};

} // namespace myln
