#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

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

} // namespace myln
