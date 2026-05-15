#pragma once
#include "math_ops.h"
#include <string>
#include <memory>

namespace myln {

// ── Base interface for swappable heads ─────────────────────
// Input:  Vec of size `dim` (frame's embedding dim)
// Output: Vec of size `dim`
class Head {
public:
    virtual ~Head() = default;
    virtual std::string name() const = 0;
    virtual Vec forward(const Vec& x, int dim) = 0;
};

using HeadPtr = std::unique_ptr<Head>;

// ── Default head: 2-layer MLP ──────────────────────────────
// Used when no specialized .mhead is loaded.
// Replace this with your domain-specific head.
class DefaultHead : public Head {
    std::string label_;
    Mat W1_, W2_;
    Vec b1_, b2_;
public:
    DefaultHead(int dim, std::string label = "default", unsigned seed = 42)
        : label_(std::move(label))
    {
        int hidden = dim * 2;
        W1_ = rand_mat(hidden, dim,    0.1f, seed);
        b1_ = zeros(hidden);
        W2_ = rand_mat(dim,    hidden, 0.1f, seed + 1);
        b2_ = zeros(dim);
    }

    std::string name() const override { return label_; }

    Vec forward(const Vec& x, int dim) override {
        int hidden = dim * 2;
        Vec h = relu(linear(W1_, b1_, x, dim, hidden));
        Vec y = linear(W2_, b2_, h, hidden, dim);
        return layer_norm(y);
    }
};

} // namespace myln
