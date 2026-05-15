#pragma once
#include "math_ops.h"
#include <array>

namespace myln {

// ── CENTER LINE (集約) ─────────────────────────────────────
// The key line in the pyramid. Attends over all 4 head outputs,
// aggregates them, then classifies.
class CenterLine {
    int dim_, n_classes_;
    Vec  q_;               // global query
    std::array<Mat, 4> Wk_; // key projections per head
    Mat  W_cls_;
    Vec  b_cls_;

public:
    CenterLine(int dim, int n_classes, unsigned seed = 30)
        : dim_(dim), n_classes_(n_classes)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> d(0.f, 0.1f);

        q_.resize(dim);
        for (auto& v : q_) v = d(rng);

        for (int i = 0; i < 4; ++i)
            Wk_[i] = rand_mat(dim, dim, 0.1f, seed + i + 1);

        W_cls_ = rand_mat(n_classes, dim, 0.1f, seed + 10);
        b_cls_ = zeros(n_classes);
    }

    // Returns softmax class probabilities
    Vec forward(const std::array<Vec, 4>& heads) const {
        // Attention: score each head with global query
        float scale = 1.f / std::sqrt((float)dim_);
        Vec scores(4);
        for (int i = 0; i < 4; ++i) {
            Vec k = linear(Wk_[i], zeros(dim_), heads[i], dim_, dim_);
            scores[i] = dot(q_, k) * scale;
        }
        softmax_inplace(scores);

        // Weighted sum of all heads
        Vec agg = zeros(dim_);
        for (int i = 0; i < 4; ++i)
            for (int d = 0; d < dim_; ++d)
                agg[d] += scores[i] * heads[i][d];

        agg = layer_norm(agg);

        // Classify
        Vec logits = linear(W_cls_, b_cls_, agg, dim_, n_classes_);
        softmax_inplace(logits);
        return logits;
    }
};

} // namespace myln
