#pragma once
#include "math_ops.h"
#include <array>

namespace myln {

// ── CENTER LINE (集約) ─────────────────────────────────────
// 4頭を注意機構で集約 → クラス分類。
// set_cls / set_query / set_key_identity でチューニング可能。
class CenterLine {
    int dim_, n_classes_;
    bool normalize_agg_;       // 集約後の layer_norm を切れる

    Vec              q_;       // グローバルクエリ
    std::array<Mat, 4> Wk_;   // キー射影
    Mat  W_cls_;
    Vec  b_cls_;

public:
    CenterLine(int dim, int n_classes, bool normalize_agg = true, unsigned seed = 30)
        : dim_(dim), n_classes_(n_classes), normalize_agg_(normalize_agg)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> d(0.f, 0.1f);
        q_.resize(dim); for (auto& v : q_) v = d(rng);
        for (int i = 0; i < 4; ++i)
            Wk_[i] = rand_mat(dim, dim, 0.1f, seed + i + 1);
        W_cls_ = rand_mat(n_classes, dim, 0.1f, seed + 10);
        b_cls_ = zeros(n_classes);
    }

    // ── チューニング用セッター ──────────────────────────────
    void set_query(Vec q)              { q_     = std::move(q); }
    void set_cls(Mat W, Vec b)         { W_cls_ = std::move(W); b_cls_ = std::move(b); }
    void set_normalize_agg(bool v)     { normalize_agg_ = v; }

    // キーを恒等写像に設定（Wk[i] = I）
    void set_key_identity() {
        for (int i = 0; i < 4; ++i) {
            Wk_[i].assign(dim_ * dim_, 0.f);
            for (int d = 0; d < dim_; ++d)
                Wk_[i][d * dim_ + d] = 1.f;
        }
    }

    // ── 推論 ───────────────────────────────────────────────
    Vec forward(const std::array<Vec, 4>& heads) const {
        float scale = 1.f / std::sqrt((float)dim_);
        Vec scores(4);
        for (int i = 0; i < 4; ++i) {
            Vec k = linear(Wk_[i], zeros(dim_), heads[i], dim_, dim_);
            scores[i] = dot(q_, k) * scale;
        }
        softmax_inplace(scores);

        Vec agg = zeros(dim_);
        for (int i = 0; i < 4; ++i)
            for (int d = 0; d < dim_; ++d)
                agg[d] += scores[i] * heads[i][d];

        if (normalize_agg_) agg = layer_norm(agg);

        Vec logits = linear(W_cls_, b_cls_, agg, dim_, n_classes_);
        softmax_inplace(logits);
        return logits;
    }
};

} // namespace myln
