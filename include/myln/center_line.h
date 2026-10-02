#pragma once
#include "math_ops.h"
#include <array>
#include <stdexcept>

namespace myln {

// ── CENTER LINE (集約) ─────────────────────────────────────
// 4頭を注意機構で集約 → クラス分類。
// set_cls / set_query / set_key_identity でチューニング可能。
//
// attention の score は q · (Wk_i h_i) = (Wk_iᵀ q) · h_i なので、
// 設定時に qk_i = Wk_iᵀ q を前計算しておけば、推論時に dim×dim の
// 行列積 4 回が dim 長の内積 4 回に減る（結果は代数的に同一）。
class CenterLine {
    int dim_, n_classes_;
    bool normalize_agg_;       // 集約後の layer_norm を切れる
    float logit_scale_ = 1.f;  // softmax 前に logits へ掛ける係数（確信度の鋭さ）

    Vec              q_;       // グローバルクエリ
    std::array<Mat, 4> Wk_;   // キー射影
    std::array<Vec, 4> qk_;   // 前計算: Wk_iᵀ q
    PackedMat W_cls_;
    Vec  b_cls_;

    void rebuild_qk() {
        for (int i = 0; i < 4; ++i) {
            qk_[i].assign(dim_, 0.f);
            for (int r = 0; r < dim_; ++r) {
                float qr = q_[r];
                if (qr == 0.f) continue;
                const float* row = &Wk_[i][(size_t)r * dim_];
                for (int c = 0; c < dim_; ++c) qk_[i][c] += qr * row[c];
            }
        }
    }

public:
    CenterLine(int dim, int n_classes, bool normalize_agg = true, unsigned seed = 30)
        : dim_(dim), n_classes_(n_classes), normalize_agg_(normalize_agg)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> d(0.f, 0.1f);
        q_.resize(dim); for (auto& v : q_) v = d(rng);
        for (int i = 0; i < 4; ++i)
            Wk_[i] = rand_mat(dim, dim, 0.1f, seed + i + 1);
        W_cls_.assign(rand_mat(n_classes, dim, 0.1f, seed + 10), n_classes, dim);
        b_cls_ = zeros(n_classes);
        rebuild_qk();
    }

    // ── チューニング用セッター ──────────────────────────────
    void set_query(Vec q) {
        if ((int)q.size() != dim_)
            throw std::invalid_argument("CenterLine::set_query: q must have size dim");
        q_ = std::move(q);
        rebuild_qk();
    }
    void set_cls(Mat W, Vec b) {
        if ((int)W.size() != n_classes_ * dim_ || (int)b.size() != n_classes_)
            throw std::invalid_argument("CenterLine::set_cls: W must be [n_classes x dim], b [n_classes]");
        W_cls_.assign(W, n_classes_, dim_);
        b_cls_ = std::move(b);
    }
    void set_normalize_agg(bool v)     { normalize_agg_ = v; }

    // logits に掛ける係数。argmax は変わらず、確率の鋭さ（確信度）だけが変わる。
    // 1.0 = 従来どおり。大きいほど max 確率が 1 に近づく。
    void  set_logit_scale(float s) {
        if (!(s > 0.f) || !std::isfinite(s))
            throw std::invalid_argument("CenterLine::set_logit_scale: must be finite and > 0");
        logit_scale_ = s;
    }
    float logit_scale() const { return logit_scale_; }

    // キーを恒等写像に設定（Wk[i] = I）
    void set_key_identity() {
        for (int i = 0; i < 4; ++i) {
            Wk_[i].assign((size_t)dim_ * dim_, 0.f);
            for (int d = 0; d < dim_; ++d)
                Wk_[i][(size_t)d * dim_ + d] = 1.f;
        }
        rebuild_qk();
    }

    // ── 推論 ───────────────────────────────────────────────
    // heads : [4 * dim]   agg : scratch [dim]   probs : [n_classes]
    // ヒープ確保なし。
    void forward_into(const float* heads, float* agg, float* probs) const {
        const float scale = 1.f / std::sqrt((float)dim_);
        float scores[4];
        for (int i = 0; i < 4; ++i)
            scores[i] = dot_raw(qk_[i].data(), heads + (size_t)i * dim_, dim_) * scale;
        softmax_raw(scores, 4);

        for (int d = 0; d < dim_; ++d) agg[d] = 0.f;
        for (int i = 0; i < 4; ++i) {
            const float* h = heads + (size_t)i * dim_;
            const float  s = scores[i];
            for (int d = 0; d < dim_; ++d) agg[d] += s * h[d];
        }

        if (normalize_agg_) layer_norm_raw(agg, dim_);

        W_cls_.apply(agg, b_cls_.data(), probs);
        if (logit_scale_ != 1.f)
            for (int c = 0; c < n_classes_; ++c) probs[c] *= logit_scale_;
        softmax_raw(probs, n_classes_);
    }

    Vec forward(const std::array<Vec, 4>& heads) const {
        Vec flat((size_t)4 * dim_), agg(dim_), probs(n_classes_);
        for (int i = 0; i < 4; ++i)
            std::copy(heads[i].begin(), heads[i].end(), flat.begin() + (size_t)i * dim_);
        forward_into(flat.data(), agg.data(), probs.data());
        return probs;
    }
};

} // namespace myln
