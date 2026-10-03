#pragma once
#include "../include/myln/frame.h"
#include "../include/myln/cascade.h"
#include "../heads/passthrough_head.h"
#include "../heads/zero_head.h"

// ── 依頼文の難易度判定チューナー ──────────────────────────────
//
// 「依頼文 → 5特徴量 → 難易度クラス」を判定するための手動チューニング。
// セキュリティ用チューナーとは独立しており、パラメータも別に調整できる。
//
// 入力ベクトル（5次元・各 0.0〜1.0）:
//   [0] tech       技術度     コード/ファイル名/技術用語の多さ
//   [1] length     長さ       依頼文の長さ
//   [2] steps      手数       「まず/次に/最後に」など手順の数
//   [3] scope      影響範囲   複数ファイル・プロジェクト全体・成果物（最重要）
//   [4] reasoning  推論度     原因調査・比較・設計など考える必要の度合い
//
// 出力クラス:
//   0=CHAT(雑談)  1=EASY(易)  2=MEDIUM(中)  3=HARD(難)  4=EXTREME(最難)
//
// スロット割り当て:
//   slot 0 → tech
//   slot 1 → steps
//   slot 2 → scope           ← 最重要
//   slot 3 → length + reasoning（それぞれ w_aux）
//
// 難易度スコア→クラスの閾値 (dim[0] の値で決定):
//   CHAT:    score < 0.5
//   EASY:    0.5 ≤ score < 1.5
//   MEDIUM:  1.5 ≤ score < 2.8
//   HARD:    2.8 ≤ score < 4.2
//   EXTREME: 4.2 ≤ score
//
// カスケード:
//   リレー（SS・tech + scope のみ）で確信度が閾値以上なら即決、
//   曖昧ならフル（T・4スロット全部）で再判定する。

namespace myln {

inline constexpr const char* DIFFICULTY_CLASS_NAMES[5]    = { "CHAT", "EASY", "MEDIUM", "HARD", "EXTREME" };
inline constexpr const char* DIFFICULTY_CLASS_NAMES_JA[5] = { "雑談", "易", "中", "難", "最難" };
inline constexpr const char* DIFFICULTY_FEATURE_NAMES[5]  = { "tech", "length", "steps", "scope", "reasoning" };

enum DifficultyFeature { DF_TECH = 0, DF_LENGTH = 1, DF_STEPS = 2, DF_SCOPE = 3, DF_REASONING = 4 };

// フル（4スロット）用パラメータ
struct DifficultyTuneParams {
    int in_dim = 5;

    // ── Router: score(dim[0]) に差し込む重み ──
    float w_tech  = 5.0f;
    float w_steps = 5.0f;
    float w_scope = 5.0f;   // 最重要
    float w_aux   = 2.5f;   // length / reasoning それぞれ

    // ── Ring Attention ──
    float ring_self = 0.6f;
    float ring_nb   = 0.2f;

    // ── Center Line ──
    float query_strength = 3.0f;
    // クラス i の得点 = slope_i * score + bias_i（交点が閾値 0.5/1.5/2.8/4.2）
    float slope[5] = { -1.0f, 0.5f, 1.5f, 2.5f, 3.5f };
    float bias [5] = {  9.25f, 8.50f, 7.00f, 4.20f, 0.00f };
};

// リレー（2スロット: tech + scope）用パラメータ
struct DifficultyRelayParams {
    int in_dim = 5;
    float w_tech  = 7.0f;
    float w_scope = 5.0f;
    float ring_self = 0.9f;   // ZeroHead による信号希釈を抑える
    float ring_nb   = 0.05f;
    float query_strength = 4.0f;
    // 傾きをフルの約2倍にして確信度を高める（曖昧な入力は低確信度→フルへ委譲）
    float slope[5] = { -2.0f, 1.0f, 3.0f, 5.0f, 7.0f };
    float bias [5] = { 18.5f, 17.0f, 14.0f, 8.4f, 0.0f };
};

namespace detail {
// 1スロット分: score(dim[0]) へ (特徴量, 重み) を差し込む
inline void difficulty_slot(Frame& f, int slot, int in_dim,
                            std::initializer_list<std::pair<int, float>> fw) {
    Mat W((size_t)f.dim() * in_dim, 0.f);
    Vec b(f.dim(), 0.f);
    for (auto kv : fw) W[0 * in_dim + kv.first] = kv.second;
    f.router().set_slot(slot, W, b);
}
inline void difficulty_center(Frame& f, float query_strength, const float* slope, const float* bias) {
    const int dim = f.dim();
    f.center().set_normalize_agg(false);
    f.center().set_key_identity();
    Vec q(dim, 0.f); q[0] = query_strength;
    f.center().set_query(q);
    Mat W(5 * (size_t)dim, 0.f); Vec b(5, 0.f);
    for (int c = 0; c < 5; ++c) { W[(size_t)c * dim + 0] = slope[c]; b[c] = bias[c]; }
    f.center().set_cls(W, b);
}
} // namespace detail

// ── フル（4スロット）のチューニング ──────────────────────────
inline void tune_difficulty(Frame& frame, const DifficultyTuneParams& p = {}) {
    frame.init_router(p.in_dim);
    frame.router().set_normalize(false);
    detail::difficulty_slot(frame, 0, p.in_dim, {{DF_TECH,  p.w_tech}});
    detail::difficulty_slot(frame, 1, p.in_dim, {{DF_STEPS, p.w_steps}});
    detail::difficulty_slot(frame, 2, p.in_dim, {{DF_SCOPE, p.w_scope}});
    detail::difficulty_slot(frame, 3, p.in_dim, {{DF_LENGTH, p.w_aux}, {DF_REASONING, p.w_aux}});

    frame.set_head(0, std::make_unique<PassthroughHead>("tech"));
    frame.set_head(1, std::make_unique<PassthroughHead>("steps"));
    frame.set_head(2, std::make_unique<PassthroughHead>("scope"));
    frame.set_head(3, std::make_unique<PassthroughHead>("length_reasoning"));

    frame.ring().set_near_identity(p.ring_self, p.ring_nb);
    detail::difficulty_center(frame, p.query_strength, p.slope, p.bias);
}

// ── リレー（tech + scope の2スロット）のチューニング ─────────
inline void tune_difficulty_relay(Frame& frame, const DifficultyRelayParams& p = {}) {
    frame.init_router(p.in_dim);
    frame.router().set_normalize(false);
    detail::difficulty_slot(frame, 0, p.in_dim, {{DF_TECH,  p.w_tech}});
    detail::difficulty_slot(frame, 1, p.in_dim, {});
    detail::difficulty_slot(frame, 2, p.in_dim, {{DF_SCOPE, p.w_scope}});
    detail::difficulty_slot(frame, 3, p.in_dim, {});

    frame.set_head(0, std::make_unique<PassthroughHead>("relay/tech"));
    frame.set_head(1, std::make_unique<ZeroHead>());
    frame.set_head(2, std::make_unique<PassthroughHead>("relay/scope"));
    frame.set_head(3, std::make_unique<ZeroHead>());

    frame.ring().set_near_identity(p.ring_self, p.ring_nb);
    detail::difficulty_center(frame, p.query_strength, p.slope, p.bias);
}

// ── CascadeFrame をまとめてチューニング ──────────────────────
inline void tune_cascade_difficulty(CascadeFrame& cascade, float threshold = 0.80f) {
    cascade.set_threshold(threshold);
    tune_difficulty_relay(cascade.relay());
    tune_difficulty      (cascade.full());
}

} // namespace myln
