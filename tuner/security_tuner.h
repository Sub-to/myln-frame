#pragma once
#include "../include/myln/frame.h"
#include "../include/myln/cascade.h"
#include "../heads/passthrough_head.h"
#include "../heads/zero_head.h"

// ── セキュリティ用手動チューニング ────────────────────────────
//
// 入力ベクトルの構造（5次元を想定）:
//   [0] proc_anomaly   0.0=正常  1.0=非常に怪しい
//   [1] cpu_spike      0.0=アイドル  1.0=フル稼働
//   [2] net_bytes      0.0=静か  1.0=大量通信
//   [3] file_change    0.0=変化なし  1.0=大量書き換え  ← 最重要
//   [4] mem_pressure   0.0=余裕  1.0=枯渇
//
// 出力クラス:
//   0=SAFE  1=LOW  2=MEDIUM  3=HIGH  4=CRITICAL
//
// スロット割り当て:
//   slot 0 → プロセス異常 (proc)
//   slot 1 → ネット通信  (net)
//   slot 2 → ファイル変化 (file) ← ランサムウェア指標として最重要
//   slot 3 → リソース    (cpu+mem の平均)
//
// 脅威スコア→クラスの閾値 (dim[0] の値で決定):
//   SAFE:     dim[0] < 0.5
//   LOW:      0.5 ≤ dim[0] < 1.5
//   MEDIUM:   1.5 ≤ dim[0] < 2.8
//   HIGH:     2.8 ≤ dim[0] < 4.2
//   CRITICAL: 4.2 ≤ dim[0]

namespace myln {

// tune_security() の設定パラメータ（調整しやすいよう一箇所にまとめる）
struct SecurityTuneParams {
    int in_dim = 5;   // 入力次元数（デフォルト5特徴量）

    // ── Router: dim[0] に差し込む重み ──
    float w_proc = 5.0f;  // slot0: proc の増幅率
    float w_net  = 5.0f;  // slot1: net の増幅率
    float w_file = 5.0f;  // slot2: file の増幅率（最重要）
    float w_res  = 2.5f;  // slot3: cpu/mem それぞれ

    // ── Ring Attention: 近似恒等写像の重み ──
    float ring_self = 0.6f;
    float ring_nb   = 0.2f;

    // ── Center Line: クエリの強さ ──
    float query_strength = 3.0f;

    // ── Center Line: W_cls の傾きとバイアス ──
    // 閾値: SAFE<0.5<LOW<1.5<MED<2.8<HIGH<4.2<CRIT
    // score_i = slope_i * dim[0] + bias_i
    // 傾きを変えることで各クラスの「引力範囲」を調整
    float slope_safe = -1.0f;  float bias_safe =  9.25f;
    float slope_low  =  0.5f;  float bias_low  =  8.50f;
    float slope_med  =  1.5f;  float bias_med  =  7.00f;
    float slope_high =  2.5f;  float bias_high =  4.20f;
    float slope_crit =  3.5f;  float bias_crit =  0.00f;
};

// ── メインのチューニング関数 ────────────────────────────────
inline void tune_security(Frame& frame, const SecurityTuneParams& p = {}) {
    int dim = frame.dim();

    // ━━ 1. Router: 明示初期化 → layer_norm を切り、特徴量を dim[0] に直接マップ ━━
    frame.init_router(p.in_dim);
    frame.router().set_normalize(false);

    auto make_slot = [&](std::initializer_list<std::pair<int,float>> feat_weights) {
        Mat W(dim * p.in_dim, 0.f);
        Vec b(dim, 0.f);
        for (auto [feat_idx, w] : feat_weights)
            W[0 * p.in_dim + feat_idx] = w;  // dim[0] にだけ重みを置く
        return std::make_pair(W, b);
    };

    auto [W0,b0] = make_slot({{0, p.w_proc}});                           // proc
    auto [W1,b1] = make_slot({{2, p.w_net}});                            // net
    auto [W2,b2] = make_slot({{3, p.w_file}});                           // file
    auto [W3,b3] = make_slot({{1, p.w_res}, {4, p.w_res}});              // cpu+mem

    frame.router().set_slot(0, W0, b0);
    frame.router().set_slot(1, W1, b1);
    frame.router().set_slot(2, W2, b2);
    frame.router().set_slot(3, W3, b3);

    // ━━ 2. Heads: パススルー（信号をそのまま通す）━━
    frame.set_head(0, std::make_unique<PassthroughHead>("proc"));
    frame.set_head(1, std::make_unique<PassthroughHead>("net"));
    frame.set_head(2, std::make_unique<PassthroughHead>("file"));
    frame.set_head(3, std::make_unique<PassthroughHead>("resource"));

    // ━━ 3. Ring Attention: 近似恒等写像 (self=0.6, 隣=0.2) ━━
    frame.ring().set_near_identity(p.ring_self, p.ring_nb);

    // ━━ 4. Center Line: 集約後の正規化を切り、手動閾値を設定 ━━
    frame.center().set_normalize_agg(false);
    frame.center().set_key_identity();

    // クエリ: dim[0] だけを見る
    Vec q(dim, 0.f);
    q[0] = p.query_strength;
    frame.center().set_query(q);

    // W_cls: [n_classes × dim]
    // 各クラスの score = slope * dim[0] + bias
    // 閾値の計算:
    //   SAFE/LOW  crossover at dim[0]=0.5
    //   LOW/MED   crossover at dim[0]=1.5
    //   MED/HIGH  crossover at dim[0]=2.8
    //   HIGH/CRIT crossover at dim[0]=4.2
    Mat W_cls(5 * dim, 0.f);
    Vec b_cls(5, 0.f);

    W_cls[0 * dim + 0] = p.slope_safe; b_cls[0] = p.bias_safe;
    W_cls[1 * dim + 0] = p.slope_low;  b_cls[1] = p.bias_low;
    W_cls[2 * dim + 0] = p.slope_med;  b_cls[2] = p.bias_med;
    W_cls[3 * dim + 0] = p.slope_high; b_cls[3] = p.bias_high;
    W_cls[4 * dim + 0] = p.slope_crit; b_cls[4] = p.bias_crit;

    frame.center().set_cls(W_cls, b_cls);
}

// ── リレー専用チューニング（2頭: proc + file のみ） ──────────
// 設計方針:
//   ① Ring を self=0.9 にして ZeroHead からの信号希釈を防ぐ
//   ② proc/file の重みを大きく取り「明確な脅威」を強調する
//   ③ W_cls の傾きをフル版の2倍にして確信度を高める
//      → 極端な入力(all-0, ransomware)で 80%+ の確信度を出す
//      → 中間的な入力では低確信度 → フルへ委譲
//
// 閾値は 0.5, 1.5, 2.8, 4.2 (フル版と同じ)
inline void tune_security_relay(Frame& frame, int in_dim = 5) {
    int dim = frame.dim();
    frame.init_router(in_dim);
    frame.router().set_normalize(false);

    // slot 0: proc（やや強め）
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      W[0*in_dim+0]=7.f; frame.router().set_slot(0,W,b); }
    // slot 1: ゼロ（使わない）
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      frame.router().set_slot(1,W,b); }
    // slot 2: file（最重要・積極的に強調）
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      W[0*in_dim+3]=12.f; frame.router().set_slot(2,W,b); }
    // slot 3: ゼロ（使わない）
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      frame.router().set_slot(3,W,b); }

    // 頭: proc・file だけ active、残りは ZeroHead
    frame.set_head(0, std::make_unique<PassthroughHead>("relay/proc"));
    frame.set_head(1, std::make_unique<ZeroHead>());
    frame.set_head(2, std::make_unique<PassthroughHead>("relay/file"));
    frame.set_head(3, std::make_unique<ZeroHead>());

    // Ring: self 強め（0.9）→ ZeroHead による信号希釈を最小化
    frame.ring().set_near_identity(0.9f, 0.05f);

    // CENTER LINE: 正規化なし、dim[0] を強めに見る
    frame.center().set_normalize_agg(false);
    frame.center().set_key_identity();
    Vec q(dim, 0.f); q[0] = 4.f;  // フルより強いクエリ
    frame.center().set_query(q);

    // W_cls: 傾きをフル版の2倍 → 確信度が高くなる
    // 閾値は同じ (0.5/1.5/2.8/4.2) だが急峻なので winner が支配的になる
    // SAFE/LOW  x=0.5:  -2*0.5+18.5 = 17.5  vs  1*0.5+17.0=17.5 ✓
    // LOW/MED   x=1.5:   1*1.5+17.0 = 18.5  vs  3*1.5+14.0=18.5 ✓
    // MED/HIGH  x=2.8:   3*2.8+14.0 = 22.4  vs  5*2.8+8.4 =22.4 ✓
    // HIGH/CRIT x=4.2:   5*4.2+8.4  = 29.4  vs  7*4.2+0   =29.4 ✓
    Mat W_cls(5*dim,0.f); Vec b_cls(5,0.f);
    W_cls[0*dim+0]=-2.f; b_cls[0]=18.5f;  // SAFE
    W_cls[1*dim+0]= 1.f; b_cls[1]=17.0f;  // LOW
    W_cls[2*dim+0]= 3.f; b_cls[2]=14.0f;  // MEDIUM
    W_cls[3*dim+0]= 5.f; b_cls[3]= 8.4f;  // HIGH
    W_cls[4*dim+0]= 7.f; b_cls[4]= 0.0f;  // CRITICAL
    frame.center().set_cls(W_cls, b_cls);
}

// ── CascadeFrame をまとめてチューニング ──────────────────────
inline void tune_cascade_security(CascadeFrame& cascade, float threshold = 0.80f) {
    cascade.set_threshold(threshold);
    tune_security_relay(cascade.relay());      // リレー: SS 2頭
    tune_security      (cascade.full());       // フル:   T  4頭
}

} // namespace myln
