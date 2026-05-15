#pragma once
#include "../include/myln/frame.h"
#include "../heads/earthquake_head.h"
#include "../heads/zero_head.h"

// ── 地震監視チューナー ──────────────────────────────────────────
//
// 入力ベクトル (5次元):
//   [0] intensity   震度 / 7.0
//   [1] magnitude   M   / 9.0
//   [2] depth_inv   1 - 深さ/700
//   [3] tsunami     0 or 1
//   [4] freq        直近1時間の発生回数 / 10
//
// 出力クラス:
//   0=SAFE  1=LOW(震度1-2)  2=MEDIUM(震度3)  3=HIGH(震度4)  4=CRITICAL(震度5+)
//
// スロット:
//   slot 0 → intensity（主指標）
//   slot 1 → magnitude（補助）
//   slot 2 → tsunami（緊急）
//   slot 3 → depth + freq（補正）

namespace myln {

inline void tune_earthquake(Frame& frame, int in_dim = 5) {
    int dim = frame.dim();
    frame.init_router(in_dim);
    frame.router().set_normalize(false);

    // ── Router: 各スロットに特徴量を割り当て ──
    // slot 0: intensity → dim[0] に直接マップ
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      W[0*in_dim+0] = 7.0f;  // 震度×7 → CRITICALゾーン到達
      frame.router().set_slot(0, W, b); }

    // slot 1: magnitude
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      W[0*in_dim+1] = 2.5f;
      frame.router().set_slot(1, W, b); }

    // slot 2: tsunami（単独でHIGH以上になる重み）
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      W[0*in_dim+3] = 4.0f;
      frame.router().set_slot(2, W, b); }

    // slot 3: depth_inv + freq（補正）
    { Mat W(dim*in_dim,0.f); Vec b(dim,0.f);
      W[0*in_dim+2] = 1.0f;
      W[0*in_dim+4] = 0.5f;
      frame.router().set_slot(3, W, b); }

    // ── Heads: 全スロット EarthquakeHead（超軽量）──
    frame.set_head(0, std::make_unique<EarthquakeHead>("intensity"));
    frame.set_head(1, std::make_unique<EarthquakeHead>("magnitude"));
    frame.set_head(2, std::make_unique<EarthquakeHead>("tsunami"));
    frame.set_head(3, std::make_unique<EarthquakeHead>("depth_freq"));

    // ── Ring: self 強め（各スロットの信号を守る）──
    frame.ring().set_near_identity(0.7f, 0.1f);

    // ── CENTER LINE ──
    frame.center().set_normalize_agg(false);
    frame.center().set_key_identity();

    Vec q(dim, 0.f);
    q[0] = 3.0f;
    frame.center().set_query(q);

    // W_cls: 震度スケールに対応した閾値
    // SAFE:     dim[0] < 0.5  (なし〜震度1未満)
    // LOW:      0.5 ≤ dim[0] < 1.5  (震度1-2)
    // MEDIUM:   1.5 ≤ dim[0] < 2.8  (震度3)
    // HIGH:     2.8 ≤ dim[0] < 4.2  (震度4)  ← アラート閾値
    // CRITICAL: 4.2 ≤ dim[0]        (震度5+・津波)
    Mat W_cls(5*dim,0.f); Vec b_cls(5,0.f);
    W_cls[0*dim+0]=-1.0f; b_cls[0]= 9.25f;  // SAFE
    W_cls[1*dim+0]= 0.5f; b_cls[1]= 8.50f;  // LOW
    W_cls[2*dim+0]= 1.5f; b_cls[2]= 7.00f;  // MEDIUM
    W_cls[3*dim+0]= 2.5f; b_cls[3]= 4.20f;  // HIGH
    W_cls[4*dim+0]= 3.5f; b_cls[4]= 0.00f;  // CRITICAL
    frame.center().set_cls(W_cls, b_cls);
}

} // namespace myln
