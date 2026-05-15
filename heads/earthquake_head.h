#pragma once
#include "../include/myln/head.h"

// ── EarthquakeHead ─────────────────────────────────────────────
//
// 超軽量地震分類ヘッド (行列積なし・~0.1µs)
//
// 入力ベクトル (5次元):
//   [0] intensity   震度 / 7.0   (0=なし  1=震度7)
//   [1] magnitude   M   / 9.0   (0=なし  1=M9.0)
//   [2] depth_inv   1 - 深さ/700km  (浅いほど1=危険)
//   [3] tsunami     津波注意報フラグ (0 or 1)
//   [4] freq        直近1時間の発生回数 / 10
//
// 出力クラス (CENTER LINE で判定):
//   SAFE / LOW / MEDIUM / HIGH / CRITICAL
//
// 重み設計方針:
//   震度が最重要 → w_int 最大
//   津波は即CRITICAL要因 → w_tsun 大
//   マグニチュードは補助 → w_mag 中
//   浅さ・頻度は補正 → w_depth / w_freq 小

namespace myln {

class EarthquakeHead : public Head {
    float w_int_;    // 震度の重み
    float w_mag_;    // マグニチュードの重み
    float w_depth_;  // 浅さ（反転深度）の重み
    float w_tsun_;   // 津波フラグの重み
    float w_freq_;   // 発生頻度の重み
    std::string name_;

public:
    explicit EarthquakeHead(
        const std::string& slot = "quake",
        float w_int   = 7.0f,   // 震度7なら dim[0]=7.0（CRITICALゾーン）
        float w_mag   = 2.5f,   // M9なら +2.5 補正
        float w_depth = 1.0f,   // 浅い地震は +1.0
        float w_tsun  = 4.0f,   // 津波あれば +4.0 (即HIGH以上)
        float w_freq  = 0.5f    // 連続発生補正
    )
        : w_int_(w_int), w_mag_(w_mag), w_depth_(w_depth)
        , w_tsun_(w_tsun), w_freq_(w_freq)
        , name_("earthquake/" + slot)
    {}

    std::string name() const override { return name_; }

    // ── forward: 行列積なし・加重和のみ ─────────────────────
    // ~0.1µs / call (PassthroughHead より軽い)
    Vec forward(const Vec& x, int dim) override {
        Vec y(dim, 0.f);

        float intensity = x.size() > 0 ? x[0] : 0.f;
        float magnitude = x.size() > 1 ? x[1] : 0.f;
        float depth_inv = x.size() > 2 ? x[2] : 0.f;
        float tsunami   = x.size() > 3 ? x[3] : 0.f;
        float freq      = x.size() > 4 ? x[4] : 0.f;

        // dim[0] = 脅威スコア（CENTER LINE の閾値と対応）
        // 閾値: SAFE<0.5<LOW<1.5<MED<2.8<HIGH<4.2<CRIT
        y[0] = intensity * w_int_
             + magnitude * w_mag_
             + depth_inv * w_depth_
             + tsunami   * w_tsun_
             + freq      * w_freq_;

        // dim[1] に津波フラグをそのまま保持（Ring で隣に伝播）
        if (dim > 1) y[1] = tsunami;

        return y;  // layer_norm なし → 信号を保持
    }
};

} // namespace myln
