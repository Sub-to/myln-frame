#pragma once
#include "frame.h"
#include <algorithm>
#include <stdexcept>

// ── CascadeFrame ───────────────────────────────────────────
//
// 2段カスケード: リレー（軽い）→ 確信度判定 → フル（重い）
//
//   入力
//    ↓
//  [RELAY]  ← SS + 2頭のみ active（proc + file）
//    ↓
//  確信度 >= threshold?
//    ├─ Yes（明確な入力）→ そのまま出力   ← 高速パス
//    └─ No（曖昧な入力） → [FULL] 4頭フル ← 精密パス
//
// ── Policy ────────────────────────────────────────────────
//
//  Exact（既定）     常にフルの結果を返す。リレーは評価しない。
//                    出力は MylnFrame(T) + tune_security と完全に同一。
//  Heuristic         上の図のとおり。リレーは proc/file しか見ない近似なので、
//                    早期終了の判定はフルと食い違うことがある（下記）。
//
// なぜ「正しい早期終了」を作らなかったか（実測）:
//   チューニング済みフルは全特徴量について単調なので、未観測の特徴量(net/cpu/mem)
//   を 0 にしたリレーは「フルのクラスの下界」になる（同サイズなら 20 万入力で違反 0）。
//   下界が最大クラスなら早期終了しても厳密に正しいが、proc=file=1.0 でも下界は
//   MEDIUM までしか届かない（CRITICAL には net/cpu/mem も要る）ので、厳密に正しい
//   早期終了は実質 0% になる。しかも下界がフルと一致するにはフルと同じ dim が必要で
//   （attention が 1/sqrt(dim) でスケールする）、そうするとリレーもフルと同じコスト。
//   チューニング済み T は ~1.6µs、SS は ~0.6µs なので、元々の「100µs 節約」という
//   動機も無くなった。よって既定を Exact にしている。
//
// 内部の Frame が作業バッファを使うので、複数スレッドから同時に呼ばないこと。

namespace myln {

class CascadeFrame {
public:
    enum class Policy { Exact, Heuristic };

private:
    Policy policy_ = Policy::Exact;
    Frame relay_;       // 軽い: SS / 2頭(proc+file)
    Frame full_;        // 重い: T  / 4頭フル
    float threshold_;   // 確信度の閾値（0.0〜1.0）
    long relay_hits_ = 0;  // リレーで終わった回数
    long full_hits_  = 0;  // フルまで回った回数

public:
    explicit CascadeFrame(float threshold = 0.80f, Policy policy = Policy::Exact)
        : policy_(policy)
        , relay_(SS, 5)   // 5クラス固定（セキュリティ用）
        , full_ (T,  5)
        , threshold_(threshold)
    {
        set_threshold(threshold);
    }

    // ── チューニング用アクセス ──────────────────────────────
    Frame& relay() { return relay_; }
    Frame& full()  { return full_;  }
    void   set_threshold(float t) {
        if (!(t >= 0.f && t <= 1.f))
            throw std::invalid_argument("CascadeFrame: threshold must be in [0, 1]");
        threshold_ = t;
    }
    float  threshold() const { return threshold_; }
    void   set_policy(Policy p) { policy_ = p; }
    Policy policy() const { return policy_; }

    // ── 推論 ───────────────────────────────────────────────
    // 戻り値: {probs, used_relay}
    struct Result {
        Vec  probs;
        bool used_relay;
    };

    // 内部の Frame が作業バッファを使うので、複数スレッドから同時に呼ばないこと。
    Result run(const Vec& input) {
        Result r{Vec(5), false};
        r.used_relay = run_into(input.data(), (int)input.size(), r.probs.data());
        return r;
    }

    // ヒープ確保なし版。probs は 5 個。戻り値: リレーで終わったか
    // （Exact では常に false）。
    bool run_into(const float* features, int n_in, float* probs) {
        if (policy_ == Policy::Exact) {
            ++full_hits_;
            full_.forward_into(features, n_in, probs);
            return false;
        }

        // 1. リレー（SS・2頭）で高速判定（Heuristic）
        relay_.forward_into(features, n_in, probs);
        float conf = *std::max_element(probs, probs + 5);

        if (conf >= threshold_) {
            ++relay_hits_;
            return true;
        }

        // 2. 確信が低い → フル（T・4頭）で精密判定
        ++full_hits_;
        full_.forward_into(features, n_in, probs);
        return false;
    }

    // monitor.py 互換: Vec だけ返すシンプル版
    Vec forward(const Vec& input) {
        return run(input).probs;
    }

    // ── 統計 ───────────────────────────────────────────────
    long  relay_hits()  const { return relay_hits_;  }
    long  full_hits()   const { return full_hits_;   }
    float relay_rate()  const {
        long total = relay_hits_ + full_hits_;
        return total ? (float)relay_hits_ / total : 0.f;
    }
    void  reset_stats() { relay_hits_ = full_hits_ = 0; }
};

} // namespace myln
