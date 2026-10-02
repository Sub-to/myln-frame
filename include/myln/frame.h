#pragma once
#include "config.h"
#include "router.h"
#include "ring_attn.h"
#include "center_line.h"
#include "head.h"
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <future>

namespace myln {

// ── MYLN-FRAME ─────────────────────────────────────────────
//
//           [ROUTER]          ← APEX: routes input to 4 slots
//         ↙  ↓  ↓  ↘
//       [A] [B] [C] [D]      ← swappable heads (.mhead)
//         ↕   ↕   ↕   ↕      ← ring attention (lateral sharing)
//           [CENTER LINE]     ← aggregates all → class output
//
// スレッド安全性: 1 つの Frame を複数スレッドから同時に forward() してはいけない
// （内部の作業バッファを使い回すため）。スレッドごとに Frame を持つこと。
class Frame {
    FrameConfig cfg_;
    int         n_classes_;
    int         last_in_dim_ = 0;

    std::unique_ptr<Router> router_;
    RingAttention           ring_;
    CenterLine              center_;
    std::array<HeadPtr, 4>  heads_;

    // 推論用の作業バッファ（forward のたびに確保しない）
    Vec routed_, head_out_, ring_out_, agg_;

    static void check_input(const float* x, int n) {
        if (n <= 0)
            throw std::invalid_argument("MYLN: input must have at least one feature");
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(x[i]))
                throw std::invalid_argument(
                    "MYLN: non-finite input (NaN/Inf) at feature " + std::to_string(i));
    }

public:
    Frame(const FrameConfig& cfg, int n_classes)
        : cfg_(cfg)
        , n_classes_(n_classes)
        , ring_(cfg.dim)
        , center_(cfg.dim, n_classes)
        , routed_  ((size_t)4 * cfg.dim)
        , head_out_((size_t)4 * cfg.dim)
        , ring_out_((size_t)4 * cfg.dim)
        , agg_     ((size_t)cfg.dim)
    {
        if (n_classes < 1)
            throw std::invalid_argument("MYLN: n_classes must be >= 1");
        for (int i = 0; i < 4; ++i)
            heads_[i] = std::make_unique<DefaultHead>(
                cfg.dim, "slot-" + std::to_string(i), (unsigned)i);
    }

    // ── コンポーネントアクセス（チューニング用）──────────────
    // チューニング前に必ず呼ぶこと（ルーターを明示的に初期化）
    void init_router(int in_dim) {
        if (in_dim <= 0)
            throw std::invalid_argument("MYLN: in_dim must be positive");
        last_in_dim_ = in_dim;
        router_ = std::make_unique<Router>(in_dim, cfg_.dim);
    }

    Router&       router() {
        if (!router_) throw std::logic_error("MYLN: call init_router(in_dim) before router()");
        return *router_;
    }
    RingAttention& ring()  { return ring_;    }
    CenterLine&   center() { return center_;  }

    // ── ヘッド差し替え ──────────────────────────────────────
    void set_head(int slot, HeadPtr head) {
        if (slot < 0 || slot > 3) throw std::out_of_range("slot must be 0-3");
        if (!head) throw std::invalid_argument("MYLN: head must not be null");
        heads_[slot] = std::move(head);
    }

    // ── 推論（ヒープ確保なし）─────────────────────────────────
    // features: n_in 個の float（全て有限値）  probs: n_classes() 個の出力先
    //
    // 例外:
    //   std::invalid_argument  NaN/Inf を含む / 入力次元がルーターと違う
    //   std::runtime_error     出力が非有限（重みの設定ミス・桁あふれ）
    //
    // parallel=false（デフォルト）: 直列・超軽量ヘッド向け
    // parallel=true:               並列・将来の分散/重量ヘッド向け
    //
    // 将来の分散構成では heads_[i]->forward() を
    // ソケット/gRPC呼び出しに差し替えるだけでOK。
    // parallel=true の場合は4頭が同時にネットワーク越しに動く。
    void forward_into(const float* features, int n_in, float* probs, bool parallel = false) {
        check_input(features, n_in);

        // ルーターが無ければ（未チューニング）最初の入力次元で作る。
        // 一度決まった入力次元が変わったら、黙って再初期化せずエラーにする
        // （チューニング済みの重みが捨てられて誤判定になるのを防ぐ）。
        if (!router_) {
            last_in_dim_ = n_in;
            router_ = std::make_unique<Router>(n_in, cfg_.dim);
        } else if (n_in != last_in_dim_) {
            throw std::invalid_argument(
                "MYLN: expected " + std::to_string(last_in_dim_) +
                " features, got " + std::to_string(n_in));
        }

        const int dim = cfg_.dim;

        // 1. APEX: 入力を4スロットに分配
        router_->forward_into(features, routed_.data());

        // 2. ヘッド実行（軽い頭=直列、重い頭/分散=並列）
        if (parallel) {
            // ── 並列モード: 分散・重量ヘッド向け ──────────────
            // heads_[i]->forward() がネットワーク通信になっても同じコード
            std::array<std::future<void>, 4> futures;
            for (int i = 0; i < 4; ++i)
                futures[i] = std::async(std::launch::async, [this, i, dim] {
                    heads_[i]->forward_into(routed_.data() + (size_t)i * dim,
                                            head_out_.data() + (size_t)i * dim, dim);
                });
            for (auto& f : futures) f.get();
        } else {
            // ── 直列モード: SS/T の軽量ヘッドはこちらが速い ──
            for (int i = 0; i < 4; ++i)
                heads_[i]->forward_into(routed_.data() + (size_t)i * dim,
                                        head_out_.data() + (size_t)i * dim, dim);
        }

        // 3. Ring sync（分散時は各ノードがここでネットワーク越しに同期）
        ring_.forward_into(head_out_.data(), ring_out_.data());

        // 4. CENTER LINE（分散時は集約ノードで実行）
        center_.forward_into(ring_out_.data(), agg_.data(), probs);

        for (int c = 0; c < n_classes_; ++c)
            if (!std::isfinite(probs[c]))
                throw std::runtime_error("MYLN: non-finite output (check weights / input scale)");
    }

    Vec forward(const Vec& input, bool parallel = false) {
        Vec probs((size_t)n_classes_);
        forward_into(input.data(), (int)input.size(), probs.data(), parallel);
        return probs;
    }

    const char* tag()       const { return cfg_.tag;      }
    int         dim()       const { return cfg_.dim;       }
    int         n_classes() const { return n_classes_;     }
    int         in_dim()    const { return last_in_dim_;   }
};

// ── ファクトリ ─────────────────────────────────────────────
inline Frame make_ss(int n_classes) { return Frame(SS, n_classes); }
inline Frame make_t (int n_classes) { return Frame(T,  n_classes); }
inline Frame make_s (int n_classes) { return Frame(S,  n_classes); }

} // namespace myln
