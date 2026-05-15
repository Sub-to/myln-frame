#pragma once
#include "config.h"
#include "router.h"
#include "ring_attn.h"
#include "center_line.h"
#include "head.h"
#include <array>
#include <memory>
#include <stdexcept>
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
class Frame {
    FrameConfig cfg_;
    int         n_classes_;
    int         last_in_dim_ = 0;

    std::unique_ptr<Router> router_;
    RingAttention           ring_;
    CenterLine              center_;
    std::array<HeadPtr, 4>  heads_;

public:
    Frame(const FrameConfig& cfg, int n_classes)
        : cfg_(cfg)
        , n_classes_(n_classes)
        , ring_(cfg.dim)
        , center_(cfg.dim, n_classes)
    {
        for (int i = 0; i < 4; ++i)
            heads_[i] = std::make_unique<DefaultHead>(
                cfg.dim, "slot-" + std::to_string(i), (unsigned)i);
    }

    // ── コンポーネントアクセス（チューニング用）──────────────
    // チューニング前に必ず呼ぶこと（ルーターを明示的に初期化）
    void init_router(int in_dim) {
        last_in_dim_ = in_dim;
        router_ = std::make_unique<Router>(in_dim, cfg_.dim);
    }

    Router&       router() { return *router_; }
    RingAttention& ring()  { return ring_;    }
    CenterLine&   center() { return center_;  }

    // ── ヘッド差し替え ──────────────────────────────────────
    void set_head(int slot, HeadPtr head) {
        if (slot < 0 || slot > 3) throw std::out_of_range("slot must be 0-3");
        heads_[slot] = std::move(head);
    }

    // ── 推論 ───────────────────────────────────────────────
    //
    // parallel=false（デフォルト）: 直列・超軽量ヘッド向け
    // parallel=true:               並列・将来の分散/重量ヘッド向け
    //
    // 将来の分散構成では heads_[i]->forward() を
    // ソケット/gRPC呼び出しに差し替えるだけでOK。
    // parallel=true の場合は4頭が同時にネットワーク越しに動く。
    Vec forward(const Vec& input, bool parallel = false) {
        if (!router_ || (int)input.size() != last_in_dim_) {
            last_in_dim_ = (int)input.size();
            router_ = std::make_unique<Router>(last_in_dim_, cfg_.dim);
        }

        // 1. APEX: 入力を4スロットに分配
        auto routed = router_->forward(input);

        // 2. ヘッド実行（軽い頭=直列、重い頭/分散=並列）
        std::array<Vec, 4> head_out;
        if (parallel) {
            // ── 並列モード: 分散・重量ヘッド向け ──────────────
            // heads_[i]->forward() がネットワーク通信になっても同じコード
            std::array<std::future<Vec>, 4> futures;
            for (int i = 0; i < 4; ++i)
                futures[i] = std::async(std::launch::async,
                    [&, i]{ return heads_[i]->forward(routed[i], cfg_.dim); });
            for (int i = 0; i < 4; ++i)
                head_out[i] = futures[i].get();
        } else {
            // ── 直列モード: SS/T の軽量ヘッドはこちらが速い ──
            for (int i = 0; i < 4; ++i)
                head_out[i] = heads_[i]->forward(routed[i], cfg_.dim);
        }

        // 3. Ring sync（分散時は各ノードがここでネットワーク越しに同期）
        auto ring_out = ring_.forward(head_out);

        // 4. CENTER LINE（分散時は集約ノードで実行）
        return center_.forward(ring_out);
    }

    const char* tag()       const { return cfg_.tag;      }
    int         dim()       const { return cfg_.dim;       }
    int         n_classes() const { return n_classes_;     }
};

// ── ファクトリ ─────────────────────────────────────────────
inline Frame make_ss(int n_classes) { return Frame(SS, n_classes); }
inline Frame make_t (int n_classes) { return Frame(T,  n_classes); }
inline Frame make_s (int n_classes) { return Frame(S,  n_classes); }

} // namespace myln
