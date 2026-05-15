#pragma once
#include "config.h"
#include "router.h"
#include "ring_attn.h"
#include "center_line.h"
#include "head.h"
#include <array>
#include <memory>
#include <stdexcept>

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
    Vec forward(const Vec& input) {
        if (!router_ || (int)input.size() != last_in_dim_) {
            last_in_dim_ = (int)input.size();
            router_ = std::make_unique<Router>(last_in_dim_, cfg_.dim);
        }
        auto routed   = router_->forward(input);
        std::array<Vec, 4> head_out;
        for (int i = 0; i < 4; ++i)
            head_out[i] = heads_[i]->forward(routed[i], cfg_.dim);
        auto ring_out = ring_.forward(head_out);
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
