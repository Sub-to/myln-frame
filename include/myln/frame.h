#pragma once
#include "config.h"
#include "router.h"
#include "ring_attn.h"
#include "center_line.h"
#include "head.h"
#include <array>
#include <memory>
#include <string>
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

    // Swap head at slot 0-3
    void set_head(int slot, HeadPtr head) {
        if (slot < 0 || slot > 3)
            throw std::out_of_range("slot must be 0-3");
        heads_[slot] = std::move(head);
    }

    // Run inference
    // input: your feature vector (any size; router adapts automatically)
    // returns: softmax probabilities over n_classes
    Vec forward(const Vec& input) {
        // Auto-init/reinit router when input dim changes
        if (!router_ || (int)input.size() != last_in_dim_) {
            last_in_dim_ = (int)input.size();
            router_ = std::make_unique<Router>(last_in_dim_, cfg_.dim);
        }

        // 1. APEX: route input → 4 slot vectors
        auto routed = router_->forward(input);

        // 2. Each head processes its slot
        std::array<Vec, 4> head_out;
        for (int i = 0; i < 4; ++i)
            head_out[i] = heads_[i]->forward(routed[i], cfg_.dim);

        // 3. Ring attention: heads share with neighbors
        auto ring_out = ring_.forward(head_out);

        // 4. CENTER LINE: aggregate → class probabilities
        return center_.forward(ring_out);
    }

    const char* tag()       const { return cfg_.tag; }
    int         dim()       const { return cfg_.dim; }
    int         n_classes() const { return n_classes_; }
};

// ── Factory helpers ────────────────────────────────────────
inline Frame make_ss(int n_classes) { return Frame(SS, n_classes); }
inline Frame make_t (int n_classes) { return Frame(T,  n_classes); }
inline Frame make_s (int n_classes) { return Frame(S,  n_classes); }

} // namespace myln
