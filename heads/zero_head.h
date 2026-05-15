#pragma once
#include "../include/myln/head.h"

namespace myln {

// ── ZeroHead ───────────────────────────────────────────────
// 常にゼロを出力する頭。
// CascadeFrame のリレー層で「使わないスロット」に置く。
// CENTER LINE の attention score がほぼ 0 になるので
// 実質的にそのスロットは無視される。
class ZeroHead : public Head {
public:
    std::string name() const override { return "zero"; }
    Vec forward(const Vec& /*x*/, int dim) override {
        return Vec(dim, 0.f);
    }
};

} // namespace myln
