#pragma once

// MYLN-FRAME frame size configurations
// SS = SuperSpeed (~1MB)  T = Tiny (~20MB)  S = Small (~100MB)

namespace myln {

struct FrameConfig {
    int         dim;         // embedding dimension
    int         n_slots;     // specialist head slots (always 4)
    int         inner_heads; // attention heads in center line
    const char* tag;
};

constexpr FrameConfig SS = {  16, 4, 2, "MYLN-SS" };
constexpr FrameConfig T  = {  64, 4, 4, "MYLN-T"  };
constexpr FrameConfig S  = { 128, 4, 8, "MYLN-S"  };

} // namespace myln
