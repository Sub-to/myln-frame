#include "myln_c_api.h"
#include "../include/myln/frame.h"
#include "../tuner/security_tuner.h"
#include <memory>
#include <string>
#include <vector>
#include <cstring>

// ── 内部コンテキスト ───────────────────────────────────────
struct MylnContext {
    std::unique_ptr<myln::Frame> frame;
    std::vector<float>           out_buf;   // 推論結果バッファ
};

static MylnContext* ctx(void* h) { return static_cast<MylnContext*>(h); }

// ── フレームのライフサイクル ──────────────────────────────
void* myln_new(const char* size, int n_classes) {
    try {
        auto c = std::make_unique<MylnContext>();
        std::string s(size);
        if      (s == "SS") c->frame = std::make_unique<myln::Frame>(myln::SS, n_classes);
        else if (s == "T")  c->frame = std::make_unique<myln::Frame>(myln::T,  n_classes);
        else if (s == "S")  c->frame = std::make_unique<myln::Frame>(myln::S,  n_classes);
        else return nullptr;
        c->out_buf.resize(n_classes, 0.f);
        return c.release();
    } catch (...) { return nullptr; }
}

void myln_free(void* frame) {
    delete ctx(frame);
}

// ── チューニング ──────────────────────────────────────────
void myln_tune_security(void* frame, int in_dim) {
    myln::SecurityTuneParams p;
    p.in_dim = in_dim;
    myln::tune_security(*ctx(frame)->frame, p);
}

// ── 推論 ─────────────────────────────────────────────────
const float* myln_infer(void* frame, const float* features, int n_in, int* out_n) {
    auto* c = ctx(frame);
    myln::Vec input(features, features + n_in);
    auto probs = c->frame->forward(input);
    int  n     = (int)probs.size();
    c->out_buf.resize(n);
    std::copy(probs.begin(), probs.end(), c->out_buf.begin());
    if (out_n) *out_n = n;
    return c->out_buf.data();
}

// ── メタ情報 ──────────────────────────────────────────────
const char* myln_tag      (void* frame) { return ctx(frame)->frame->tag(); }
int         myln_dim      (void* frame) { return ctx(frame)->frame->dim(); }
int         myln_n_classes(void* frame) { return ctx(frame)->frame->n_classes(); }
const char* myln_version  (void)        { return "0.1.0"; }
