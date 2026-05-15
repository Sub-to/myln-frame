#include "myln_c_api.h"
#include "../include/myln/frame.h"
#include "../include/myln/cascade.h"
#include "../tuner/security_tuner.h"
#include "../tuner/earthquake_tuner.h"
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

// ── 地震チューニング ──────────────────────────────────────────
void myln_tune_earthquake(void* frame, int in_dim) {
    myln::tune_earthquake(*ctx(frame)->frame, in_dim);
}

// ── カスケード ─────────────────────────────────────────────
struct CascadeCtx {
    myln::CascadeFrame cas;
    std::vector<float> out_buf;
    explicit CascadeCtx(float thr) : cas(thr) { out_buf.resize(5, 0.f); }
};
static CascadeCtx* cctx(void* h) { return static_cast<CascadeCtx*>(h); }

void* myln_cascade_new(float threshold) {
    try { return new CascadeCtx(threshold); }
    catch (...) { return nullptr; }
}
void myln_cascade_free(void* cas) { delete cctx(cas); }

void myln_cascade_tune_security(void* cas, int in_dim) {
    myln::tune_cascade_security(cctx(cas)->cas, cctx(cas)->cas.threshold());
    (void)in_dim;
}

const float* myln_cascade_infer(void* cas, const float* features,
                                int n_in, int* out_n, int* out_used_relay) {
    auto* c = cctx(cas);
    myln::Vec input(features, features + n_in);
    auto res = c->cas.run(input);
    int n = (int)res.probs.size();
    c->out_buf.resize(n);
    std::copy(res.probs.begin(), res.probs.end(), c->out_buf.begin());
    if (out_n)          *out_n = n;
    if (out_used_relay) *out_used_relay = res.used_relay ? 1 : 0;
    return c->out_buf.data();
}

float myln_cascade_relay_rate(void* cas) {
    return cctx(cas)->cas.relay_rate();
}
