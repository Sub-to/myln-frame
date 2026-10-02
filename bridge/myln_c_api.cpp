#include "myln_c_api.h"
#include "../include/myln/frame.h"
#include "../include/myln/cascade.h"
#include "../tuner/security_tuner.h"
#include "../tuner/earthquake_tuner.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#ifndef MYLN_VERSION_STRING
#  define MYLN_VERSION_STRING "0.2.0"
#endif

// ── エラー状態（スレッドごと）────────────────────────────────
static thread_local std::string g_last_error;

static void set_error(const char* msg) { g_last_error = msg ? msg : "unknown error"; }

// C++ 例外を C の境界で必ず止める。失敗時は fail_value を返す。
template <class R, class F>
static R guarded(R fail_value, F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown C++ exception");
    }
    return fail_value;
}

// ── 内部コンテキスト ───────────────────────────────────────
struct MylnContext {
    std::unique_ptr<myln::Frame> frame;
    std::vector<float>           out_buf;   // 推論結果バッファ
};

static MylnContext* ctx(void* h) { return static_cast<MylnContext*>(h); }

static MylnContext* need_ctx(void* h) {
    if (!h) throw std::invalid_argument("null frame handle");
    return ctx(h);
}

const char* myln_last_error(void) { return g_last_error.c_str(); }

// ── フレームのライフサイクル ──────────────────────────────
void* myln_new(const char* size, int n_classes) {
    return guarded<void*>(nullptr, [&]() -> void* {
        if (!size) throw std::invalid_argument("size must not be null");
        auto c = std::make_unique<MylnContext>();
        std::string s(size);
        if      (s == "SS") c->frame = std::make_unique<myln::Frame>(myln::SS, n_classes);
        else if (s == "T")  c->frame = std::make_unique<myln::Frame>(myln::T,  n_classes);
        else if (s == "S")  c->frame = std::make_unique<myln::Frame>(myln::S,  n_classes);
        else throw std::invalid_argument("size must be \"SS\", \"T\" or \"S\"");
        c->out_buf.resize(n_classes, 0.f);
        return c.release();
    });
}

void myln_free(void* frame) {
    delete ctx(frame);
}

// ── チューニング ──────────────────────────────────────────
int myln_tune_security(void* frame, int in_dim) {
    return guarded<int>(-1, [&] {
        myln::SecurityTuneParams p;
        p.in_dim = in_dim;
        myln::tune_security(*need_ctx(frame)->frame, p);
        return 0;
    });
}

int myln_set_logit_scale(void* frame, float scale) {
    return guarded<int>(-1, [&] {
        need_ctx(frame)->frame->center().set_logit_scale(scale);
        return 0;
    });
}

int myln_tune_earthquake(void* frame, int in_dim) {
    return guarded<int>(-1, [&] {
        myln::tune_earthquake(*need_ctx(frame)->frame, in_dim);
        return 0;
    });
}

// ── 推論 ─────────────────────────────────────────────────
static void check_features(const float* features, int n_in) {
    if (!features) throw std::invalid_argument("features must not be null");
    if (n_in <= 0) throw std::invalid_argument("n_in must be positive");
}

const float* myln_infer(void* frame, const float* features, int n_in, int* out_n) {
    return guarded<const float*>(nullptr, [&]() -> const float* {
        auto* c = need_ctx(frame);
        check_features(features, n_in);
        c->out_buf.resize(c->frame->n_classes());
        c->frame->forward_into(features, n_in, c->out_buf.data());
        if (out_n) *out_n = c->frame->n_classes();
        return c->out_buf.data();
    });
}

int myln_infer_into(void* frame, const float* features, int n_in, float* out_probs) {
    return guarded<int>(-1, [&] {
        auto* c = need_ctx(frame);
        check_features(features, n_in);
        if (!out_probs) throw std::invalid_argument("out_probs must not be null");
        c->frame->forward_into(features, n_in, out_probs);
        return 0;
    });
}

// ── メタ情報 ──────────────────────────────────────────────
const char* myln_tag      (void* frame) { return frame ? ctx(frame)->frame->tag() : ""; }
int         myln_dim      (void* frame) { return frame ? ctx(frame)->frame->dim() : 0; }
int         myln_n_classes(void* frame) { return frame ? ctx(frame)->frame->n_classes() : 0; }
const char* myln_version  (void)        { return MYLN_VERSION_STRING; }

// ── カスケード ─────────────────────────────────────────────
struct CascadeCtx {
    myln::CascadeFrame cas;
    std::vector<float> out_buf;
    explicit CascadeCtx(float thr) : cas(thr) { out_buf.resize(5, 0.f); }
};
static CascadeCtx* cctx(void* h) { return static_cast<CascadeCtx*>(h); }
static CascadeCtx* need_cctx(void* h) {
    if (!h) throw std::invalid_argument("null cascade handle");
    return cctx(h);
}

void* myln_cascade_new(float threshold) {
    return guarded<void*>(nullptr, [&]() -> void* { return new CascadeCtx(threshold); });
}
void myln_cascade_free(void* cas) { delete cctx(cas); }

int myln_cascade_set_policy(void* cas, int exact) {
    return guarded<int>(-1, [&] {
        need_cctx(cas)->cas.set_policy(exact ? myln::CascadeFrame::Policy::Exact
                                             : myln::CascadeFrame::Policy::Heuristic);
        return 0;
    });
}

int myln_cascade_tune_security(void* cas, int in_dim) {
    return guarded<int>(-1, [&] {
        auto* c = need_cctx(cas);
        myln::tune_cascade_security(c->cas, c->cas.threshold(), in_dim);
        return 0;
    });
}

const float* myln_cascade_infer(void* cas, const float* features,
                                int n_in, int* out_n, int* out_used_relay) {
    return guarded<const float*>(nullptr, [&]() -> const float* {
        auto* c = need_cctx(cas);
        check_features(features, n_in);
        bool relay = c->cas.run_into(features, n_in, c->out_buf.data());
        if (out_n)          *out_n = 5;
        if (out_used_relay) *out_used_relay = relay ? 1 : 0;
        return c->out_buf.data();
    });
}

int myln_cascade_infer_into(void* cas, const float* features, int n_in,
                            float* out_probs, int* out_used_relay) {
    return guarded<int>(-1, [&] {
        auto* c = need_cctx(cas);
        check_features(features, n_in);
        if (!out_probs) throw std::invalid_argument("out_probs must not be null");
        bool relay = c->cas.run_into(features, n_in, out_probs);
        if (out_used_relay) *out_used_relay = relay ? 1 : 0;
        return 0;
    });
}

float myln_cascade_relay_rate(void* cas) {
    return cas ? cctx(cas)->cas.relay_rate() : 0.f;
}
