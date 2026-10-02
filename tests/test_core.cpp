// MYLN-FRAME コアのテスト（外部フレームワーク不要）。
// ctest から実行: すべて通れば exit 0。
#include "myln/frame.h"
#include "myln/cascade.h"
#include "tuner/security_tuner.h"
#include "tuner/earthquake_tuner.h"
#include "heads/passthrough_head.h"
#include "heads/zero_head.h"
#include "../bridge/myln_c_api.h"
#include "golden_security.inc"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using namespace myln;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

#define CHECK_THROWS(expr, ExType)                                               \
    do {                                                                         \
        ++g_checks; bool caught = false;                                         \
        try { expr; } catch (const ExType&) { caught = true; } catch (...) {}    \
        if (!caught) { ++g_fail; std::printf("  FAIL %s:%d  expected %s from %s\n", __FILE__, __LINE__, #ExType, #expr); } \
    } while (0)

// 処理系に依存しない決定的な一様乱数（std::uniform_real_distribution は実装依存）
struct Lcg {
    uint32_t s;
    float next() { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0f / 16777216.0f); }
};

static int argmax(const Vec& v) { return int(std::max_element(v.begin(), v.end()) - v.begin()); }
static float maxp(const Vec& v) { return *std::max_element(v.begin(), v.end()); }

// ── PackedMat: 疎・密どちらも素朴な行列積と一致する ──────────────
static void test_packed_mat() {
    std::printf("packed_mat\n");
    Lcg r{99};
    for (float density : {0.0f, 0.05f, 0.3f, 0.6f, 1.0f}) {
        const int rows = 17, cols = 23;   // 4 の倍数にしない（端数処理の確認）
        Mat W(rows * cols); Vec x(cols), b(rows);
        for (auto& v : W) v = (r.next() < density) ? r.next() * 2.f - 1.f : 0.f;
        for (auto& v : x) v = r.next() * 2.f - 1.f;
        for (auto& v : b) v = r.next();
        Vec ref = linear(W, b, x, cols, rows);

        PackedMat P(W, rows, cols);
        Vec y(rows);
        P.apply(x.data(), b.data(), y.data());
        float err = 0.f;
        for (int i = 0; i < rows; ++i) err = std::max(err, std::fabs(y[i] - ref[i]));
        CHECK(err < 1e-5f);

        // accumulate=true は y += W x （b なし）
        Vec acc(rows, 1.f);
        P.apply(x.data(), nullptr, acc.data(), true);
        Vec nob = linear(W, zeros(rows), x, cols, rows);
        err = 0.f;
        for (int i = 0; i < rows; ++i) err = std::max(err, std::fabs(acc[i] - (1.f + nob[i])));
        CHECK(err < 1e-5f);

        CHECK(P.sparse() == (density <= 0.3f));
    }
}

// ── 確率の健全性（和=1・有限・[0,1]）──────────────────────────
static void test_probabilities_sane() {
    std::printf("probabilities_sane\n");
    Lcg r{1};
    for (auto cfg : {SS, T, S}) {
        Frame f(cfg, 5); tune_security(f);
        Frame untuned(cfg, 5);   // ランダム重みでも壊れないこと
        for (int i = 0; i < 500; ++i) {
            Vec x(5); for (auto& v : x) v = r.next();
            for (Frame* fp : {&f, &untuned}) {
                Vec p = fp->forward(x);
                float sum = 0.f; bool ok = (p.size() == 5);
                for (float v : p) { ok = ok && std::isfinite(v) && v >= 0.f && v <= 1.f; sum += v; }
                CHECK(ok);
                CHECK(std::fabs(sum - 1.f) < 1e-4f);
            }
        }
    }
}

// ── 旧実装(v0.1.0)と同じ判定になること（回帰テスト）──────────────
static void test_golden_regression() {
    std::printf("golden_regression\n");
    struct Case { const char* name; FrameConfig cfg; const char* golden; };
    for (auto c : {Case{"T", T, kGoldenSecurityT}, Case{"SS", SS, kGoldenSecuritySS}}) {
        Frame f(c.cfg, 5); tune_security(f);
        Lcg r{12345};
        int checked = 0, mismatch = 0;
        for (size_t i = 0; i < std::strlen(c.golden); ++i) {
            Vec x(5); for (auto& v : x) v = r.next();
            if (c.golden[i] == '-') continue;
            ++checked;
            if (argmax(f.forward(x)) != c.golden[i] - '0') ++mismatch;
        }
        std::printf("  %s: %d samples, %d mismatches\n", c.name, checked, mismatch);
        CHECK(checked > 2900);
        CHECK(mismatch == 0);
    }
}

// ── 単調性: どの特徴量を上げても深刻度クラスは下がらない ─────────────
static void test_monotone() {
    std::printf("monotone\n");
    Frame f = make_t(5); tune_security(f);
    Lcg r{5};
    int viol = 0;
    for (int i = 0; i < 2000; ++i) {
        Vec x(5); for (auto& v : x) v = r.next();
        int base = argmax(f.forward(x));
        for (int k = 0; k < 5; ++k) {
            Vec y = x; y[k] = std::min(1.f, y[k] + 0.1f);
            if (argmax(f.forward(y)) < base) ++viol;
        }
    }
    CHECK(viol == 0);
}

// ── 代表シナリオ ─────────────────────────────────────────────
static void test_anchors() {
    std::printf("anchors\n");
    struct A { Vec x; int want; };
    const A anchors[] = {
        {{0.00f, 0.05f, 0.01f, 0.00f, 0.10f}, 0},   // idle
        {{0.10f, 0.30f, 0.40f, 0.05f, 0.25f}, 1},   // normal
        {{0.20f, 0.20f, 0.90f, 0.10f, 0.20f}, 2},   // port scan
        {{0.50f, 0.70f, 0.30f, 0.95f, 0.60f}, 3},   // mass file write
        {{0.90f, 0.95f, 0.80f, 0.99f, 0.85f}, 4},   // ransomware
    };
    for (auto cfg : {SS, T, S}) {
        Frame f(cfg, 5); tune_security(f);
        for (auto& a : anchors) CHECK(argmax(f.forward(a.x)) == a.want);
    }

    // logit_scale は判定クラスを変えず、確信度だけを上げる
    Frame base = make_t(5); tune_security(base);
    SecurityTuneParams sharp; sharp.logit_scale = 3.0f;
    Frame sh = make_t(5); tune_security(sh, sharp);
    for (auto& a : anchors) {
        Vec pb = base.forward(a.x), ps = sh.forward(a.x);
        CHECK(argmax(pb) == argmax(ps));
        CHECK(maxp(ps) > maxp(pb));
        CHECK(maxp(ps) > 0.65f);
    }
    Lcg r{8};
    int diff = 0;
    for (int i = 0; i < 2000; ++i) {
        Vec x(5); for (auto& v : x) v = r.next();
        Vec pb = base.forward(x), ps = sh.forward(x);
        diff += argmax(pb) != argmax(ps);
    }
    CHECK(diff == 0);
}

// ── 入力検証 ─────────────────────────────────────────────────
static void test_validation() {
    std::printf("validation\n");
    Frame f = make_t(5); tune_security(f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    CHECK_THROWS(f.forward({nan, 0, 0, 0, 0}), std::invalid_argument);
    CHECK_THROWS(f.forward({0, 0, inf, 0, 0}), std::invalid_argument);
    CHECK_THROWS(f.forward({}), std::invalid_argument);
    // チューニング済みなのに次元違いを渡したら、黙って再初期化せず失敗する
    CHECK_THROWS(f.forward({0.1f, 0.2f, 0.3f}), std::invalid_argument);
    // 失敗後もフレームは壊れていない
    CHECK(argmax(f.forward({0.9f, 0.95f, 0.8f, 0.99f, 0.85f})) == 4);

    // 桁あふれは非有限出力として検出（誤った判定を返さない）
    CHECK_THROWS(f.forward({1e38f, 1e38f, 1e38f, 1e38f, 1e38f}), std::runtime_error);

    // 不正な設定
    Frame g = make_t(5);
    CHECK_THROWS(g.router(), std::logic_error);              // init_router 前
    CHECK_THROWS(g.set_head(4, std::make_unique<ZeroHead>()), std::out_of_range);
    CHECK_THROWS(g.set_head(0, nullptr), std::invalid_argument);
    CHECK_THROWS(tune_security(g, [] { SecurityTuneParams p; p.in_dim = 3; return p; }()),
                 std::invalid_argument);
    Frame three = make_t(3);
    CHECK_THROWS(tune_security(three), std::invalid_argument);
    g.init_router(5);
    CHECK_THROWS(g.router().set_slot(0, Mat(3, 0.f), Vec(2, 0.f)), std::invalid_argument);
    CHECK_THROWS(g.center().set_logit_scale(0.f), std::invalid_argument);
    CHECK_THROWS(g.center().set_logit_scale(nan), std::invalid_argument);
    CHECK_THROWS(CascadeFrame(1.5f), std::invalid_argument);
    CHECK_THROWS(Frame(T, 0), std::invalid_argument);
}

// ── forward_into / parallel / forward が同じ結果 ─────────────────
static void test_paths_agree() {
    std::printf("paths_agree\n");
    Frame f = make_t(5); tune_security(f);
    Frame untuned = make_s(5);                // 密な DefaultHead（既定の forward_into）も通す
    Lcg r{3};
    for (int i = 0; i < 100; ++i) {
        Vec x(5); for (auto& v : x) v = r.next();
        for (Frame* fp : {&f, &untuned}) {
            Vec a = fp->forward(x);
            float b[5]; fp->forward_into(x.data(), 5, b);
            Vec c = fp->forward(x, /*parallel=*/true);
            for (int k = 0; k < 5; ++k) {
                CHECK(a[k] == b[k]);
                CHECK(a[k] == c[k]);
            }
        }
    }
}

// ── 部品の旧 API（Vec 版 forward）が forward_into と一致する ─────────
static void test_component_apis() {
    std::printf("component_apis\n");
    const int dim = 16;
    Lcg r{21};
    Router router(5, dim);
    Vec in(5); for (auto& v : in) v = r.next();
    auto routed = router.forward(in);
    std::vector<float> flat(4 * dim);
    router.forward_into(in.data(), flat.data());
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < dim; ++d) CHECK(routed[s][d] == flat[s * dim + d]);

    RingAttention ring(dim);
    ring.set_near_identity(0.6f, 0.2f);
    auto ro = ring.forward(routed);
    // near-identity: out_i = 0.6 h_i + 0.2 h_left + 0.2 h_right
    for (int i = 0; i < 4; ++i)
        for (int d = 0; d < dim; ++d) {
            float want = 0.6f * routed[i][d] + 0.2f * routed[(i + 3) % 4][d] + 0.2f * routed[(i + 1) % 4][d];
            CHECK(std::fabs(ro[i][d] - want) < 1e-5f);
        }

    // 個別設定した Ring（密な行列）も [dim x 3dim] の分割が正しい
    Mat W = rand_mat(dim, dim * 3, 0.3f, 4);
    ring.set_merge(2, W, Vec(dim, 0.5f));
    auto ro2 = ring.forward(routed);
    Vec combined = concat(concat(routed[2], routed[1]), routed[3]);
    Vec want2 = linear(W, Vec(dim, 0.5f), combined, dim * 3, dim);
    for (int d = 0; d < dim; ++d) CHECK(std::fabs(ro2[2][d] - want2[d]) < 1e-4f);

    // CenterLine: 前計算した qk が「q·(Wk h)」と一致する
    CenterLine center(dim, 5, /*normalize_agg=*/false);
    Vec probs = center.forward(ro2);
    float sum = 0.f; for (float v : probs) sum += v;
    CHECK(std::fabs(sum - 1.f) < 1e-5f);
}

// ── カスケード ──────────────────────────────────────────────
static void test_cascade() {
    std::printf("cascade\n");
    Frame full = make_t(5); tune_security(full);

    // 既定は Exact: 常にフルと完全に同じ出力（リレーは使わない）
    {
        CascadeFrame cas(0.80f); tune_cascade_security(cas, 0.80f);
        CHECK(cas.policy() == CascadeFrame::Policy::Exact);
        Lcg r{77};
        int differ = 0;
        for (int i = 0; i < 2000; ++i) {
            Vec x(5); for (auto& v : x) v = r.next();
            auto res = cas.run(x);
            Vec f = full.forward(x);
            if (res.used_relay) ++differ;
            for (int k = 0; k < 5; ++k) if (res.probs[k] != f[k]) ++differ;
        }
        CHECK(differ == 0);
        CHECK(cas.relay_hits() == 0 && cas.full_hits() == 2000);
    }

    // Heuristic: 明確なアイドル / 明確な脅威はリレーで終わり、判定はフルと同じ
    CascadeFrame cas(0.80f, CascadeFrame::Policy::Heuristic);
    tune_cascade_security(cas, 0.80f);
    auto idle = cas.run({0.00f, 0.05f, 0.01f, 0.00f, 0.10f});
    CHECK(idle.used_relay);
    CHECK(argmax(idle.probs) == 0);
    auto ransom = cas.run({0.90f, 0.95f, 0.80f, 0.99f, 0.85f});
    CHECK(ransom.used_relay);
    CHECK(argmax(ransom.probs) == 4);

    // 曖昧な入力はフルまで回り、フルと全く同じ出力になる
    Vec amb = {0.20f, 0.20f, 0.90f, 0.10f, 0.20f};
    auto r = cas.run(amb);
    CHECK(!r.used_relay);
    Vec f = full.forward(amb);
    for (int k = 0; k < 5; ++k) CHECK(r.probs[k] == f[k]);

    // 統計
    cas.reset_stats();
    cas.run({0.f, 0.f, 0.f, 0.f, 0.f});
    cas.run(amb);
    CHECK(cas.relay_hits() == 1 && cas.full_hits() == 1);
    CHECK(std::fabs(cas.relay_rate() - 0.5f) < 1e-6f);

    // 閾値 1.0 は常にフル（確率が丸めで 1.0 ちょうどにならない限り）
    cas.set_threshold(1.0f);
    cas.reset_stats();
    cas.run({0.9f, 0.95f, 0.8f, 0.99f, 0.85f});
    CHECK(cas.full_hits() == 1);

    // 設計上の前提の回帰テスト: 未観測の特徴量を 0 にしたリレー（同サイズ）は
    // フルのクラスの下界で、proc=file=1.0 でも CRITICAL には届かない。
    // → 「厳密に正しい早期終了」は成立しない（cascade.h のコメント参照）
    {
        Frame lo = make_t(5); tune_security(lo);
        Mat W(lo.dim() * 5, 0.f); Vec b(lo.dim(), 0.f);
        lo.router().set_slot(1, W, b);
        lo.router().set_slot(3, W, b);
        Lcg rr{4};
        int viol = 0;
        for (int i = 0; i < 20000; ++i) {
            Vec x(5); for (auto& v : x) v = rr.next();
            if (argmax(lo.forward(x)) > argmax(full.forward(x))) ++viol;
        }
        CHECK(viol == 0);
        CHECK(argmax(lo.forward({1.f, 0.f, 0.f, 1.f, 0.f})) < 4);
    }
}

// ── 地震チューナー ──────────────────────────────────────────
static void test_earthquake() {
    std::printf("earthquake\n");
    for (auto cfg : {SS, T, S}) {
        Frame f(cfg, 5); tune_earthquake(f);
        auto cls = [&](Vec x) { return argmax(f.forward(x)); };
        CHECK(cls({0, 0, 0, 0, 0}) == 0);                              // 無感 → SAFE
        CHECK(cls({1.f / 7, 0.2f, 0.0f, 0, 0}) == 1);                  // 震度1・深い → LOW
        CHECK(cls({1, 1, 1, 1, 1}) == 4);                              // 震度7+津波 → CRITICAL
        CHECK(cls({0, 0, 0, 1, 0}) >= 3);                              // 津波のみ → HIGH 以上
        CHECK(cls({5.f / 7, 6.5f / 9, 0.99f, 0, 0.3f}) == 4);          // 震度5 → CRITICAL
        // 単調性: 震度を上げて下がらない
        int prev = 0;
        for (int s = 0; s <= 7; ++s) {
            int c = cls({s / 7.f, 0.5f, 0.5f, 0, 0});
            CHECK(c >= prev);
            prev = c;
        }
    }
    // 重みの二重掛けの再発防止: 震度1の地震が CRITICAL になってはいけない
    Frame f = make_ss(5); tune_earthquake(f);
    CHECK(argmax(f.forward({1.f / 7, 3.f / 9, 0.94f, 0, 0.1f})) < 4);
}

// ── C API ───────────────────────────────────────────────────
static void test_c_api() {
    std::printf("c_api\n");
    CHECK(std::strlen(myln_version()) > 0);

    // 異常系: クラッシュせず NULL / -1 とメッセージが返る
    CHECK(myln_new("XL", 5) == nullptr);
    CHECK(std::strlen(myln_last_error()) > 0);
    CHECK(myln_new(nullptr, 5) == nullptr);
    CHECK(myln_new("T", 0) == nullptr);
    CHECK(myln_tune_security(nullptr, 5) == -1);
    int n = 0;
    CHECK(myln_infer(nullptr, nullptr, 0, &n) == nullptr);

    void* h = myln_new("T", 5);
    CHECK(h != nullptr);
    CHECK(myln_n_classes(h) == 5 && myln_dim(h) == 64 && std::strcmp(myln_tag(h), "MYLN-T") == 0);
    CHECK(myln_tune_security(h, 3) == -1);              // in_dim 不足
    CHECK(myln_tune_security(h, 5) == 0);

    const float ransom[5] = {0.9f, 0.95f, 0.8f, 0.99f, 0.85f};
    const float* p = myln_infer(h, ransom, 5, &n);
    CHECK(p != nullptr && n == 5);
    if (p) CHECK(std::max_element(p, p + 5) - p == 4);

    float out[5];
    CHECK(myln_infer_into(h, ransom, 5, out) == 0);
    CHECK(std::max_element(out, out + 5) - out == 4);

    const float bad[5] = {0.f, std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, 0.f};
    CHECK(myln_infer(h, bad, 5, &n) == nullptr);
    CHECK(myln_infer_into(h, bad, 5, out) == -1);
    CHECK(myln_infer_into(h, ransom, 4, out) == -1);    // 次元違い
    CHECK(myln_infer_into(h, ransom, 5, nullptr) == -1);
    CHECK(myln_infer_into(h, nullptr, 5, out) == -1);

    CHECK(myln_set_logit_scale(h, 3.f) == 0);
    CHECK(myln_set_logit_scale(h, -1.f) == -1);
    CHECK(myln_infer_into(h, ransom, 5, out) == 0);
    CHECK(out[4] > 0.65f);
    myln_free(h);
    myln_free(nullptr);   // no-op

    // 地震
    void* q = myln_new("SS", 5);
    CHECK(myln_tune_earthquake(q, 5) == 0);
    const float shindo1[5] = {1.f / 7, 3.f / 9, 0.94f, 0.f, 0.1f};
    CHECK(myln_infer_into(q, shindo1, 5, out) == 0);
    CHECK(std::max_element(out, out + 5) - out < 4);
    myln_free(q);

    // カスケード
    void* c = myln_cascade_new(0.8f);
    CHECK(c != nullptr);
    CHECK(myln_cascade_new(2.0f) == nullptr);
    CHECK(myln_cascade_tune_security(c, 5) == 0);
    int used = 1;
    CHECK(myln_cascade_infer_into(c, ransom, 5, out, &used) == 0 && used == 0);   // 既定は exact
    CHECK(myln_cascade_set_policy(c, 0) == 0);                                     // 近似に切り替え
    CHECK(myln_cascade_infer_into(c, ransom, 5, out, &used) == 0 && used == 1);
    const float* cp = myln_cascade_infer(c, ransom, 5, &n, &used);
    CHECK(cp != nullptr && n == 5 && used == 1);
    CHECK(myln_cascade_set_policy(nullptr, 1) == -1);
    CHECK(myln_cascade_infer_into(c, bad, 5, out, &used) == -1);
    CHECK(myln_cascade_relay_rate(c) > 0.f);
    myln_cascade_free(c);
}

int main() {
    test_packed_mat();
    test_probabilities_sane();
    test_golden_regression();
    test_monotone();
    test_anchors();
    test_validation();
    test_paths_agree();
    test_component_apis();
    test_cascade();
    test_earthquake();
    test_c_api();
    std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
