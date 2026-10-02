// MYLN-FRAME benchmark + behaviour report.
// 公開APIだけを使うので、旧バージョンのヘッダにも対してビルドできる（before/after 比較用）。
//
//   latency   : 1 推論あたりの平均 / p50 / p99 (ns)
//   monotone  : ある特徴量を増やしてもクラスが下がらないか（違反数）
//   anchors   : README の代表シナリオ 5 件の判定クラスと確信度
//   cascade   : exact はフルと一致 / heuristic はリレー通過率とフルとの一致率
#include "myln/frame.h"
#include "myln/cascade.h"
#include "tuner/security_tuner.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <algorithm>

using Clock = std::chrono::steady_clock;
using namespace myln;

static int argmax(const Vec& v) { return int(std::max_element(v.begin(), v.end()) - v.begin()); }
static float maxp(const Vec& v) { return *std::max_element(v.begin(), v.end()); }

struct Stat { double mean, p50, p99; };
template <class F>
static Stat time_ns(const std::vector<Vec>& xs, int reps, F&& f) {
    std::vector<double> t; t.reserve(xs.size() * reps);
    volatile float sink = 0;
    for (int r = 0; r < reps; ++r)
        for (auto& x : xs) {
            auto a = Clock::now();
            sink = sink + f(x);
            auto b = Clock::now();
            t.push_back(std::chrono::duration<double, std::nano>(b - a).count());
        }
    std::sort(t.begin(), t.end());
    double m = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
    return {m, t[t.size() / 2], t[size_t(t.size() * 0.99)]};
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 5000;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::vector<Vec> xs(n, Vec(5));
    for (auto& x : xs) for (auto& v : x) v = U(rng);

    Frame ss = make_ss(5), t = make_t(5), s = make_s(5);
    tune_security(ss); tune_security(t); tune_security(s);
    CascadeFrame cas(0.80f, CascadeFrame::Policy::Heuristic); tune_cascade_security(cas, 0.80f);
    CascadeFrame exact(0.80f);                                tune_cascade_security(exact, 0.80f);

    // warmup
    for (auto& x : xs) { ss.forward(x); t.forward(x); s.forward(x); cas.run(x); }

    auto L = [&](const char* name, Stat st) {
        std::printf("latency  %-9s mean %7.0f ns   p50 %7.0f   p99 %7.0f\n", name, st.mean, st.p50, st.p99);
    };
    L("SS",      time_ns(xs, 5, [&](const Vec& x){ return ss.forward(x)[0]; }));
    L("T",       time_ns(xs, 5, [&](const Vec& x){ return t.forward(x)[0]; }));
    L("S",       time_ns(xs, 5, [&](const Vec& x){ return s.forward(x)[0]; }));
    L("casc-exact", time_ns(xs, 5, [&](const Vec& x){ return exact.run(x).probs[0]; }));
    L("casc-heur",  time_ns(xs, 5, [&](const Vec& x){ return cas.run(x).probs[0]; }));

    // 単調性: 特徴量を +0.1 してもクラスが下がってはいけない
    long viol = 0, tot = 0;
    for (auto& x : xs) {
        int base = argmax(t.forward(x));
        for (int k = 0; k < 5; ++k) {
            Vec y = x; y[k] = std::min(1.f, y[k] + 0.1f);
            ++tot; if (argmax(t.forward(y)) < base) ++viol;
        }
    }
    std::printf("monotone T        violations %ld / %ld (%.2f%%)\n", viol, tot, 100.0 * viol / tot);

    // アンカー
    struct A { const char* name; Vec x; int want; };
    std::vector<A> anchors = {
        {"idle",       {0.00f,0.05f,0.01f,0.00f,0.10f}, 0},
        {"normal",     {0.10f,0.30f,0.40f,0.05f,0.25f}, 1},
        {"portscan",   {0.20f,0.20f,0.90f,0.10f,0.20f}, 2},
        {"massfile",   {0.50f,0.70f,0.30f,0.95f,0.60f}, 3},
        {"ransomware", {0.90f,0.95f,0.80f,0.99f,0.85f}, 4},
    };
    static const char* C[] = {"SAFE","LOW","MEDIUM","HIGH","CRITICAL"};
    int ok = 0; double conf = 0;
    for (auto& a : anchors) {
        auto p = t.forward(a.x); int b = argmax(p);
        std::printf("anchor   %-11s -> %-8s conf %.2f %s\n", a.name, C[b], maxp(p), b == a.want ? "" : "  (!= expected)");
        ok += b == a.want; conf += maxp(p);
    }
    std::printf("anchors  %d/5 ok, mean conf %.2f\n", ok, conf / 5);

    // カスケード
    //   exact     : 常にフルと同じ（食い違い 0 のはず）
    //   heuristic : リレーの早期終了。全体の一致率と、早期終了した入力だけの一致率
    {
        int ex_diff = 0;
        for (auto& x : xs) ex_diff += argmax(exact.run(x).probs) != argmax(t.forward(x));
        std::printf("cascade  exact      class mismatches vs full: %d / %d\n", ex_diff, n);

        cas.reset_stats();
        int agree = 0, exit_n = 0, exit_ok = 0;
        for (auto& x : xs) {
            auto r = cas.run(x);
            int want = argmax(t.forward(x));
            agree += argmax(r.probs) == want;
            if (r.used_relay) { ++exit_n; exit_ok += argmax(r.probs) == want; }
        }
        std::printf("cascade  heuristic  relay_rate %.1f%%   agreement-with-full %.2f%%   (early exits only: %.2f%%)\n",
                    100.0 * cas.relay_rate(), 100.0 * agree / n, exit_n ? 100.0 * exit_ok / exit_n : 0.0);
    }
    return 0;
}
