#pragma once
// ── 汎用チューニング（JSON 設定から Frame / CascadeFrame を設定する）──
//
// 仕様: docs/tuning-config.md  （スキーマ名 "myln-tune/1"）
//
//   tune_custom(frame, path_or_json)
//   tune_cascade_custom(cascade, path_or_json)
//
// path_or_json: 先頭の空白を除いて '{' で始まれば JSON 文字列、それ以外はファイルパス。
// 先に全項目を検証してから適用する（検証エラーなら frame は一切変更されない）。
// エラーは std::runtime_error（どの項目が悪いかを含むメッセージ）。

#include "frame.h"
#include "cascade.h"
#include "mini_json.h"
#include "../../heads/passthrough_head.h"
#include "../../heads/zero_head.h"
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace myln {
namespace cfg {

using json::Value;

inline std::string fmt_idx(const std::string& ctx, size_t i) { return ctx + "[" + std::to_string(i) + "]"; }

[[noreturn]] inline void bad(const std::string& ctx, const std::string& msg) {
    throw std::runtime_error(ctx + ": " + msg);
}

inline double as_num(const Value& v, const std::string& ctx) {
    if (!v.is_number() || !std::isfinite(v.n)) bad(ctx, "expected a finite number");
    return v.n;
}

inline int as_index(const Value& v, int lo, int hi_excl, const std::string& ctx) {
    double d = as_num(v, ctx);
    if (d != std::floor(d)) bad(ctx, "expected an integer index");
    if (d < lo || d >= hi_excl)
        bad(ctx, "index " + std::to_string((long long)d) + " out of range [" +
                 std::to_string(lo) + "," + std::to_string(hi_excl) + ")");
    return (int)d;
}

inline const std::vector<Value>& as_arr(const Value& v, const std::string& ctx) {
    if (!v.is_array()) bad(ctx, "expected an array");
    return v.arr();
}

inline const Value& as_obj(const Value& v, const std::string& ctx) {
    if (!v.is_object()) bad(ctx, "expected an object");
    return v;
}

// 未知のキーを弾く（typo 検出）
inline void check_keys(const Value& obj, std::initializer_list<const char*> allowed, const std::string& ctx) {
    for (auto& kv : obj.obj()) {
        bool ok = false;
        for (auto a : allowed) if (kv.first == a) { ok = true; break; }
        if (!ok) bad(ctx, "unknown key '" + kv.first + "'");
    }
}

// terms: [[row, col, value], ...] を行列(row-major, stride=cols)へ上書きする
inline void apply_terms(Mat& M, int rows, int cols, const Value& terms, const std::string& ctx) {
    const auto& arr = as_arr(terms, ctx);
    for (size_t k = 0; k < arr.size(); ++k) {
        std::string c = fmt_idx(ctx, k);
        const auto& t = as_arr(arr[k], c);
        if (t.size() != 3) bad(c, "expected [row, col, value]");
        int r  = as_index(t[0], 0, rows, c + "[0]");
        int cc = as_index(t[1], 0, cols, c + "[1]");
        M[(size_t)r * cols + cc] = (float)as_num(t[2], c + "[2]");
    }
}

// bias: [[index, value], ...] をベクトルへ上書きする
inline void apply_bias(Vec& b, int n, const Value& bias, const std::string& ctx) {
    const auto& arr = as_arr(bias, ctx);
    for (size_t k = 0; k < arr.size(); ++k) {
        std::string c = fmt_idx(ctx, k);
        const auto& t = as_arr(arr[k], c);
        if (t.size() != 2) bad(c, "expected [index, value]");
        b[as_index(t[0], 0, n, c + "[0]")] = (float)as_num(t[1], c + "[1]");
    }
}

// 密行列 "W": [[...], ...] （rows × cols）
inline void apply_dense(Mat& M, int rows, int cols, const Value& W, const std::string& ctx) {
    const auto& arr = as_arr(W, ctx);
    if ((int)arr.size() != rows) bad(ctx, "expected " + std::to_string(rows) + " rows");
    for (int r = 0; r < rows; ++r) {
        std::string c = fmt_idx(ctx, r);
        const auto& row = as_arr(arr[r], c);
        if ((int)row.size() != cols) bad(c, "expected " + std::to_string(cols) + " columns");
        for (int j = 0; j < cols; ++j) M[(size_t)r * cols + j] = (float)as_num(row[j], fmt_idx(c, j));
    }
}

struct HeadSpec { std::string type; std::string name; unsigned seed = 0; };

// ── 検証済みの設定（apply 前の中間表現）──────────────────────
struct FramePlan {
    bool has_router = false;
    int  in_dim = 0;
    bool router_normalize = true;
    bool router_normalize_set = false;
    Mat  router_W[4]; Vec router_b[4];

    bool has_heads = false;
    HeadSpec heads[4];

    bool ring_near = false; float ring_self = 0.f, ring_nb = 0.f;
    bool ring_custom = false;
    Mat  ring_W[4]; Vec ring_b[4];

    bool normalize_agg_set = false; bool normalize_agg = true;
    bool key_identity = false;
    bool has_query = false; Vec query;
    bool has_cls = false; Mat cls_W; Vec cls_b;
};

inline FramePlan parse_frame(const Value& root, const Frame& frame, const std::string& ctx) {
    as_obj(root, ctx);
    check_keys(root, {"schema", "name", "description", "classes", "features", "size", "n_classes",
                      "in_dim", "router", "heads", "ring", "center"}, ctx);
    const int dim = frame.dim();
    const int ncls = frame.n_classes();
    FramePlan p;

    if (auto* v = root.find("size")) {
        if (!v->is_string()) bad(ctx + ".size", "expected a string (\"SS\"/\"T\"/\"S\")");
        if ("MYLN-" + v->s != std::string(frame.tag()))
            bad(ctx + ".size", "config is for " + v->s + " but frame is " + frame.tag());
    }
    if (auto* v = root.find("n_classes")) {
        if ((int)as_num(*v, ctx + ".n_classes") != ncls)
            bad(ctx + ".n_classes", "config is for " + std::to_string((int)v->n) +
                                    " classes but frame has " + std::to_string(ncls));
    }
    if (auto* v = root.find("classes")) {
        const auto& a = as_arr(*v, ctx + ".classes");
        if ((int)a.size() != ncls) bad(ctx + ".classes", "expected " + std::to_string(ncls) + " names");
        for (size_t i = 0; i < a.size(); ++i)
            if (!a[i].is_string()) bad(fmt_idx(ctx + ".classes", i), "expected a string");
    }
    if (auto* v = root.find("features")) {
        const auto& a = as_arr(*v, ctx + ".features");
        for (size_t i = 0; i < a.size(); ++i)
            if (!a[i].is_string()) bad(fmt_idx(ctx + ".features", i), "expected a string");
    }
    if (auto* v = root.find("in_dim")) {
        double d = as_num(*v, ctx + ".in_dim");
        if (d < 1 || d != std::floor(d) || d > 4096) bad(ctx + ".in_dim", "expected an integer in [1,4096]");
        p.in_dim = (int)d;
    }

    // ── router ──
    if (auto* r = root.find("router")) {
        std::string rc = ctx + ".router";
        as_obj(*r, rc);
        check_keys(*r, {"normalize", "slots"}, rc);
        if (p.in_dim == 0) bad(ctx, "'in_dim' is required when 'router' is given");
        p.has_router = true;
        if (auto* n = r->find("normalize")) {
            if (!n->is_bool()) bad(rc + ".normalize", "expected true/false");
            p.router_normalize = n->b; p.router_normalize_set = true;
        }
        auto* sl = r->find("slots");
        if (!sl) bad(rc, "'slots' is required");
        const auto& sa = as_arr(*sl, rc + ".slots");
        if (sa.size() != 4) bad(rc + ".slots", "expected exactly 4 slots");
        for (int s = 0; s < 4; ++s) {
            std::string c = fmt_idx(rc + ".slots", s);
            as_obj(sa[s], c);
            check_keys(sa[s], {"W", "terms", "bias"}, c);
            p.router_W[s].assign((size_t)dim * p.in_dim, 0.f);
            p.router_b[s].assign(dim, 0.f);
            if (auto* w = sa[s].find("W"))     apply_dense(p.router_W[s], dim, p.in_dim, *w, c + ".W");
            if (auto* t = sa[s].find("terms")) apply_terms(p.router_W[s], dim, p.in_dim, *t, c + ".terms");
            if (auto* b = sa[s].find("bias"))  apply_bias(p.router_b[s], dim, *b, c + ".bias");
        }
    }

    // ── heads ──
    if (auto* h = root.find("heads")) {
        std::string hc = ctx + ".heads";
        const auto& ha = as_arr(*h, hc);
        if (ha.size() != 4) bad(hc, "expected exactly 4 heads");
        p.has_heads = true;
        for (int s = 0; s < 4; ++s) {
            std::string c = fmt_idx(hc, s);
            HeadSpec hs;
            if (ha[s].is_string()) hs.type = ha[s].s;
            else {
                as_obj(ha[s], c);
                check_keys(ha[s], {"type", "name", "seed"}, c);
                auto* t = ha[s].find("type");
                if (!t || !t->is_string()) bad(c, "'type' (string) is required");
                hs.type = t->s;
                if (auto* n = ha[s].find("name")) {
                    if (!n->is_string()) bad(c + ".name", "expected a string");
                    hs.name = n->s;
                }
                if (auto* sd = ha[s].find("seed")) hs.seed = (unsigned)as_index(*sd, 0, 2147483647, c + ".seed");
            }
            if (hs.type != "passthrough" && hs.type != "zero" && hs.type != "default")
                bad(c, "unknown head type '" + hs.type + "' (use passthrough / zero / default)");
            if (hs.name.empty()) hs.name = hs.type + "-" + std::to_string(s);
            if (hs.type == "default" && !ha[s].is_object()) hs.seed = (unsigned)s;
            p.heads[s] = hs;
        }
    }

    // ── ring ──
    if (auto* r = root.find("ring")) {
        std::string rc = ctx + ".ring";
        as_obj(*r, rc);
        check_keys(*r, {"mode", "self", "neighbor", "slots"}, rc);
        auto* m = r->find("mode");
        if (!m || !m->is_string()) bad(rc, "'mode' (\"near_identity\" or \"custom\") is required");
        if (m->s == "near_identity") {
            p.ring_near = true;
            p.ring_self = (float)(r->find("self")     ? as_num(*r->find("self"),     rc + ".self")     : 0.6);
            p.ring_nb   = (float)(r->find("neighbor") ? as_num(*r->find("neighbor"), rc + ".neighbor") : 0.2);
        } else if (m->s == "custom") {
            auto* sl = r->find("slots");
            if (!sl) bad(rc, "'slots' is required for mode \"custom\"");
            const auto& sa = as_arr(*sl, rc + ".slots");
            if (sa.size() != 4) bad(rc + ".slots", "expected exactly 4 slots");
            p.ring_custom = true;
            for (int s = 0; s < 4; ++s) {
                std::string c = fmt_idx(rc + ".slots", s);
                as_obj(sa[s], c);
                check_keys(sa[s], {"terms", "bias"}, c);
                p.ring_W[s].assign((size_t)dim * dim * 3, 0.f);
                p.ring_b[s].assign(dim, 0.f);
                if (auto* t = sa[s].find("terms")) apply_terms(p.ring_W[s], dim, dim * 3, *t, c + ".terms");
                if (auto* b = sa[s].find("bias"))  apply_bias(p.ring_b[s], dim, *b, c + ".bias");
            }
        } else bad(rc + ".mode", "unknown mode '" + m->s + "'");
    }

    // ── center ──
    if (auto* c = root.find("center")) {
        std::string cc = ctx + ".center";
        as_obj(*c, cc);
        check_keys(*c, {"normalize_agg", "key", "query", "cls"}, cc);
        if (auto* n = c->find("normalize_agg")) {
            if (!n->is_bool()) bad(cc + ".normalize_agg", "expected true/false");
            p.normalize_agg = n->b; p.normalize_agg_set = true;
        }
        if (auto* k = c->find("key")) {
            if (!k->is_string() || k->s != "identity") bad(cc + ".key", "only \"identity\" is supported");
            p.key_identity = true;
        }
        if (auto* q = c->find("query")) {
            p.has_query = true; p.query.assign(dim, 0.f);
            apply_bias(p.query, dim, *q, cc + ".query");   // [[dim_index, value], ...]
        }
        if (auto* cl = c->find("cls")) {
            std::string c2 = cc + ".cls";
            as_obj(*cl, c2);
            check_keys(*cl, {"W", "terms", "bias"}, c2);
            p.has_cls = true;
            p.cls_W.assign((size_t)ncls * dim, 0.f);
            p.cls_b.assign(ncls, 0.f);
            if (auto* w = cl->find("W"))     apply_dense(p.cls_W, ncls, dim, *w, c2 + ".W");
            if (auto* t = cl->find("terms")) apply_terms(p.cls_W, ncls, dim, *t, c2 + ".terms");
            if (auto* b = cl->find("bias"))  apply_bias(p.cls_b, ncls, *b, c2 + ".bias");
        }
    }
    return p;
}

inline void apply_frame(Frame& frame, FramePlan& p) {
    const int dim = frame.dim();
    if (p.has_router) {
        frame.init_router(p.in_dim);
        if (p.router_normalize_set) frame.router().set_normalize(p.router_normalize);
        for (int s = 0; s < 4; ++s) frame.router().set_slot(s, p.router_W[s], p.router_b[s]);
    }
    if (p.has_heads) {
        for (int s = 0; s < 4; ++s) {
            const HeadSpec& h = p.heads[s];
            if      (h.type == "passthrough") frame.set_head(s, std::make_unique<PassthroughHead>(h.name));
            else if (h.type == "zero")        frame.set_head(s, std::make_unique<ZeroHead>());
            else                              frame.set_head(s, std::make_unique<DefaultHead>(dim, h.name, h.seed));
        }
    }
    if (p.ring_near)   frame.ring().set_near_identity(p.ring_self, p.ring_nb);
    if (p.ring_custom) for (int s = 0; s < 4; ++s) frame.ring().set_merge(s, p.ring_W[s], p.ring_b[s]);
    if (p.normalize_agg_set) frame.center().set_normalize_agg(p.normalize_agg);
    if (p.key_identity)      frame.center().set_key_identity();
    if (p.has_query)         frame.center().set_query(p.query);
    if (p.has_cls)           frame.center().set_cls(p.cls_W, p.cls_b);
}

// '{' で始まれば JSON 文字列、それ以外はパスとして読む
inline std::string load_text(const std::string& path_or_json) {
    size_t i = path_or_json.find_first_not_of(" \t\r\n");
    if (i != std::string::npos && path_or_json[i] == '{') return path_or_json;
    std::ifstream f(path_or_json, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open config file: " + path_or_json);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

} // namespace cfg

// ── 公開API ────────────────────────────────────────────────
inline void tune_custom(Frame& frame, const std::string& path_or_json) {
    auto root = json::parse(cfg::load_text(path_or_json));
    auto plan = cfg::parse_frame(root, frame, "config");
    cfg::apply_frame(frame, plan);
}

// cascade: {"threshold": 0.8, "relay": {frame設定}, "full": {frame設定}}
inline void tune_cascade_custom(CascadeFrame& cascade, const std::string& path_or_json) {
    auto root = json::parse(cfg::load_text(path_or_json));
    cfg::as_obj(root, "config");
    cfg::check_keys(root, {"schema", "name", "description", "classes", "features", "threshold", "relay", "full"}, "config");
    auto* r = root.find("relay");
    auto* f = root.find("full");
    if (!r || !f) cfg::bad("config", "cascade config needs both 'relay' and 'full'");
    float thr = cascade.threshold();
    if (auto* t = root.find("threshold")) {
        double d = cfg::as_num(*t, "config.threshold");
        if (d < 0.0 || d > 1.0) cfg::bad("config.threshold", "expected a value in [0,1]");
        thr = (float)d;
    }
    auto rp = cfg::parse_frame(*r, cascade.relay(), "config.relay");
    auto fp = cfg::parse_frame(*f, cascade.full(),  "config.full");
    cascade.set_threshold(thr);
    cfg::apply_frame(cascade.relay(), rp);
    cfg::apply_frame(cascade.full(),  fp);
}

} // namespace myln
