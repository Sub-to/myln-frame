#include "../include/myln/cascade.h"
#include "../include/myln/tune_config.h"
#include "../tuner/difficulty_tuner.h"
#include <iostream>
#include <iomanip>

// 難易度チューナー: C++ チューナーと JSON 設定(configs/difficulty_cascade.json)が同じ結果になることを示す。
// 実行: difficulty_demo [configs/difficulty_cascade.json]

static void run(const char* label, myln::CascadeFrame& a, myln::CascadeFrame& b, const myln::Vec& f) {
    auto ra = a.run(f);
    auto rb = b.run(f);
    int ia = (int)(std::max_element(ra.probs.begin(), ra.probs.end()) - ra.probs.begin());
    int ib = (int)(std::max_element(rb.probs.begin(), rb.probs.end()) - rb.probs.begin());
    std::cout << std::left << std::setw(26) << label
              << " C++: " << std::setw(8) << myln::DIFFICULTY_CLASS_NAMES[ia]
              << " (" << myln::DIFFICULTY_CLASS_NAMES_JA[ia] << ")  "
              << (ra.used_relay ? "relay" : "full ")
              << "   JSON: " << myln::DIFFICULTY_CLASS_NAMES[ib]
              << ((ia == ib && ra.used_relay == rb.used_relay) ? "  == same" : "  != DIFFERENT") << "\n";
}

int main(int argc, char** argv) {
    const char* cfg = argc > 1 ? argv[1] : "configs/difficulty_cascade.json";

    myln::CascadeFrame by_cpp(0.80f);
    myln::tune_cascade_difficulty(by_cpp);

    myln::CascadeFrame by_json(0.80f);
    try { myln::tune_cascade_custom(by_json, cfg); }
    catch (const std::exception& e) { std::cerr << "config error: " << e.what() << "\n"; return 1; }

    std::cout << "features = [tech, length, steps, scope, reasoning]\n\n";
    //                              tech  len   steps scope reason
    run("chat",                   by_cpp, by_json, {0.0f, 0.01f, 0.0f, 0.0f, 0.0f});
    run("rename a function",      by_cpp, by_json, {0.33f, 0.03f, 0.0f, 0.0f, 0.0f});
    run("fix bug in main.py",     by_cpp, by_json, {0.33f, 0.05f, 0.0f, 0.0f, 0.67f});
    run("multi-step refactor",    by_cpp, by_json, {0.67f, 0.14f, 1.0f, 1.0f, 1.0f});
    run("repo-wide migration",    by_cpp, by_json, {1.0f, 0.17f, 0.33f, 1.0f, 0.0f});
    return 0;
}
