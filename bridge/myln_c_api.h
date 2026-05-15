/*
 * MYLN-FRAME  C API  ── Universal Bridge
 * ========================================
 * C99互換のフラットAPIで、あらゆる言語から呼べる量産型ブリッジ。
 *
 * Python  → ctypes / cffi
 * Node.js → ffi-napi
 * Ruby    → Fiddle
 * Go      → cgo
 * Rust    → bindgen
 *
 * Usage (どの言語でも同じ流れ):
 *   frame = myln_new("T", 5)
 *   myln_tune_security(frame, 5)
 *   probs = myln_infer(frame, features, 5, &n)
 *   myln_free(frame)
 */

#ifndef MYLN_C_API_H
#define MYLN_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* エクスポートマクロ */
#if defined(_WIN32)
#  define MYLN_API __declspec(dllexport)
#else
#  define MYLN_API __attribute__((visibility("default")))
#endif

/* ── フレームのライフサイクル ──────────────────────────────
 * size     : "SS" / "T" / "S"
 * n_classes: 出力クラス数（チビタルなら 5）
 * 戻り値   : フレームハンドル (NULL = 失敗)
 */
MYLN_API void* myln_new (const char* size, int n_classes);
MYLN_API void  myln_free(void* frame);

/* ── チューニング ──────────────────────────────────────────
 * 現在利用可能なチューニング:
 *   myln_tune_security : セキュリティ監視用（チビタル互換）
 *
 * 将来の頭:
 *   myln_tune_weather  : 天気・台風判定（九州男丸）
 *   myln_tune_voice    : 音声・会話
 *   myln_tune_custom   : 設定ファイルから読み込み（将来）
 */
MYLN_API void myln_tune_security(void* frame, int in_dim);

/* ── 推論 ──────────────────────────────────────────────────
 * features : 入力特徴量配列 (float32)
 * n_in     : 入力次元数
 * out_n    : [out] 出力クラス数が書き込まれる
 * 戻り値   : クラス確率配列 (フレームが所有するバッファ)
 *            スレッドセーフではない。コピーが必要な場合は呼び出し側で行うこと。
 */
MYLN_API const float* myln_infer(void* frame, const float* features, int n_in, int* out_n);

/* ── メタ情報 ──────────────────────────────────────────────*/
MYLN_API const char* myln_tag      (void* frame);
MYLN_API int         myln_dim      (void* frame);
MYLN_API int         myln_n_classes(void* frame);
MYLN_API const char* myln_version  (void);

#ifdef __cplusplus
}
#endif
#endif /* MYLN_C_API_H */
