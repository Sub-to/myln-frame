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

/* ── エラー処理 ─────────────────────────────────────────────
 * 失敗する関数は NULL（ポインタを返すもの）または -1（int を返すもの）を返し、
 * 直前のエラーメッセージを myln_last_error() で取得できる。
 * C++ 例外は C の境界を越えない。メッセージはスレッドごとに保持される。
 * 成功した呼び出しではエラーはクリアされない（失敗の直後に読むこと）。
 */
MYLN_API const char* myln_last_error(void);

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
MYLN_API int  myln_tune_security(void* frame, int in_dim);

/* 出力確率の鋭さ（logits に掛ける係数, > 0）。判定クラスは変わらず、確信度だけが
 * 変わる。既定 1.0。3.0 前後で代表シナリオが 0.7〜0.85 になる。 */
MYLN_API int  myln_set_logit_scale(void* frame, float scale);

/* ── 推論 ──────────────────────────────────────────────────
 * features : 入力特徴量配列 (float32)
 * n_in     : 入力次元数
 * out_n    : [out] 出力クラス数が書き込まれる
 * 戻り値   : クラス確率配列 (フレームが所有するバッファ)。失敗時は NULL。
 *            1 つのフレームを複数スレッドから同時に使ってはいけない。
 *            次の推論で上書きされるので、必要なら呼び出し側でコピーすること。
 *
 * NaN/Inf を含む入力、入力次元の不一致は失敗（NULL）になる。
 */
MYLN_API const float* myln_infer(void* frame, const float* features, int n_in, int* out_n);

/* 呼び出し側のバッファに書き込む版（フレームのバッファを共有しない）。
 * out_probs には myln_n_classes() 個の float を確保しておく。
 * 戻り値: 0 = 成功 / -1 = 失敗（myln_last_error 参照） */
MYLN_API int myln_infer_into(void* frame, const float* features, int n_in, float* out_probs);

/* ── メタ情報 ──────────────────────────────────────────────*/
MYLN_API const char* myln_tag      (void* frame);
MYLN_API int         myln_dim      (void* frame);
MYLN_API int         myln_n_classes(void* frame);
MYLN_API const char* myln_version  (void);

/* ── 地震監視 API ──────────────────────────────────────────
 * features: [intensity/7, magnitude/9, depth_inv, tsunami, freq/10]
 * 出力クラス: SAFE / LOW / MEDIUM / HIGH / CRITICAL
 */
MYLN_API int  myln_tune_earthquake(void* frame, int in_dim);

/* ── カスケード（2段リレー）API ────────────────────────────
 * リレー（SS 2頭）で高速判定 → 曖昧なら フル（T 4頭）へ
 * threshold: 確信度の閾値 (0.0〜1.0, 推奨 0.80)
 */
MYLN_API void*       myln_cascade_new      (float threshold);
MYLN_API void        myln_cascade_free     (void* cas);
MYLN_API int         myln_cascade_tune_security(void* cas, int in_dim);
MYLN_API const float* myln_cascade_infer   (void* cas, const float* features,
                                            int n_in, int* out_n,
                                            int* out_used_relay);
MYLN_API float       myln_cascade_relay_rate(void* cas); // リレー通過率
MYLN_API int         myln_cascade_infer_into(void* cas, const float* features, int n_in,
                                             float* out_probs /* 5 個 */,
                                             int* out_used_relay);

#ifdef __cplusplus
}
#endif
#endif /* MYLN_C_API_H */
