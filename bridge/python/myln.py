"""
MYLN-FRAME Python Bridge
========================
ctypes経由でC APIを呼ぶ薄いラッパー。
GPUもフレームワークも不要。

Usage:
    from myln import MylnFrame

    frame = MylnFrame(size="T", n_classes=5)
    frame.tune_security(in_dim=5)

    probs  = frame.infer([0.9, 0.95, 0.8, 0.99, 0.85])
    label  = frame.predict([0.9, 0.95, 0.8, 0.99, 0.85])
    print(label)  # → CRITICAL

ライブラリの場所は環境変数 MYLN_LIB（libmyln.so / .dylib / .dll へのパス）で
上書きできる。無ければ <repo>/build/ とカレントディレクトリを探す。
"""

import ctypes
import math
import os
from pathlib import Path
from typing import Optional, Sequence

__all__ = ["MylnFrame", "MylnCascade", "MylnError"]


class MylnError(RuntimeError):
    """libmyln が失敗を返したときの例外（メッセージは myln_last_error）。"""


# ── libmyln の検索 ────────────────────────────────────────
def _find_lib() -> str:
    env = os.environ.get("MYLN_LIB")
    if env:
        if Path(env).exists():
            return env
        raise FileNotFoundError(f"MYLN_LIB={env!r} が存在しません")

    root = Path(__file__).resolve().parent.parent.parent
    names = ["libmyln.so", "libmyln.dylib", "myln.dll", "libmyln.dll"]
    dirs = [root / "build", root / "build" / "Release", root / "build" / "Debug", Path(".")]
    for d in dirs:
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
    raise FileNotFoundError(
        "libmyln が見つかりません。まず build/ でビルドしてください。\n"
        "  cmake -S . -B build && cmake --build build --target myln\n"
        "（別の場所にある場合は環境変数 MYLN_LIB にパスを指定）"
    )


# ── C API バインディング ──────────────────────────────────
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_int_p = ctypes.POINTER(ctypes.c_int)


class _CAPI:
    def __init__(self, lib_path: Optional[str] = None):
        path = lib_path or _find_lib()
        lib = ctypes.CDLL(path)

        def sig(name, restype, argtypes):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            return fn

        sig("myln_last_error", ctypes.c_char_p, [])
        sig("myln_new", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int])
        sig("myln_free", None, [ctypes.c_void_p])
        sig("myln_tune_security", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int])
        sig("myln_tune_earthquake", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int])
        sig("myln_set_logit_scale", ctypes.c_int, [ctypes.c_void_p, ctypes.c_float])
        sig("myln_infer_into", ctypes.c_int,
            [ctypes.c_void_p, _c_float_p, ctypes.c_int, _c_float_p])
        sig("myln_tag", ctypes.c_char_p, [ctypes.c_void_p])
        sig("myln_dim", ctypes.c_int, [ctypes.c_void_p])
        sig("myln_n_classes", ctypes.c_int, [ctypes.c_void_p])
        sig("myln_version", ctypes.c_char_p, [])

        # カスケード API
        sig("myln_cascade_new", ctypes.c_void_p, [ctypes.c_float])
        sig("myln_cascade_free", None, [ctypes.c_void_p])
        sig("myln_cascade_set_policy", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int])
        sig("myln_cascade_tune_security", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int])
        sig("myln_cascade_infer_into", ctypes.c_int,
            [ctypes.c_void_p, _c_float_p, ctypes.c_int, _c_float_p, _c_int_p])
        sig("myln_cascade_relay_rate", ctypes.c_float, [ctypes.c_void_p])

        self.lib = lib

    def version(self) -> str:
        return self.lib.myln_version().decode()

    def error(self) -> str:
        msg = self.lib.myln_last_error()
        return msg.decode() if msg else "unknown error"

    def check(self, rc: int) -> None:
        if rc != 0:
            raise MylnError(self.error())


def _to_c_floats(features: Sequence[float]):
    """検証つきで float 配列に変換（NaN/Inf は C 側に渡す前に弾く）。"""
    n = len(features)
    if n == 0:
        raise ValueError("features must not be empty")
    for i, v in enumerate(features):
        if not math.isfinite(v):
            raise ValueError(f"feature {i} is not finite: {v!r}")
    return (ctypes.c_float * n)(*features), n


# ── メインクラス ──────────────────────────────────────────
class MylnFrame:
    """
    MYLN-FRAME の Python ラッパー。
    どのプラットフォームでも ctypes だけで動く量産型ブリッジ。

    1 つのインスタンスを複数スレッドから同時に使ってはいけない。
    """

    SECURITY_CLASSES = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    EARTHQUAKE_CLASSES = SECURITY_CLASSES

    def __init__(
        self,
        size: str = "T",
        n_classes: int = 5,
        lib_path: Optional[str] = None,
    ):
        self._handle = None
        self._api    = _CAPI(lib_path)
        handle = self._api.lib.myln_new(size.encode(), n_classes)
        if not handle:
            raise MylnError(self._api.error())
        self._handle = handle
        self._n_classes = n_classes

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._api.lib.myln_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self) -> "MylnFrame":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _live(self):
        if not self._handle:
            raise MylnError("frame is closed")
        return self._handle

    # ── チューニング ────────────────────────────────────────
    def tune_security(self, in_dim: int = 5, sharpness: float = 1.0) -> "MylnFrame":
        """
        セキュリティ監視用に重みを手動チューニングする。

        sharpness: 出力確率の鋭さ（既定 1.0 = 従来どおり）。判定クラスは変わらず
                   確信度だけが上がる。3.0 前後で代表シナリオが 0.7〜0.85 になる。
        """
        h = self._live()
        self._api.check(self._api.lib.myln_tune_security(h, in_dim))
        if sharpness != 1.0:
            self.set_sharpness(sharpness)
        return self  # メソッドチェーン用

    def tune_earthquake(self, in_dim: int = 5) -> "MylnFrame":
        """地震監視用チューニング。features: [intensity/7, magnitude/9, depth_inv, tsunami, freq/10]"""
        self._api.check(self._api.lib.myln_tune_earthquake(self._live(), in_dim))
        return self

    def set_sharpness(self, sharpness: float) -> "MylnFrame":
        self._api.check(self._api.lib.myln_set_logit_scale(self._live(), sharpness))
        return self

    # ── 推論 ───────────────────────────────────────────────
    def infer(self, features: Sequence[float]) -> list:
        """
        特徴量リストを渡してクラス確率を返す。
        features: [proc_anomaly, cpu_spike, net_bytes, file_change, mem_pressure]
                  すべて 0.0〜1.0 の有限値
        """
        arr, n_in = _to_c_floats(features)
        out = (ctypes.c_float * self._n_classes)()
        self._api.check(self._api.lib.myln_infer_into(self._live(), arr, n_in, out))
        return list(out)

    def predict(self, features: Sequence[float], classes: Optional[list] = None) -> str:
        """最も確率の高いクラス名を返す。"""
        labels = classes or self.SECURITY_CLASSES
        probs  = self.infer(features)
        return labels[probs.index(max(probs))]

    def predict_with_score(self, features: Sequence[float], classes: Optional[list] = None):
        """(クラス名, 確率) のタプルを返す。"""
        labels = classes or self.SECURITY_CLASSES
        probs  = self.infer(features)
        best   = probs.index(max(probs))
        return labels[best], probs[best]

    # ── メタ情報 ───────────────────────────────────────────
    @property
    def tag(self)       -> str: return self._api.lib.myln_tag(self._live()).decode()
    @property
    def dim(self)       -> int: return self._api.lib.myln_dim(self._live())
    @property
    def n_classes(self) -> int: return self._api.lib.myln_n_classes(self._live())
    @property
    def version(self)   -> str: return self._api.version()

    def __repr__(self):
        if not getattr(self, "_handle", None):
            return "MylnFrame(closed)"
        return f"MylnFrame(tag={self.tag!r}, dim={self.dim}, classes={self.n_classes})"


# ── カスケード（2段リレー）────────────────────────────────────
class MylnCascade:
    """
    2段カスケード分類器（リレー: SS 2頭 proc+file / フル: T 4頭）。

    exact=True（既定）: 常にフルの結果を返す（MylnFrame("T").tune_security() と同一）。
    exact=False       : リレーで高速判定し、確信度 >= threshold ならそのまま、
                        そうでなければフルへ。リレーは proc / file しか見ない近似なので、
                        早期終了の判定がフルと食い違うことがある（README の Cascade 節参照）。

    Usage:
        cas = MylnCascade(threshold=0.80, exact=False).tune_security()
        label, conf, used_relay = cas.predict_with_path([0.9,0.95,0.8,0.99,0.85])
    """
    SECURITY_CLASSES = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(self, threshold: float = 0.80, exact: bool = True,
                 lib_path: Optional[str] = None):
        self._handle = None
        self._api    = _CAPI(lib_path)
        handle = self._api.lib.myln_cascade_new(ctypes.c_float(threshold))
        if not handle:
            raise MylnError(self._api.error())
        self._handle = handle
        self._api.check(self._api.lib.myln_cascade_set_policy(handle, 1 if exact else 0))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._api.lib.myln_cascade_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self) -> "MylnCascade":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _live(self):
        if not self._handle:
            raise MylnError("cascade is closed")
        return self._handle

    def tune_security(self, in_dim: int = 5) -> "MylnCascade":
        self._api.check(self._api.lib.myln_cascade_tune_security(self._live(), in_dim))
        return self

    def infer(self, features: Sequence[float]) -> tuple:
        """(probs, used_relay) を返す"""
        arr, n_in = _to_c_floats(features)
        out   = (ctypes.c_float * 5)()
        relay = ctypes.c_int(0)
        self._api.check(self._api.lib.myln_cascade_infer_into(
            self._live(), arr, n_in, out, ctypes.byref(relay)))
        return list(out), bool(relay.value)

    def predict(self, features: Sequence[float]) -> str:
        probs, _ = self.infer(features)
        return self.SECURITY_CLASSES[probs.index(max(probs))]

    def predict_with_path(self, features: Sequence[float]) -> tuple:
        """(クラス名, 確信度, リレー使用?) を返す"""
        probs, used_relay = self.infer(features)
        best = probs.index(max(probs))
        return self.SECURITY_CLASSES[best], probs[best], used_relay

    @property
    def relay_rate(self) -> float:
        return self._api.lib.myln_cascade_relay_rate(self._live())
