"""
ik_solver.py - 逆運動学（IK）モジュール

手法記号から手先目標位置を定義し、解析的 IK で肩・肘の DOF 角度を計算する。

新しい DOF 名（motion_generator.py の DOF 定義に対応）:
  left_shoulder_rx, left_shoulder_ry, left_shoulder_rz, left_elbow
  right_shoulder_rx, right_shoulder_ry, right_shoulder_rz, right_elbow
"""

import numpy as np
from typing import Dict, Optional

# 手法記号 → 手先目標位置（体幹ローカル座標 [前x, 左y, 上z] メートル）
# 座標系: x=前方, y=左が正（左手側）, z=上方
HAND_EE_TARGETS: Dict[str, Dict[str, np.ndarray]] = {
    "抱": {
        "left":  np.array([0.25,  0.15, 0.10]),
        "right": np.array([0.25, -0.15, 0.10]),
    },
    "掤": {
        "left":  np.array([0.35,  0.10, 0.20]),
        "right": np.array([0.20, -0.05, 0.15]),
    },
    "捋": {
        "left":  np.array([0.20,  0.25, 0.15]),
        "right": np.array([0.40, -0.15, 0.10]),
    },
    "擠": {
        "left":  np.array([0.40,  0.05, 0.15]),
        "right": np.array([0.40, -0.05, 0.15]),
    },
    "按": {
        "left":  np.array([0.30,  0.10, 0.00]),
        "right": np.array([0.30, -0.10, 0.00]),
    },
    "分": {
        "left":  np.array([0.10,  0.20, 0.35]),
        "right": np.array([0.10, -0.20,-0.10]),
    },
    "穿": {
        "left":  np.array([0.40,  0.20, 0.10]),
        "right": np.array([0.20, -0.10, 0.00]),
    },
    "雲": {
        "left":  np.array([0.30,  0.20, 0.25]),
        "right": np.array([0.30, -0.20, 0.25]),
    },
    "架": {
        "left":  np.array([0.15,  0.10, 0.40]),
        "right": np.array([0.20, -0.15, 0.05]),
    },
    "打": {
        "left":  np.array([0.15,  0.10, 0.00]),
        "right": np.array([0.45, -0.05, 0.15]),
    },
    "推": {
        "left":  np.array([0.20,  0.10, 0.10]),
        "right": np.array([0.45, -0.05, 0.10]),
    },
    "定": {
        "left":  np.array([0.15,  0.10, 0.00]),
        "right": np.array([0.15, -0.10, 0.00]),
    },
}

# IK を適用しない記号（プリミティブをそのまま使う）
IK_SKIP_SYMBOLS = {"抱"}

# 腕リンク長（humanoid.urdf スケール概算）
UPPER_ARM = 0.26   # 上腕長 (m)
FOREARM   = 0.24   # 前腕長 (m)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _arm_analytic_ik(target: np.ndarray, side: str) -> Dict[str, float]:
    """
    手先目標位置 [x,y,z] から肩・肘の DOF 角度を解析的に計算する。

    DOF 名（新システム）:
      {side}_shoulder_rx, {side}_shoulder_ry, {side}_shoulder_rz
      {side}_elbow

    Args:
        target: [前x, 側y(左+), 上z]  (体幹ローカル m)
        side: "left" or "right"
    """
    x, y, z = target

    # 右腕は y 方向を反転（右は -y が外側）
    if side == "right":
        y = -y

    dist = np.linalg.norm([x, y, z])
    dist = np.clip(dist, 0.05, UPPER_ARM + FOREARM - 0.01)

    # 余弦定理で肘角度（正値=曲げる）
    cos_e = (UPPER_ARM**2 + FOREARM**2 - dist**2) / (2 * UPPER_ARM * FOREARM)
    cos_e = np.clip(cos_e, -1.0, 1.0)
    elbow = float(np.pi - np.arccos(cos_e))   # 0（伸ばす）～ π（完全屈曲）

    # 肩角度: ry = 前後方向の挙上, rx = 側方開き, rz = 回旋
    shoulder_ry = float(np.arctan2(z, max(x, 0.01)))
    shoulder_rx = float(np.arctan2(y, np.sqrt(x**2 + z**2)))
    shoulder_rz = 0.0  # 回旋は記号から来る値のままにする

    # リミット
    shoulder_ry = _clamp(shoulder_ry, -0.2, 1.6)
    shoulder_rx = _clamp(shoulder_rx, -0.8, 0.8)
    elbow       = _clamp(elbow,        0.0, 2.5)

    prefix = side
    return {
        f"{prefix}_shoulder_ry": shoulder_ry,
        f"{prefix}_shoulder_rx": shoulder_rx,
        f"{prefix}_shoulder_rz": shoulder_rz,
        f"{prefix}_elbow":       elbow,
    }


def compute_arm_ik(hand_symbol: str) -> Optional[Dict[str, float]]:
    """
    手法記号から左右の腕 DOF 角度を IK で計算する。

    Returns:
        DOF 名 → angle のdict、または IK 対象外の場合 None。
    """
    if hand_symbol not in HAND_EE_TARGETS or hand_symbol in IK_SKIP_SYMBOLS:
        return None

    targets = HAND_EE_TARGETS[hand_symbol]
    result: Dict[str, float] = {}

    for side in ("left", "right"):
        angles = _arm_analytic_ik(targets[side], side)
        result.update(angles)

    return result


def apply_ik_to_keyframe(joint_target: Dict[str, float],
                          hand_symbol: str) -> Dict[str, float]:
    """
    プリミティブライブラリの DOF dict に IK の結果を上書きする。

    Args:
        joint_target: resolve_symbols() から得た DOF dict のコピー
        hand_symbol:  手法記号

    Returns:
        IK 適用後の DOF dict（コピー）
    """
    result = dict(joint_target)
    ik_angles = compute_arm_ik(hand_symbol)
    if ik_angles is not None:
        result.update(ik_angles)
    return result


if __name__ == "__main__":
    for sym in ["分", "掤", "雲", "按", "架", "打"]:
        angles = compute_arm_ik(sym)
        if angles:
            print(f"\n手法「{sym}」IK結果:")
            for k, v in sorted(angles.items()):
                print(f"  {k}: {np.degrees(v):.1f}°")
        else:
            print(f"\n手法「{sym}」: IKスキップ")
