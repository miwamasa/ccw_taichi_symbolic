"""
ik_solver.py - 逆運動学（IK）モジュール

ikpy を使って腕の手先IKを解く。手法記号から手先目標位置を求め、
肩・肘の関節角度に変換する。

PyBullet humanoid.urdf の腕チェーン構造:
    base -> ... -> chest -> right_shoulder1 -> right_shoulder2 -> right_elbow -> right_wrist
                         -> left_shoulder1  -> left_shoulder2  -> left_elbow  -> left_wrist
"""

import numpy as np
from typing import Dict, Optional, Tuple

# 手法記号 → 手先目標位置（胸部ローカル座標 [前方x, 側方y, 高さz] メートル）
# 座標系: x=前方, y=左が正（左手側）, z=上方
HAND_EE_TARGETS: Dict[str, Dict[str, np.ndarray]] = {
    "抱": {
        "left":  np.array([0.25, +0.15, 0.30]),
        "right": np.array([0.25, -0.15, 0.30]),
    },
    "掤": {
        "left":  np.array([0.35, +0.10, 0.40]),
        "right": np.array([0.20, -0.05, 0.35]),
    },
    "捋": {
        "left":  np.array([0.20, +0.20, 0.35]),
        "right": np.array([0.40, -0.10, 0.30]),
    },
    "擠": {
        "left":  np.array([0.40, +0.05, 0.35]),
        "right": np.array([0.40, -0.05, 0.35]),
    },
    "按": {
        "left":  np.array([0.30, +0.10, 0.20]),
        "right": np.array([0.30, -0.10, 0.20]),
    },
    "分": {
        "left":  np.array([0.10, +0.25, 0.55]),
        "right": np.array([0.10, -0.25, -0.05]),
    },
    "穿": {
        "left":  np.array([0.40, +0.20, 0.30]),
        "right": np.array([0.20, -0.10, 0.15]),
    },
    "雲": {
        "left":  np.array([0.30, +0.15, 0.40]),
        "right": np.array([0.30, -0.15, 0.40]),
    },
    "架": {
        "left":  np.array([0.15, +0.10, 0.55]),
        "right": np.array([0.20, -0.15, 0.20]),
    },
    "打": {
        "left":  np.array([0.15, +0.10, 0.20]),
        "right": np.array([0.45, -0.05, 0.35]),
    },
    "推": {
        "left":  np.array([0.20, +0.10, 0.30]),
        "right": np.array([0.45, -0.05, 0.30]),
    },
    "定": {
        "left":  np.array([0.20, +0.10, 0.25]),
        "right": np.array([0.20, -0.10, 0.25]),
    },
}

# IK を使わず primitive_library を直接使う記号
IK_SKIP_SYMBOLS = {"定", "抱"}

# 関節角リミット（ラジアン）
JOINT_LIMITS = {
    "left_shoulder1":  (-0.2, 1.5),
    "left_shoulder2":  (-0.8, 0.8),
    "left_elbow":      (-1.8, 0.0),
    "right_shoulder1": (-0.2, 1.5),
    "right_shoulder2": (-0.8, 0.8),
    "right_elbow":     (-1.8, 0.0),
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _target_to_arm_angles(target: np.ndarray, side: str) -> Dict[str, float]:
    """
    目標手先位置から肩・肘の関節角度を解析的に近似する。

    ikpy を使う場合の代替として、ジオメトリから直接計算する簡易IK。
    上腕長 = 0.28m, 前腕長 = 0.25m と仮定（humanoid.urdfスケール）。

    Args:
        target: [x, y, z] 胸部ローカル座標での手先目標位置
        side: "left" or "right"

    Returns:
        {joint_name: angle_rad}
    """
    x, y, z = target

    # 右腕は y を反転（右側はマイナス方向）
    if side == "right":
        y = -y
        prefix = "right"
    else:
        prefix = "left"

    upper_arm = 0.28  # 上腕長（m）
    forearm   = 0.25  # 前腕長（m）

    # 距離を計算
    dist = np.sqrt(x**2 + y**2 + z**2)
    dist = np.clip(dist, 0.05, upper_arm + forearm - 0.01)

    # 余弦定理で肘角度を計算
    cos_elbow = (upper_arm**2 + forearm**2 - dist**2) / (2 * upper_arm * forearm)
    cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
    # 肘は曲がる方向（負の角度）
    elbow_angle = -(np.pi - np.arccos(cos_elbow))

    # 肩の角度: 前方向き(x)と高さ(z)から計算
    shoulder1 = np.arctan2(z, x)  # 前方・高さ方向
    shoulder2 = np.arctan2(y, np.sqrt(x**2 + z**2))  # 側方

    # リミットを適用
    shoulder1 = _clamp(shoulder1, *JOINT_LIMITS[f"{prefix}_shoulder1"])
    shoulder2 = _clamp(shoulder2, *JOINT_LIMITS[f"{prefix}_shoulder2"])
    elbow_angle = _clamp(elbow_angle, *JOINT_LIMITS[f"{prefix}_elbow"])

    return {
        f"{prefix}_shoulder1": shoulder1,
        f"{prefix}_shoulder2": shoulder2,
        f"{prefix}_elbow": elbow_angle,
    }


def compute_arm_ik(hand_symbol: str) -> Optional[Dict[str, float]]:
    """
    手法記号から左右の腕関節角度をIKで計算する。

    Returns:
        {"left_shoulder1": ..., "left_elbow": ..., "right_shoulder1": ..., ...}
        IK対象外の記号の場合は None を返す。
    """
    if hand_symbol not in HAND_EE_TARGETS or hand_symbol in IK_SKIP_SYMBOLS:
        return None

    targets = HAND_EE_TARGETS[hand_symbol]
    result = {}

    for side in ["left", "right"]:
        target = targets[side]
        angles = _target_to_arm_angles(target, side)
        result.update(angles)

    return result


def apply_ik_to_keyframe(joint_target: Dict[str, float],
                          hand_symbol: str) -> Dict[str, float]:
    """
    プリミティブライブラリから得たキーフレームにIKを適用する。
    IKの結果で肩・肘角度を上書きする。

    Args:
        joint_target: primitive_library から解決した関節角辞書
        hand_symbol:  手法記号

    Returns:
        IK適用後の関節角辞書（コピー）
    """
    result = dict(joint_target)
    ik_angles = compute_arm_ik(hand_symbol)
    if ik_angles is not None:
        result.update(ik_angles)
    return result


def try_import_ikpy():
    """ikpy が使用可能かチェック。使用可能なら True を返す。"""
    try:
        import ikpy  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # 動作確認テスト
    for symbol in ["分", "掤", "雲", "按", "架"]:
        angles = compute_arm_ik(symbol)
        if angles:
            print(f"\n手法「{symbol}」のIK結果:")
            for joint, angle in sorted(angles.items()):
                print(f"  {joint}: {np.degrees(angle):.1f}°")
        else:
            print(f"\n手法「{symbol}」: IKスキップ（primitiveを直接使用）")
