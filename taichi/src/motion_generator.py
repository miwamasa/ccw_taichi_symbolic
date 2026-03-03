"""
motion_generator.py - モーション軌道生成モジュール

実際の humanoid.urdf 関節構造に対応した DOF 設計:

  REVOLUTE 関節 (4本):
    right_elbow  (idx=4,  範囲 0 ~ π)
    left_elbow   (idx=7,  範囲 0 ~ π)
    right_knee   (idx=10, 範囲 -π ~ 0)  ← 負値で屈曲
    left_knee    (idx=13, 範囲 -π ~ 0)

  SPHERICAL 関節 (7個 × 3軸 euler = 21 DOF):
    chest, right_shoulder, left_shoulder,
    right_hip, right_ankle, left_hip, left_ankle

合計 25 DOF。球面関節は euler[rx,ry,rz] で表現し、
simulator.py でクォータニオンに変換して setJointMotorControlMultiDof を呼ぶ。
"""

import json
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple

try:
    from src.ik_solver import apply_ik_to_keyframe
except ImportError:
    from ik_solver import apply_ik_to_keyframe

# ────────────────────────────────────────────────────────────
# DOF 定義
# ────────────────────────────────────────────────────────────

# 回転関節（スカラー角）
REVOLUTE_DOF_NAMES = [
    "right_elbow",
    "left_elbow",
    "right_knee",
    "left_knee",
]

# 球面関節（各 3 euler 軸: rx=roll, ry=pitch, rz=yaw）
SPHERICAL_JOINT_NAMES = [
    "chest",
    "right_shoulder",
    "left_shoulder",
    "right_hip",
    "right_ankle",
    "left_hip",
    "left_ankle",
]

SPHERICAL_DOF_NAMES = [
    f"{j}_{ax}"
    for j in SPHERICAL_JOINT_NAMES
    for ax in ["rx", "ry", "rz"]
]

ALL_DOF_NAMES = REVOLUTE_DOF_NAMES + SPHERICAL_DOF_NAMES
N_DOFS = len(ALL_DOF_NAMES)          # 4 + 7*3 = 25
DOF_INDEX = {name: i for i, name in enumerate(ALL_DOF_NAMES)}

# URDF 関節インデックス
JOINT_IDX = {
    "chest":          1,
    "neck":           2,
    "right_shoulder": 3,
    "right_elbow":    4,
    "left_shoulder":  6,
    "left_elbow":     7,
    "right_hip":      9,
    "right_knee":     10,
    "right_ankle":    11,
    "left_hip":       12,
    "left_knee":      13,
    "left_ankle":     14,
}


def load_primitives(primitive_json_path: str) -> Dict:
    """プリミティブライブラリを読み込む。"""
    with open(primitive_json_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_symbols(symbols: Dict[str, str],
                    primitives: Dict,
                    use_ik: bool = True) -> np.ndarray:
    """
    記号辞書からターゲット DOF ベクトルを生成する。

    Returns:
        shape (N_DOFS,) のベクトル（revolute はラジアン, spherical は euler 角ラジアン）
    """
    target: Dict[str, float] = {}

    for axis in ["歩型", "手法", "身法", "歩法"]:
        symbol = symbols.get(axis)
        if symbol and axis in primitives:
            lib = primitives[axis]
            if symbol in lib:
                for dof, val in lib[symbol].items():
                    if not dof.startswith("_"):
                        target[dof] = target.get(dof, 0.0) + val

    # IK で手先位置から肩・肘角度を精密化
    if use_ik:
        hand_symbol = symbols.get("手法", "定")
        target = apply_ik_to_keyframe(target, hand_symbol)

    # DOF ベクトルに変換
    q = np.zeros(N_DOFS)
    for dof, val in target.items():
        if dof in DOF_INDEX:
            q[DOF_INDEX[dof]] = val

    return q


def generate_form_trajectory(events: List[Dict],
                              primitives: Dict,
                              fps: int = 60,
                              use_ik: bool = True) -> np.ndarray:
    """
    フェーズイベント列から DOF 軌道（CubicSpline 補間）を生成する。

    Returns:
        shape (N_frames, N_DOFS)
    """
    if not events:
        return np.zeros((0, N_DOFS))

    total_dur = events[-1]["t_end"]
    n_frames  = max(int(total_dur * fps), 1)

    # キーフレームをフェーズ中点に配置
    knot_t = []
    knot_q = []
    for ev in events:
        t_mid = (ev["t_start"] + ev["t_end"]) / 2.0
        q = resolve_symbols(ev["symbols"], primitives, use_ik)
        knot_t.append(t_mid)
        knot_q.append(q)

    # 始点・終点アンカー（clamped 条件用）
    knot_t = np.array([0.0] + knot_t + [total_dur])
    knot_q = np.vstack([knot_q[0], np.array(knot_q), knot_q[-1]])

    # 時刻が単調増加になるよう最小間隔を保証
    for i in range(1, len(knot_t)):
        if knot_t[i] <= knot_t[i - 1]:
            knot_t[i] = knot_t[i - 1] + 1e-3

    t_dense = np.linspace(0.0, total_dur, n_frames)
    traj = np.zeros((n_frames, N_DOFS))

    for j in range(N_DOFS):
        cs = CubicSpline(knot_t, knot_q[:, j], bc_type="clamped")
        traj[:, j] = cs(t_dense)

    return traj


def chain_trajectories(trajectories: List[np.ndarray],
                        blend_frames: int = 30) -> np.ndarray:
    """
    複数式の軌道をコサインイーズでブレンドしながら接続する。
    """
    if not trajectories:
        return np.zeros((0, N_DOFS))
    if len(trajectories) == 1:
        return trajectories[0]

    parts = [trajectories[0]]
    for curr in trajectories[1:]:
        prev = parts[-1]
        if blend_frames > 0 and len(prev) and len(curr):
            alpha = (1.0 - np.cos(np.pi * np.linspace(0, 1, blend_frames))) / 2.0
            blend = (prev[-1] * (1 - alpha[:, None]) +
                     curr[0]  *      alpha[:, None])
            parts.append(blend)
        parts.append(curr)

    return np.vstack(parts)


def mirror_trajectory(traj: np.ndarray) -> np.ndarray:
    """左右の式を鏡像にする。"""
    m = traj.copy()

    swap_pairs = [
        ("right_elbow",       "left_elbow"),
        ("right_knee",        "left_knee"),
        ("right_shoulder_rx", "left_shoulder_rx"),
        ("right_shoulder_ry", "left_shoulder_ry"),
        ("right_shoulder_rz", "left_shoulder_rz"),
        ("right_hip_rx",      "left_hip_rx"),
        ("right_hip_ry",      "left_hip_ry"),
        ("right_hip_rz",      "left_hip_rz"),
        ("right_ankle_rx",    "left_ankle_rx"),
        ("right_ankle_ry",    "left_ankle_ry"),
        ("right_ankle_rz",    "left_ankle_rz"),
    ]

    for a, b in swap_pairs:
        if a in DOF_INDEX and b in DOF_INDEX:
            ia, ib = DOF_INDEX[a], DOF_INDEX[b]
            m[:, ia] = traj[:, ib].copy()
            m[:, ib] = traj[:, ia].copy()

    # 側方（rx, rz）成分の符号を反転
    for name in ALL_DOF_NAMES:
        if name.endswith("_rx") or name.endswith("_rz"):
            m[:, DOF_INDEX[name]] *= -1.0

    return m


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.matrix_parser import load_forms, parse_form

    forms = load_forms("data/forms.json")
    prims = load_primitives("data/primitive_library.json")

    form = forms[1]  # 野馬分鬃
    events = parse_form(form)
    traj = generate_form_trajectory(events, prims, fps=60, use_ik=True)
    print(f"DOF数: {N_DOFS}")
    print(f"DOF名: {ALL_DOF_NAMES}")
    print(f"軌道形状: {traj.shape}  ({traj.shape[0]/60:.1f}秒 @60fps)")
