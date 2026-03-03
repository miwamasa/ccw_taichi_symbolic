"""
motion_generator.py - モーション軌道生成モジュール

記号マトリクスのフェーズイベント列から、関節角の時系列軌道（numpy配列）を生成する。
CubicSpline で補間し、各式の連続性を確保する。
式間の接続も担当する。
"""

import json
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Optional

from src.ik_solver import apply_ik_to_keyframe


# PyBullet humanoid.urdf の関節名リスト（既知の関節のみ使用）
# 実際のURDFに存在する関節名を使う（simulator.py でマッピングを行う）
ALL_JOINT_NAMES = [
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "right_ankle_x",
    "right_ankle_y",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "left_ankle_x",
    "left_ankle_y",
    "abdomen_z",
    "abdomen_x",
    "abdomen_y",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]

N_JOINTS = len(ALL_JOINT_NAMES)
JOINT_INDEX = {name: i for i, name in enumerate(ALL_JOINT_NAMES)}


def load_primitives(primitive_json_path: str) -> Dict:
    """プリミティブライブラリを読み込む。"""
    with open(primitive_json_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_symbols(symbols: Dict[str, str],
                    primitives: Dict,
                    use_ik: bool = True) -> np.ndarray:
    """
    記号辞書からターゲット関節角ベクトルを生成する。

    Args:
        symbols: {"歩型": "弓", "手法": "分", ...}
        primitives: primitive_library.json の内容
        use_ik: True の場合、手先IKで肩・肘角度を上書き

    Returns:
        joint_angles: shape (N_JOINTS,) の関節角ベクトル（ラジアン）
    """
    target = {}

    # 4つの主要記号要素からオフセットを加算
    for axis in ["歩型", "手法", "身法", "歩法"]:
        symbol = symbols.get(axis)
        if symbol and axis in primitives:
            if symbol in primitives[axis]:
                for joint, val in primitives[axis][symbol].items():
                    if not joint.startswith("_"):
                        target[joint] = target.get(joint, 0.0) + val

    # IK で手先位置から肩・肘角度を精密化
    if use_ik:
        hand_symbol = symbols.get("手法", "定")
        target = apply_ik_to_keyframe(target, hand_symbol)

    # 関節角ベクトルに変換
    q = np.zeros(N_JOINTS)
    for joint, val in target.items():
        if joint in JOINT_INDEX:
            q[JOINT_INDEX[joint]] = val

    return q


def get_timing_scale(symbols: Dict[str, str], primitives: Dict) -> float:
    """呼吸・意念記号からタイミングスケールを計算する。"""
    scale = 1.0
    for axis in ["呼吸", "意念"]:
        symbol = symbols.get(axis)
        if symbol and axis in primitives:
            if symbol in primitives[axis]:
                ts = primitives[axis][symbol].get("_timing_scale", 1.0)
                scale *= ts
    return scale


def generate_form_trajectory(events: List[Dict],
                              primitives: Dict,
                              fps: int = 60,
                              use_ik: bool = True) -> np.ndarray:
    """
    1つの式のフェーズイベント列から関節角軌道を生成する。

    CubicSpline のノットを各フェーズ中点に配置し、
    bc_type='clamped' でエンドポイントの速度を0にする。

    Args:
        events: parse_form() の出力
        primitives: load_primitives() の出力
        fps: サンプリング周波数
        use_ik: True の場合 IK を使用

    Returns:
        trajectory: shape (N_frames, N_JOINTS) の関節角時系列
    """
    if not events:
        return np.zeros((0, N_JOINTS))

    total_duration = events[-1]["t_end"]
    total_frames = max(int(total_duration * fps), 1)

    # キーフレームを各フェーズ中点に配置
    knot_times = []
    knot_angles = []

    for event in events:
        t_mid = (event["t_start"] + event["t_end"]) / 2.0
        q = resolve_symbols(event["symbols"], primitives, use_ik)
        knot_times.append(t_mid)
        knot_angles.append(q)

    # 始点・終点にアンカーを追加（零速度条件のため）
    knot_times = [0.0] + knot_times + [total_duration]
    knot_angles = [knot_angles[0]] + knot_angles + [knot_angles[-1]]

    knot_times = np.array(knot_times)
    knot_angles = np.array(knot_angles)  # (K, N_JOINTS)

    # 時刻の重複を避ける（最小間隔を確保）
    for i in range(1, len(knot_times)):
        if knot_times[i] <= knot_times[i - 1]:
            knot_times[i] = knot_times[i - 1] + 1e-3

    # 評価時刻
    t_dense = np.linspace(0.0, total_duration, total_frames)

    # 各関節について CubicSpline で補間
    trajectory = np.zeros((total_frames, N_JOINTS))
    for j in range(N_JOINTS):
        cs = CubicSpline(knot_times, knot_angles[:, j], bc_type="clamped")
        trajectory[:, j] = cs(t_dense)

    return trajectory


def chain_trajectories(trajectories: List[np.ndarray],
                        blend_frames: int = 30) -> np.ndarray:
    """
    複数の式の軌道を滑らかに接続する。

    式間のブレンドにコサインイーズを使い速度不連続を防ぐ。

    Args:
        trajectories: 各式の trajectory (shape: (N_i, N_JOINTS))
        blend_frames: ブレンドに使うフレーム数（デフォルト30フレーム=0.5秒@60fps）

    Returns:
        full_trajectory: shape (Total_frames, N_JOINTS)
    """
    if not trajectories:
        return np.zeros((0, N_JOINTS))

    if len(trajectories) == 1:
        return trajectories[0]

    result = [trajectories[0]]

    for i in range(1, len(trajectories)):
        prev = trajectories[i - 1]
        curr = trajectories[i]

        if blend_frames > 0 and len(prev) > 0 and len(curr) > 0:
            last_frame = prev[-1]
            first_frame = curr[0]

            # コサインイーズでブレンド
            alpha = np.linspace(0.0, 1.0, blend_frames)
            alpha = (1.0 - np.cos(np.pi * alpha)) / 2.0  # smoothstep

            blend = (
                last_frame[np.newaxis, :] * (1.0 - alpha[:, np.newaxis]) +
                first_frame[np.newaxis, :] * alpha[:, np.newaxis]
            )
            result.append(blend)

        result.append(curr)

    return np.vstack(result)


def mirror_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """
    左右の式を鏡像にする（左右対称な動作の生成に使用）。

    左右の関節を入れ替え、側方成分の符号を反転する。
    """
    mirrored = trajectory.copy()

    # 左右ペアを定義
    swap_pairs = [
        ("right_hip_x",    "left_hip_x"),
        ("right_hip_z",    "left_hip_z"),
        ("right_hip_y",    "left_hip_y"),
        ("right_knee",     "left_knee"),
        ("right_ankle_x",  "left_ankle_x"),
        ("right_ankle_y",  "left_ankle_y"),
        ("right_shoulder1","left_shoulder1"),
        ("right_shoulder2","left_shoulder2"),
        ("right_elbow",    "left_elbow"),
    ]

    for r_name, l_name in swap_pairs:
        if r_name in JOINT_INDEX and l_name in JOINT_INDEX:
            ri = JOINT_INDEX[r_name]
            li = JOINT_INDEX[l_name]
            mirrored[:, ri] = trajectory[:, li].copy()
            mirrored[:, li] = trajectory[:, ri].copy()

    # 左右方向（x, z 回旋成分）の符号を反転
    lateral_joints = ["right_hip_x", "left_hip_x", "right_hip_z", "left_hip_z",
                      "abdomen_z", "right_shoulder2", "left_shoulder2"]
    for name in lateral_joints:
        if name in JOINT_INDEX:
            mirrored[:, JOINT_INDEX[name]] *= -1.0

    return mirrored


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.matrix_parser import load_forms, parse_form

    forms_path = "data/forms.json"
    prim_path  = "data/primitive_library.json"

    forms = load_forms(forms_path)
    primitives = load_primitives(prim_path)

    # 最初の式でテスト
    form = forms[1]  # 野馬分鬃
    events = parse_form(form)

    print(f"テスト: 式{form['id']}「{form['name']}」")
    print(f"  フェーズ数: {len(events)}")

    traj = generate_form_trajectory(events, primitives, fps=60, use_ik=True)
    print(f"  軌道形状: {traj.shape}")
    print(f"  フレーム数: {traj.shape[0]} ({traj.shape[0]/60:.1f}秒 @60fps)")
    print(f"  関節数: {traj.shape[1]}")
    print(f"  関節名: {ALL_JOINT_NAMES[:5]}...")
