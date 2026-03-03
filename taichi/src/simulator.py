"""
simulator.py - PyBullet シミュレータモジュール

PyBullet を使って humanoid モデルに関節角軌道を再生する。
PDポジションコントロールで各関節を駆動する。
"""

import time
import numpy as np
from typing import Dict, List, Optional

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[WARNING] pybullet がインストールされていません。シミュレーションは実行できません。")

from src.motion_generator import ALL_JOINT_NAMES, JOINT_INDEX


# PyBullet humanoid.urdf の関節名からインデックスへのマッピング
# 実行時に build_joint_map() で動的に構築する
_JOINT_MAP: Optional[Dict[str, int]] = None

# primitive_library.json のキー名 → PyBullet URDF 関節名マッピング
# （humanoid.urdf を確認して調整が必要な場合がある）
CANONICAL_TO_URDF = {
    "right_hip_x":    ["right_hip_x"],
    "right_hip_z":    ["right_hip_z"],
    "right_hip_y":    ["right_hip_y"],
    "right_knee":     ["right_knee"],
    "right_ankle_x":  ["right_ankle_x"],
    "right_ankle_y":  ["right_ankle_y"],
    "left_hip_x":     ["left_hip_x"],
    "left_hip_z":     ["left_hip_z"],
    "left_hip_y":     ["left_hip_y"],
    "left_knee":      ["left_knee"],
    "left_ankle_x":   ["left_ankle_x"],
    "left_ankle_y":   ["left_ankle_y"],
    "abdomen_z":      ["abdomen_z"],
    "abdomen_x":      ["abdomen_x"],
    "abdomen_y":      ["abdomen_y"],
    "right_shoulder1":["right_shoulder1"],
    "right_shoulder2":["right_shoulder2"],
    "right_elbow":    ["right_elbow"],
    "left_shoulder1": ["left_shoulder1"],
    "left_shoulder2": ["left_shoulder2"],
    "left_elbow":     ["left_elbow"],
}


def build_joint_map(robot_id: int) -> Dict[str, int]:
    """
    ロードした humanoid URDF の全関節名→インデックスマッピングを構築する。

    Args:
        robot_id: PyBullet の robot body ID

    Returns:
        {"joint_name": index, ...}
    """
    name_to_idx = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            name_to_idx[joint_name] = i

    print(f"[INFO] 可動関節数: {len(name_to_idx)}")
    if len(name_to_idx) < 5:
        print("[INFO] 全関節一覧:")
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            print(f"  [{i}] {info[1].decode()}: type={info[2]}")

    return name_to_idx


def resolve_joint_indices(joint_map: Dict[str, int]) -> Dict[str, int]:
    """
    canonical 名から実際の PyBullet 関節インデックスを解決する。

    URDF の実際の関節名と canonical 名が異なる場合に部分一致で探す。
    """
    canonical_to_idx = {}

    for canonical, urdf_candidates in CANONICAL_TO_URDF.items():
        found = False
        for candidate in urdf_candidates:
            if candidate in joint_map:
                canonical_to_idx[canonical] = joint_map[candidate]
                found = True
                break
        if not found:
            # 部分一致フォールバック
            for actual_name, idx in joint_map.items():
                if canonical.lower() in actual_name.lower():
                    canonical_to_idx[canonical] = idx
                    found = True
                    break

    mapped = len(canonical_to_idx)
    total = len(CANONICAL_TO_URDF)
    print(f"[INFO] 関節マッピング: {mapped}/{total} canonical 名を解決")
    unmapped = [k for k in CANONICAL_TO_URDF if k not in canonical_to_idx]
    if unmapped:
        print(f"[WARN] 未解決の関節: {unmapped}")

    return canonical_to_idx


class TaichiSimulator:
    """
    PyBullet を使った太極拳モーションシミュレータ。

    使用例::

        sim = TaichiSimulator(use_gui=True)
        sim.setup()
        sim.play(trajectory, fps=60)
        sim.close()
    """

    def __init__(self, use_gui: bool = True):
        """
        Args:
            use_gui: True の場合 GUI モード（可視化あり）
        """
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("pybullet がインストールされていません: pip install pybullet")

        self.use_gui = use_gui
        self.robot_id: Optional[int] = None
        self.canonical_to_idx: Dict[str, int] = {}
        self._connected = False

        # PDコントロールゲイン
        self.position_gain = 0.5
        self.velocity_gain = 0.1
        self.max_force = 200.0

        # 内部シミュレーション周波数（PyBullet デフォルト）
        self.sim_hz = 240

    def setup(self, urdf_path: Optional[str] = None):
        """
        PyBullet を起動してシーンをセットアップする。

        Args:
            urdf_path: humanoid URDF のパス（None の場合 pybullet_data の humanoid を使用）
        """
        if self.use_gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)

        self._connected = True

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.sim_hz)

        # 地面
        self._plane_id = p.loadURDF("plane.urdf")

        # Humanoid
        if urdf_path is None:
            urdf_path = "humanoid/humanoid.urdf"

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.95],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
        )

        # 関節マップを構築
        joint_map = build_joint_map(self.robot_id)
        self.canonical_to_idx = resolve_joint_indices(joint_map)

        # カメラ設定
        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=30,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 1.0]
            )

        print(f"[INFO] PyBullet シミュレータ起動完了")
        print(f"[INFO] Humanoid: {self.robot_id}, 地面: {self._plane_id}")

    def apply_joint_angles(self, joint_angle_vec: np.ndarray):
        """
        関節角ベクトルを PyBullet の PD コントローラに設定する。

        Args:
            joint_angle_vec: shape (N_JOINTS,) の関節角ベクトル（ラジアン）
        """
        for canonical_name, idx in self.canonical_to_idx.items():
            if canonical_name in JOINT_INDEX:
                angle = float(joint_angle_vec[JOINT_INDEX[canonical_name]])
                p.setJointMotorControl2(
                    self.robot_id,
                    idx,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    positionGain=self.position_gain,
                    velocityGain=self.velocity_gain,
                    force=self.max_force,
                )

    def play(self,
             trajectory: np.ndarray,
             fps: int = 60,
             form_start_frames: Optional[List[int]] = None,
             form_names: Optional[List[str]] = None,
             loop: bool = False):
        """
        軌道を再生する。

        Args:
            trajectory: shape (N_frames, N_JOINTS) の関節角時系列
            fps: 再生フレームレート
            form_start_frames: 各式の開始フレームインデックス（オプション）
            form_names: 各式の名前（表示用オプション）
            loop: True の場合ループ再生
        """
        if self.robot_id is None:
            raise RuntimeError("setup() を先に呼んでください")

        dt = 1.0 / fps
        steps_per_frame = max(1, int(self.sim_hz / fps))

        # 式名の表示用セット
        form_frame_set = set(form_start_frames) if form_start_frames else set()
        form_idx = 0

        n_frames = trajectory.shape[0]
        total_time = n_frames / fps
        print(f"[INFO] 再生開始: {n_frames} フレーム ({total_time:.1f}秒 @{fps}fps)")

        try:
            while True:
                start_wall = time.time()

                for frame_idx in range(n_frames):
                    # 式の切り替わりを表示
                    if form_start_frames and frame_idx in form_frame_set:
                        idx_in_list = form_start_frames.index(frame_idx)
                        name = form_names[idx_in_list] if form_names else f"式{idx_in_list+1}"
                        print(f"  [{frame_idx/fps:.1f}s] {name}")

                    # 関節角を設定
                    self.apply_joint_angles(trajectory[frame_idx])

                    # シミュレーションステップ
                    for _ in range(steps_per_frame):
                        p.stepSimulation()

                    # 実時間同期
                    if self.use_gui:
                        elapsed = time.time() - start_wall - frame_idx * dt
                        sleep_time = dt - elapsed % dt
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                if not loop:
                    break
                print("[INFO] ループ再生...")

        except KeyboardInterrupt:
            print("\n[INFO] 再生を中断しました")

        print("[INFO] 再生完了")

    def close(self):
        """PyBullet 接続を切断する。"""
        if self._connected:
            p.disconnect(self._physics_client)
            self._connected = False
            print("[INFO] PyBullet 接続を切断しました")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def check_urdf_joints(urdf_path: Optional[str] = None):
    """
    URDF の全関節名を表示するデバッグ用関数。
    初回実行時にどの関節名が利用可能かを確認するために使う。
    """
    if not PYBULLET_AVAILABLE:
        print("pybullet が利用できません")
        return

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if urdf_path is None:
        urdf_path = "humanoid/humanoid.urdf"

    robot = p.loadURDF(urdf_path)

    print(f"\n=== {urdf_path} の関節一覧 ===")
    print(f"{'Index':>6}  {'Name':<30}  {'Type':<12}  {'Lower':>8}  {'Upper':>8}")
    print("-" * 70)

    type_names = {
        0: "REVOLUTE",
        1: "PRISMATIC",
        2: "SPHERICAL",
        3: "PLANAR",
        4: "FIXED",
    }

    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        idx = info[0]
        name = info[1].decode("utf-8")
        jtype = type_names.get(info[2], str(info[2]))
        lower = info[8]
        upper = info[9]
        if info[2] != 4:  # FIXED でないもの
            print(f"{idx:>6}  {name:<30}  {jtype:<12}  {lower:>8.3f}  {upper:>8.3f}")

    p.disconnect(client)


if __name__ == "__main__":
    print("PyBullet humanoid.urdf 関節確認:")
    check_urdf_joints()
