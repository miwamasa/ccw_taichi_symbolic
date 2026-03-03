"""
simulator.py - PyBullet シミュレータモジュール

humanoid.urdf の実際の関節構造に対応:
  SPHERICAL 関節 (7個): chest, right_shoulder, left_shoulder,
                         right_hip, right_ankle, left_hip, left_ankle
  REVOLUTE  関節 (4個): right_elbow, left_elbow, right_knee, left_knee
  FIXED     関節 (3個): root, right_wrist, left_wrist (制御不要)

球面関節は setJointMotorControlMultiDof + クォータニオンで制御。
useFixedBase=True で重力落下を防ぎ、直立姿勢を維持する。
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
    print("[WARNING] pybullet がインストールされていません: pip install pybullet")

try:
    from src.motion_generator import (
        ALL_DOF_NAMES, DOF_INDEX, N_DOFS,
        REVOLUTE_DOF_NAMES, SPHERICAL_JOINT_NAMES, JOINT_IDX
    )
except ImportError:
    from motion_generator import (
        ALL_DOF_NAMES, DOF_INDEX, N_DOFS,
        REVOLUTE_DOF_NAMES, SPHERICAL_JOINT_NAMES, JOINT_IDX
    )


def euler_to_quaternion(rx: float, ry: float, rz: float):
    """
    euler 角（ラジアン）をクォータニオン [qx, qy, qz, qw] に変換する。
    """
    return p.getQuaternionFromEuler([rx, ry, rz])


class TaichiSimulator:
    """
    PyBullet を使った太極拳モーションシミュレータ。

    使用例::

        sim = TaichiSimulator(use_gui=True)
        sim.setup()
        sim.play(trajectory, fps=60)
        sim.close()
    """

    # 球面関節のPDゲイン（力: 各軸 N・m）
    SPHERICAL_KP    = 0.5
    SPHERICAL_KD    = 0.05
    SPHERICAL_FORCE = 100.0

    # 回転関節のPDゲイン
    REVOLUTE_KP    = 0.5
    REVOLUTE_KD    = 0.05
    REVOLUTE_FORCE = 100.0

    # 内部シミュレーション周波数
    SIM_HZ = 240

    def __init__(self, use_gui: bool = True):
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("pybullet が必要です: pip install pybullet")
        self.use_gui = use_gui
        self.robot_id: Optional[int] = None
        self._connected = False

    def setup(self, urdf_path: Optional[str] = None):
        """PyBullet を起動し、シーンと humanoid をセットアップする。"""
        if self.use_gui:
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)
        self._connected = True

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.SIM_HZ)

        # 地面
        p.loadURDF("plane.urdf")

        if urdf_path is None:
            urdf_path = "humanoid/humanoid.urdf"

        # useFixedBase=True: ルートリンクを固定し、重力落下を防ぐ
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.97],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # カメラ設定（斜め前から見る）
        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=45,
                cameraPitch=-15,
                cameraTargetPosition=[0, 0, 1.0],
            )

        self._print_joint_info()
        print(f"[INFO] Humanoid ロード完了 (useFixedBase=True, z=0.97m)")

    def _print_joint_info(self):
        """デバッグ用: 全関節情報を表示。"""
        type_names = {0: "REVOLUTE", 1: "PRISMATIC", 2: "SPHERICAL",
                      3: "PLANAR",   4: "FIXED"}
        print(f"\n{'Idx':>4}  {'Name':<25}  {'Type':<10}  {'Lo':>7}  {'Hi':>7}")
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name  = info[1].decode()
            jtype = type_names.get(info[2], str(info[2]))
            lo, hi = info[8], info[9]
            print(f"{i:>4}  {name:<25}  {jtype:<10}  {lo:>7.3f}  {hi:>7.3f}")
        print()

    def apply_dof_vector(self, dof_vec: np.ndarray):
        """
        DOF ベクトル（25次元）を PyBullet 関節コントローラに設定する。

        - REVOLUTE 関節: setJointMotorControl2 (スカラー角度)
        - SPHERICAL 関節: setJointMotorControlMultiDof (クォータニオン)
        """
        # ── REVOLUTE 関節 ──────────────────────────────────────
        for dof_name in REVOLUTE_DOF_NAMES:
            if dof_name not in JOINT_IDX or dof_name not in DOF_INDEX:
                continue
            joint_idx = JOINT_IDX[dof_name]
            angle     = float(dof_vec[DOF_INDEX[dof_name]])
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=self.REVOLUTE_KP,
                velocityGain=self.REVOLUTE_KD,
                force=self.REVOLUTE_FORCE,
            )

        # ── SPHERICAL 関節 ─────────────────────────────────────
        for joint_name in SPHERICAL_JOINT_NAMES:
            if joint_name not in JOINT_IDX:
                continue
            joint_idx = JOINT_IDX[joint_name]

            # euler 3成分を取得
            rx_key = f"{joint_name}_rx"
            ry_key = f"{joint_name}_ry"
            rz_key = f"{joint_name}_rz"
            rx = float(dof_vec[DOF_INDEX[rx_key]]) if rx_key in DOF_INDEX else 0.0
            ry = float(dof_vec[DOF_INDEX[ry_key]]) if ry_key in DOF_INDEX else 0.0
            rz = float(dof_vec[DOF_INDEX[rz_key]]) if rz_key in DOF_INDEX else 0.0

            quat = euler_to_quaternion(rx, ry, rz)

            p.setJointMotorControlMultiDof(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=quat,
                targetVelocity=[0.0, 0.0, 0.0],
                positionGain=self.SPHERICAL_KP,
                velocityGain=self.SPHERICAL_KD,
                force=[self.SPHERICAL_FORCE] * 3,
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
            trajectory: shape (N_frames, N_DOFS) の DOF 時系列
            fps: 再生フレームレート
            form_start_frames: 各式の開始フレームインデックス（表示用）
            form_names: 各式の名前（表示用）
            loop: True の場合ループ再生
        """
        if self.robot_id is None:
            raise RuntimeError("setup() を先に呼んでください")

        dt               = 1.0 / fps
        steps_per_frame  = max(1, int(self.SIM_HZ / fps))
        form_frame_set   = set(form_start_frames) if form_start_frames else set()
        n_frames         = trajectory.shape[0]

        print(f"[INFO] 再生開始: {n_frames} フレーム "
              f"({n_frames/fps:.1f}秒 @{fps}fps)")

        try:
            while True:
                start_wall = time.time()
                for fi in range(n_frames):
                    # 式切り替わりを表示
                    if form_start_frames and fi in form_frame_set:
                        idx = form_start_frames.index(fi)
                        name = form_names[idx] if form_names else f"式{idx+1}"
                        t_sec = fi / fps
                        print(f"  [{t_sec:6.1f}s] {name}")

                    self.apply_dof_vector(trajectory[fi])

                    for _ in range(steps_per_frame):
                        p.stepSimulation()

                    # 実時間同期（GUI モードのみ）
                    if self.use_gui:
                        target_t = start_wall + (fi + 1) * dt
                        sleep_t  = target_t - time.time()
                        if sleep_t > 0:
                            time.sleep(sleep_t)

                if not loop:
                    break
                print("[INFO] ループ再生...")

        except KeyboardInterrupt:
            print("\n[INFO] 再生を中断しました")

        print("[INFO] 再生完了")

    def close(self):
        """PyBullet 接続を切断する。"""
        if self._connected:
            p.disconnect(self._client)
            self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def check_urdf_joints(urdf_path: Optional[str] = None):
    """URDF の全関節名を表示するデバッグ用関数。"""
    if not PYBULLET_AVAILABLE:
        print("pybullet が利用できません")
        return

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if urdf_path is None:
        urdf_path = "humanoid/humanoid.urdf"

    robot = p.loadURDF(urdf_path)
    type_names = {0:"REVOLUTE", 1:"PRISMATIC", 2:"SPHERICAL",
                  3:"PLANAR",   4:"FIXED"}

    print(f"\n=== {urdf_path} の関節一覧 ===")
    print(f"{'Idx':>4}  {'Name':<25}  {'Type':<10}  {'Lower':>8}  {'Upper':>8}")
    print("-" * 65)
    for i in range(p.getNumJoints(robot)):
        info  = p.getJointInfo(robot, i)
        name  = info[1].decode()
        jtype = type_names.get(info[2], str(info[2]))
        lo, hi = info[8], info[9]
        print(f"{i:>4}  {name:<25}  {jtype:<10}  {lo:>8.3f}  {hi:>8.3f}")

    p.disconnect(client)


if __name__ == "__main__":
    check_urdf_joints()
