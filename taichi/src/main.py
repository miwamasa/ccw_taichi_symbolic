"""
main.py - 太極拳24式シミュレーション メインエントリポイント

全24式のモーションを生成してシミュレーションを実行する。

使用方法::

    # 全24式を連続シミュレーション
    python -m src.main

    # 特定の式のみシミュレーション
    python -m src.main --form 2

    # 最初の N 式をシミュレーション
    python -m src.main --forms 1-9

    # GUI なし（軌道生成のみ）
    python -m src.main --no-gui

    # URDF 関節確認
    python -m src.main --check-joints
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# パスを追加（taichi/ ディレクトリからの実行に対応）
_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

from src.matrix_parser import load_forms, parse_form, validate_forms
from src.motion_generator import (
    load_primitives,
    generate_form_trajectory,
    chain_trajectories,
    mirror_trajectory,
    ALL_DOF_NAMES,
)


# データファイルパス
DATA_DIR = _HERE / "data"
FORMS_JSON   = DATA_DIR / "forms.json"
PRIMS_JSON   = DATA_DIR / "primitive_library.json"

# 式間のデフォルトブレンドフレーム数（@60fps で 0.5秒）
DEFAULT_BLEND_FRAMES = 30

# 鏡像にする式の ID リスト（左右対称な繰り返し動作など）
MIRROR_FORM_IDS = {10, 14, 16, 17, 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="太極拳24式モーションシミュレーション"
    )
    parser.add_argument(
        "--form", type=int, default=None,
        help="単一の式を再生（式番号 1-24）"
    )
    parser.add_argument(
        "--forms", type=str, default=None,
        help="範囲指定（例: '1-9'）または カンマ区切り（例: '1,2,7'）"
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="再生フレームレート（デフォルト: 60）"
    )
    parser.add_argument(
        "--blend", type=int, default=DEFAULT_BLEND_FRAMES,
        help=f"式間ブレンドフレーム数（デフォルト: {DEFAULT_BLEND_FRAMES}）"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="GUI なし（軌道生成のみ）"
    )
    parser.add_argument(
        "--no-ik", action="store_true",
        help="IK を使わない（プリミティブ直接使用）"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="ループ再生"
    )
    parser.add_argument(
        "--check-joints", action="store_true",
        help="URDF の関節名を確認して終了"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="データファイルを検証して終了"
    )
    parser.add_argument(
        "--urdf", type=str, default=None,
        help="humanoid URDF パス（デフォルト: pybullet_data 内の humanoid.urdf）"
    )
    return parser.parse_args()


def parse_form_range(spec: str, max_id: int) -> List[int]:
    """
    フォーム範囲指定文字列をIDリストに変換する。

    Examples:
        "1-9"   -> [1,2,3,4,5,6,7,8,9]
        "1,3,7" -> [1,3,7]
        "1-3,7" -> [1,2,3,7]
    """
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo.strip()), int(hi.strip())
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(part))
    return [i for i in ids if 1 <= i <= max_id]


def build_trajectory(
    forms: List[dict],
    target_ids: List[int],
    primitives: dict,
    fps: int,
    blend_frames: int,
    use_ik: bool,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    指定された式IDリストから連続モーション軌道を生成する。

    Returns:
        (trajectory, form_start_frames, form_names)
    """
    form_by_id = {f["id"]: f for f in forms}

    trajectories = []
    form_start_frames = []
    form_names = []
    current_frame = 0

    print(f"\n[軌道生成] {len(target_ids)} 式を処理中...")

    for fid in target_ids:
        if fid not in form_by_id:
            print(f"[WARN] 式{fid} が見つかりません。スキップします。")
            continue

        form = form_by_id[fid]
        name = form.get("name", f"式{fid}")
        events = parse_form(form)

        print(f"  式{fid:02d}「{name}」: "
              f"{len(events)}フェーズ, "
              f"{events[-1]['t_end']:.1f}秒")

        traj = generate_form_trajectory(
            events, primitives, fps=fps, use_ik=use_ik
        )

        # 鏡像処理
        if fid in MIRROR_FORM_IDS:
            traj = mirror_trajectory(traj)
            print(f"    -> 鏡像適用")

        form_start_frames.append(current_frame)
        form_names.append(f"式{fid:02d}「{name}」")
        trajectories.append(traj)

        # ブレンドフレーム + 式のフレーム数
        current_frame += traj.shape[0]
        if len(trajectories) > 1:
            current_frame += blend_frames

    if not trajectories:
        print("[ERROR] 軌道が1つも生成されませんでした")
        return np.zeros((0, len(ALL_DOF_NAMES))), [], []

    # 全式を接続
    print(f"\n[接続] {len(trajectories)} 式を接続中...")
    full_trajectory = chain_trajectories(trajectories, blend_frames=blend_frames)

    total_sec = full_trajectory.shape[0] / fps
    print(f"[完成] 合計フレーム数: {full_trajectory.shape[0]} ({total_sec:.1f}秒)")

    return full_trajectory, form_start_frames, form_names


def main():
    args = parse_args()

    # --validate: データ検証のみ
    if args.validate:
        print(f"[検証] {FORMS_JSON}")
        ok = validate_forms(str(FORMS_JSON))
        if ok:
            print("[OK] データ検証完了")
        else:
            print("[NG] 検証エラーあり")
            sys.exit(1)
        return

    # --check-joints: URDF 関節確認のみ
    if args.check_joints:
        from src.simulator import check_urdf_joints
        check_urdf_joints(args.urdf)
        return

    # データ読み込み
    if not FORMS_JSON.exists():
        print(f"[ERROR] forms.json が見つかりません: {FORMS_JSON}")
        sys.exit(1)
    if not PRIMS_JSON.exists():
        print(f"[ERROR] primitive_library.json が見つかりません: {PRIMS_JSON}")
        sys.exit(1)

    forms = load_forms(str(FORMS_JSON))
    primitives = load_primitives(str(PRIMS_JSON))

    print(f"[INFO] {len(forms)} 式を読み込みました")

    # 対象式IDを決定
    if args.form is not None:
        target_ids = [args.form]
    elif args.forms is not None:
        target_ids = parse_form_range(args.forms, max_id=len(forms))
    else:
        target_ids = list(range(1, len(forms) + 1))

    print(f"[INFO] 対象式: {target_ids}")

    # 軌道生成
    use_ik = not args.no_ik
    trajectory, form_start_frames, form_names = build_trajectory(
        forms, target_ids, primitives,
        fps=args.fps,
        blend_frames=args.blend,
        use_ik=use_ik,
    )

    if trajectory.shape[0] == 0:
        print("[ERROR] 軌道が空です")
        sys.exit(1)

    # GUI なし の場合はここで終了
    if args.no_gui:
        print("[INFO] --no-gui モード: 軌道生成のみ完了")
        print(f"[INFO] 軌道形状: {trajectory.shape}")
        return

    # PyBullet シミュレーション
    from src.simulator import TaichiSimulator

    with TaichiSimulator(use_gui=True) as sim:
        sim.setup(urdf_path=args.urdf)
        sim.play(
            trajectory,
            fps=args.fps,
            form_start_frames=form_start_frames,
            form_names=form_names,
            loop=args.loop,
        )


if __name__ == "__main__":
    main()
