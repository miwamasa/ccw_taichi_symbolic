#!/usr/bin/env python3
"""
run.py - 太極拳24式シミュレーション 起動スクリプト

使用方法:
    python run.py                    # 全24式を連続シミュレーション
    python run.py --form 2           # 野馬分鬃のみ
    python run.py --forms 1-9        # 最初の9式
    python run.py --forms 1,2,7      # 指定した式のみ
    python run.py --no-gui           # 軌道生成のみ（表示なし）
    python run.py --check-joints     # URDF の関節名を確認
    python run.py --validate         # データファイルを検証
    python run.py --help             # ヘルプを表示
"""

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    main()
