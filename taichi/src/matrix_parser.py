"""
matrix_parser.py - 記号マトリクスのパーサ

forms.json から式を読み込み、各フェーズを時刻付きイベントとして出力する。
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_forms(forms_json_path: str) -> List[Dict]:
    """全24式のマトリクスを読み込む。"""
    with open(forms_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["forms"]


def parse_form(form: Dict) -> List[Dict]:
    """
    1つの式を時刻付きフェーズイベント列に変換する。

    Returns:
        [
            {
                "t_start": 0.0,
                "t_end": 0.6,
                "phase": "起",
                "duration": 0.6,
                "symbols": {
                    "歩型": "虚",
                    "歩法": "起",
                    "手法": "抱",
                    "身法": "含",
                    "呼吸": "吸",
                    "意念": "集"
                }
            },
            ...
        ]
    """
    events = []
    t = 0.0
    for col in form["columns"]:
        duration = col["duration"]
        event = {
            "t_start": t,
            "t_end": t + duration,
            "phase": col.get("phase", ""),
            "duration": duration,
            "symbols": {
                "歩型": col.get("歩型", "中"),
                "歩法": col.get("歩法", "定"),
                "手法": col.get("手法", "定"),
                "身法": col.get("身法", "立"),
                "呼吸": col.get("呼吸", "静"),
                "意念": col.get("意念", "静"),
            }
        }
        events.append(event)
        t += duration
    return events


def get_form_duration(events: List[Dict]) -> float:
    """フェーズイベント列の合計時間を返す。"""
    if not events:
        return 0.0
    return events[-1]["t_end"]


def validate_forms(forms_json_path: str) -> bool:
    """全式のデータを検証する。"""
    forms = load_forms(forms_json_path)
    valid = True
    for form in forms:
        fid = form.get("id")
        name = form.get("name", "?")
        cols = form.get("columns", [])
        if len(cols) != 6:
            print(f"[WARN] 式{fid}「{name}」: フェーズ数が{len(cols)}（期待値6）")
            valid = False
        for col in cols:
            for key in ["歩型", "手法", "身法"]:
                if key not in col:
                    print(f"[WARN] 式{fid}「{name}」フェーズ「{col.get('phase','')}」: {key} が未定義")
    return valid


if __name__ == "__main__":
    import sys
    forms_path = sys.argv[1] if len(sys.argv) > 1 else "data/forms.json"
    print(f"[検証] {forms_path}")
    validate_forms(forms_path)
    forms = load_forms(forms_path)
    print(f"[OK] {len(forms)} 式を読み込みました")
    for form in forms[:3]:
        events = parse_form(form)
        print(f"  式{form['id']}「{form['name']}」: {len(events)}フェーズ, "
              f"合計{get_form_duration(events):.1f}秒")
