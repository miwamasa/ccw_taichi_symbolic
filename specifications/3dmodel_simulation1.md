了解です。
**A）PyBullet + Python の最小プロトタイプ**を、
「パターン・マトリクス JSON → 角度列生成 → PyBulletで再生」まで一気通しで提示します。

---

# 全体構成

```
project/
 ├─ matrix.json
 ├─ primitive_library.json
 └─ simulate.py
```

依存：

```
pip install pybullet numpy scipy
```

---

# 1️⃣ matrix.json（例：野馬分鬃）

```json
{
  "meta": {"height_m":1.70, "mass_kg":65, "fps":60},
  "columns": [
    {"duration":0.6,"歩型":"虚","歩法":"起","手法":"抱","身法":"含"},
    {"duration":0.8,"歩型":"虚","歩法":"移","手法":"掤","身法":"転"},
    {"duration":0.6,"歩型":"中","歩法":"送","手法":"分","身法":"旋"},
    {"duration":0.7,"歩型":"弓","歩法":"進","手法":"分","身法":"開"},
    {"duration":0.6,"歩型":"弓","歩法":"定","手法":"按","身法":"沈"},
    {"duration":0.4,"歩型":"弓","歩法":"定","手法":"定","身法":"立"}
  ]
}
```

---

# 2️⃣ primitive_library.json（超簡易版）

※ 角度はラジアン

```json
{
  "歩型": {
    "虚": {"hip": 0.05, "knee": 0.2},
    "中": {"hip": 0.1,  "knee": 0.3},
    "弓": {"hip": 0.2,  "knee": 0.5}
  },
  "手法": {
    "抱": {"shoulder": 0.2, "elbow": -0.5},
    "掤": {"shoulder": 0.3, "elbow": -0.3},
    "分": {"shoulder": 0.6, "elbow": -0.2},
    "按": {"shoulder": 0.1, "elbow": -0.1},
    "定": {"shoulder": 0.0, "elbow": 0.0}
  },
  "身法": {
    "含": {"torso": 0.05},
    "転": {"torso": 0.15},
    "旋": {"torso": 0.3},
    "開": {"torso": 0.2},
    "沈": {"torso": 0.1},
    "立": {"torso": 0.0}
  }
}
```

これは最小の「象徴→角度オフセット」辞書です。

---

# 3️⃣ simulate.py（完全実行コード）

```python
import pybullet as p
import pybullet_data
import numpy as np
import json
import time
from scipy.interpolate import CubicSpline

# --- Load data ---
with open("matrix.json") as f:
    matrix = json.load(f)

with open("primitive_library.json") as f:
    primitives = json.load(f)

fps = matrix["meta"]["fps"]
dt = 1.0 / fps

# --- Build trajectory ---
joint_names = ["hip","knee","shoulder","elbow","torso"]
trajectory = []

for col in matrix["columns"]:
    duration = col["duration"]
    steps = int(duration * fps)

    target = dict.fromkeys(joint_names, 0.0)

    for axis in ["歩型","手法","身法"]:
        symbol = col.get(axis)
        if symbol and symbol in primitives[axis]:
            for j,v in primitives[axis][symbol].items():
                target[j] += v

    for _ in range(steps):
        trajectory.append([target[j] for j in joint_names])

trajectory = np.array(trajectory)

# --- Smooth trajectory ---
t = np.arange(len(trajectory))
smooth = []
for i in range(trajectory.shape[1]):
    cs = CubicSpline(t, trajectory[:,i])
    smooth.append(cs(t))
trajectory = np.stack(smooth, axis=1)

# --- PyBullet setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

robot = p.loadURDF("humanoid/humanoid.urdf", [0,0,1])

p.setGravity(0,0,-9.8)

joint_indices = []
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot,i)
    joint_indices.append(i)

# --- Playback ---
for frame in trajectory:
    for i,joint_index in enumerate(joint_indices[:len(frame)]):
        p.setJointMotorControl2(robot,
                                joint_index,
                                p.POSITION_CONTROL,
                                targetPosition=frame[i],
                                force=200)

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()
```

---

# 🔎 これで何が起きるか

* マトリクスを時間展開
* 記号 → 関節角オフセットへ変換
* CubicSplineで平滑化
* PyBulletのhumanoidに角度入力
* GUIで再生

---

# ⚠️ 限界（今の最小版）

* IK未使用（単純角度直接指定）
* バランス制御なし
* 接地判定なし
* 足運び未実装

→ ただし「象徴マトリクスから3D動作生成」の最短動線としては十分

---

# 🚀 次段階アップグレード案

1. IK導入（手先位置目標化）
2. 重心制御（CoM安定）
3. 足接地フェーズ導入
4. MuJoCo移行（精密物理）

---

もしよければ次は：

* 🧠 本当に安定させる「CoM制御付き版」
* 🦶 歩法を物理的に正しく実装
* 📊 モーションをBVHに出力
* 🧬 機械学習で記号→角度を学習

どこまで攻めますか？
