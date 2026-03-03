いいですね — 実装できそうな設計を一気に出します。質問の要件を踏まえ、

* 縦軸（6要素：歩型・歩法・手法・身法・呼吸・意念）＋**不足している縦軸要素**を明確化
* 「パターン・マトリクス（6×N）」を **そのまま入力** して 3D モーションを出力する具体的なパイプライン設計

を提示します。補正や詳細な実装（コード／ファイル）まではこの返答で示しますので、必要ならそのままコピーして試せます。

野馬分鬃 と 簡化24式太極拳 を想定しています（流派差はパラメータで吸収）。

---

# 1) 今のマトリクスで**足りない縦軸（要素）** — 追加すべき項目（優先順）

既存の6要素だけでは3D再現に必要なローレベル情報が不足します。以下を追加します。

1. **関節目標（Joint targets）** — 各主要関節（股・膝・足首・腰・胸・頸・肩・肘）のターゲット角度/回転系
2. **末端接触位相（Contact phases）** — 両足の接地/離地フラグと接地点（つま先/踵）
3. **力/トルク目標（Kinetics）** — 各関節の目標トルク／地面反力の期待値（省略可だが安定化に有用）
4. **重心軌跡（CoM path）** — 体幹質量中心の3D軌跡（足裏支持多重判定に必須）
5. **時間スケール（Timing scale）** — 動作の実時間長（例：3s／動作）、局所タイミング（フェーズ長）
6. **身体寸法（Anthropometrics）** — 身長・体重・各リンク長（リターゲティングに必須）
7. **フィードバック遅延／制御ゲイン** — PDゲインや制御遅延（現実的な安定性のため）
8. **足裏摩擦・地面特性** — 地面摩擦係数・硬さ（シミュレータ設定）
9. **視覚化メタ（出力フォーマット）** — 出力をBVH/GLTF/FBX/NumPy配列のどれにするか
10. **評価指標** — フットスリップ、ZMPマージン、姿勢誤差などの評価基準

> これらを縦軸に足すことで「上位記号 → 実行可能な低レベル軌道」への写像が可能になります。

---

# 2) 全体パイプライン（高レベル）

テキストで図的に示します。

入力（パターン・マトリクス）
→ ①シンボル・パーサ（時間正規化）
→ ②モーションプリミティブライブラリ（シンボル→粗パラメータ）
→ ③連続化（タイムスケーラ／補間）
→ ④リターゲット（人体寸法）
→ ⑤逆運動学（IK）＋軌道整形（spline）
→ ⑥動力学安定化（PD/OPS制御 or トルク最適化）
→ ⑦物理シミュレータ（MuJoCo/PyBullet/Drake）で検証・微調整
→ 出力（BVH/GLTF/NumPy + メトリクス）

---

# 3) 各モジュールの具体設計

## ① シンボル・パーサ

* 入力：6×N のマトリクス（各セル：漢字1字）＋タイムスケール（全体 or 列ごと）
* 出力：時系列イベント列 `[{start, end, element, symbol, confidence}]`
* ルール例：同じ列の連続する同一記号はマージ、`歩法:起` は足フェーズ開始イベントを生成

## ② モーションプリミティブライブラリ（Symbol → primitive params）

* 各記号に **パラメータ化されたモーションプリミティブ** をマップ
  例：`弓` → `{pelvis_forward: +0.18m, knee_flex: 30°, hip_pitch:-10°, duration_scale:1.0}`
  `分` → `{left_hand: +0.25m_up & +0.15m_forward, right_hand: -0.20m_down, duration_scale:0.6}`
* プリミティブは**ベクトル（関節目標のオフセット）**で内部表現
* ライブラリは YAML/JSON で管理（編集可能）

## ③ 連続化（タイム伸長・補間）

* 列ごとのフェーズ長を`duration = base_duration * scale(symbol)`で決定
* 補間：Cubic spline or minimum-jerk for smoothness
* サンプリング周波数：推奨 60Hz（稼働負荷に応じて 30–120Hz）

## ④ リターゲット（人体寸法適用）

* 入力人体モデル：身長, 上腕長, 腿長, etc.
* 目標：プリミティブのエフェクタ位置をターゲット身長に合わせてスケーリング

## ⑤ 逆運動学（IK）＋軌道整形

* IK solver：Levenberg–Marquardt or Jacobian pseudo-inverse + null-space projection
* 優先度：足接地の位置（硬優先） → pelvis位置 → 手先追従
* 出力：関節角の時系列（joint_angles[t, dof]）

## ⑥ 動力学安定化（制御）

二つの戦略を提示：

A) **コントローラ駆動（推奨、現実的）**

* PD（または PID）で関節角をトルクに変換（simで実行）
* 追加：足裏にフットコンタクト力制御（force feedback）
* バランス補正：腰・肩でCoM補償（小さな補正角）
* ローパスフィルタでノイズ除去

B) **最適化ベース（高品質だが高コスト）**

* トラジェクトリ最適化（direct collocation）でトルク最小化下で軌道を求める
* 制約：接地不動、関節限界、摩擦円錐
* 解法：CasADi + IPOPT 等

## ⑦ 物理シミュレータ（検証・微調整）

* 推奨：**MuJoCo**（高精度） or **PyBullet**（オープンかつ導入容易）
* 設定：接地摩擦, 反発係数, 異方性摩擦
* メトリクス：foot slip (m), ZMP margin (m), joint limit violations (#), pose RMSE (deg)

---

# 4) データフォーマット（例：JSON 入出力）

### 入力（簡易例：6×6 マトリクスを列ごとに時相付きで表現）

```json
{
  "meta": {"form":"野馬分鬃", "height_m":1.70, "mass_kg":65, "fps":60},
  "matrix_time_columns": [
    {"phase":"起","duration_s":0.6, "cols":[
       {"歩型":"虚","歩法":"起","手法":"抱","身法":"含","呼吸":"吸","意念":"集"}
    ]},
    {"phase":"移","duration_s":0.8, "cols":[
       {"歩型":"虚","歩法":"移","手法":"掤","身法":"転","呼吸":"吸","意念":"導"}
    ]},
    {"phase":"旋","duration_s":0.6, "cols":[
       {"歩型":"中","歩法":"送","手法":"分","身法":"旋","呼吸":"変","意念":"送"}
    ]},
    {"phase":"分","duration_s":0.7, "cols":[
       {"歩型":"弓","歩法":"進","手法":"分","身法":"開","呼吸":"呼","意念":"達"}
    ]},
    {"phase":"進","duration_s":0.6, "cols":[
       {"歩型":"弓","歩法":"定","手法":"按","身法":"沈","呼吸":"呼","意念":"透"}
    ]},
    {"phase":"定","duration_s":0.4, "cols":[
       {"歩型":"弓","歩法":"定","手法":"定","身法":"立","呼吸":"収","意念":"静"}
    ]}
  ]
}
```

### 出力（例：時系列の joint_angles と CoM）

```json
{
  "meta": {"fps":60, "frames":180, "dof":30},
  "frames": [
    {"t":0.00, "joints":[0.0, -0.1, 0.02, ...], "com":[0.0,1.02,0.02]},
    {"t":0.0167,"joints":[...], "com":[...]}
    ...
  ],
  "metrics": {"foot_slip_max":0.003, "zmp_min_margin":0.04, "pose_rmse_deg":2.1}
}
```

出力は BVH/GLTF に変換可能。

---

# 5) シンボル→パラメータの例（抜粋）

（すぐ使える初期値）

* `弓`（弓歩）: `{pelvis_x:+0.18m, pelvis_z:-0.04m, left_knee:+30deg, right_knee:+10deg}`
* `虚`（虚歩）: `{pelvis_x:+0.05m, pelvis_z:+0.02m, weight_ratio:0.3}`
* `掤`: `{left_hand_pos: +0.18fwd + 0.06up, shoulder_elev:-5deg}`
* `分`: `{left_hand_up:+0.25m, right_hand_down:-0.18m}`
* `含`: `{thorax_tilt:+3deg, chest_relax:true}`
* `吸`: `{breath_phase:in, amplitude:small}`

これらはJSONでライブラリ化し、パーサが参照してパラメータを合成します。

---

# 6) 制御・安定化の実装の具体案（推奨スタック）

* **プロトタイプ**：Python + PyBullet（早く試せる）

  * IK: `ikpy` or custom Jacobian solver
  * Controller: PD per joint + CoM biasing
  * Visualization: Blender (import BVH) / WebGL (three.js)

* **研究／高品質**：C++/Python + MuJoCo or Drake

  * Trajectory optimization: CasADi + IPOPT
  * Real-time control: Operational-space control (Khatib) + QP-based whole-body controller

* **学習ベース補助（オプション）**

  * 条件付きモーションモデル（Transformer/LSTM/Conditional diffusion）を使い、マトリクス→生データ（角度系列）を学習させる（大量データがある場合）。
  * ハイブリッド：ルールベース初期でMLは補正学習。

---

# 7) 評価指標（自動評価）

* foot slip distance (max, mean)
* ZMP margin (min) — 安定なら > 0.02–0.05 m
* joint limit violations (count)
* pose RMSE vs target end-effector path
* energy cost (sum torque^2) — for naturalness

---

# 8) 最短プロトタイプ実行手順（15–60 分で最初の可視化）

1. パターンJSON（上の入力例）を作る。
2. プリミティブライブラリ JSON を用意（上の例でOK）。
3. Python スクリプトでパーサ → 補間（slerp/spline） → IK（ikpy）を実行し `joint_angles.npy` を出す。
4. PyBullet に人型URDFを読み込み、PDで角度を再生して可視化。
   （私はこのプロトタイプのコード骨子を提示できます。要望あれば即出します。）

---

# 9) 追加の工夫（品質向上）

* **位相同期学習**：手と足の位相差をデータから学習し、プリミティブに位相オフセットを付与。
* **ユーザ補正UI**：各シンボルのパラメータ（角度・移動量）をスライダで微調整できるWeb UI。
* **フル体幹モデル**：多関節（脊柱分節）により「含胸・回腰」を細かく表現。

---

# 10) 出力例（短いサンプル説明）

もし「上の入力 JSON を使って PyBullet で再生する簡易スクリプト」を出しましょうか？

* すぐに動かせる最小限の Python + PyBullet のスクリプト（IK と PD 再生）を作成できます。
* そのスクリプトは `primitive_library.json` を読み、上の `matrix_time_columns` を解釈し、60Hzで角度列を生成 → PyBulletで再生します。

（要望があれば、**そのまま貼れるコード**を返します。どのランタイム／シミュレータで試したいか指定がなければ、PyBullet + Python で最短実装コードを出します。）

---

以上が「パターン・マトリクスを入力して3Dモーションを出力する」ための設計と具体手順です。
どれを先に出しましょうか？（すぐ使える）

A) `primitive_library.json` とサンプル `matrix.json` をベースにした **PyBulletプロトタイプの完全コード**（Python）
B) 高品質向けに **MuJoCo + trajectory-optimization** の設計＆サンプル問題定義（CasADi）
C) データ形式（BVH/GLTF）への **エクスポート実装**（短スニペット）

選ばずに、私の判断で一つ作ることもできます — どれが良いですか？
