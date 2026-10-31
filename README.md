# CRANE+ V2 自動把持・ACTプロジェクト

講義の `matsuolab/pai_training` を土台に、Gazebo Sim上の立方体を把持し、5 cm持ち上げて1秒保持する課題を実装します。物体座標はシミュレータの真値を利用します。画像認識は使用しません。

## 現在の状態

- 講義のLeRobotコミットをsubmoduleとして固定済み。
- シミュレーション時刻で20 Hz、関節5個＋物体座標3個→目標関節5個のデータ契約。
- Gazebo Simで物体生成・位置変更、MoveIt IKによる教師、成功判定、成功データ変換、ACT推論を実装。
- 単体テストと、合成データによるLeRobot変換・ACT損失計算・逆伝播・保存/再読込を検証。
- **物理把持・実データ学習・学習済みモデルの成功率は未検証です。** `calibrated=false` はその状態を表します。

元の講義READMEは `docs/lecture_README.md` に保存しています。

## 既存環境を使う

ホスト側:

```bash
cd /home/tomo/workspace/robots/physical_ai_pickup
scripts/container.sh scripts/pickup_robot.py diagnose
```

`ros2-docker` の既存ROS 2 Humble / Gazebo Sim / MoveIt環境と、`/home/ubuntu/robots/pai_training/.pixi/envs/default/bin/python` を利用します。既存環境のインストールは変更しません。プロジェクトの実行用コピーはコンテナの `/tmp/physical_ai_pickup` へ配置します。

データとモデルは、ホストの `ros2/ai_robot_book_data/pickup_experiment/{data,checkpoints}` に永続化します。プロジェクトの `data` と `checkpoints` はそこへのリンクです。コンテナ内の一時コードを編集せず、ホストのソースを編集してください。次回実行時にコピーされます。

ネイティブの物体追跡ヘルパーはコンテナ内で初回にコンパイルします。この環境の `Pose_V -> TFMessage` 変換では名前が失われるため、Gazebo Transportで名前を保持して座標を読みます。

## 実行順序

```bash
# 初回のみ: 赤い2.5cm立方体を追加（既存物体は削除しません）
scripts/container.sh scripts/pickup_robot.py spawn
# 関節・物体座標・IKサービスの確認
scripts/container.sh scripts/pickup_robot.py diagnose
# 指令を送らず、現在位置に対するIK解だけ確認
scripts/container.sh scripts/pickup_robot.py plan
```

`configs/pickup.json` の把持オフセット・手先姿勢・グリッパー開閉値・可動域を調整します。ロボット基準座標の原点はworldの `[0,0,1.015]`、軸は同じ方向という前提です。モデルを移動/回転した場合は座標変換を変更してください。既存の手先フレームはlink4原点のため、指の実際の把持中心とのオフセットが必要です。

初期設定は候補値です。IK成功だけでは経路の衝突や物理把持の成功は保証されません。現在の教師は関節補間を使い、MoveItの経路計画は使いません。候補動作を検証できる状態になったら `calibrated=true` に変更し、まず1件で確認します。実機を接続しないシミュレータで実行してください。

```bash
scripts/container.sh scripts/pickup_robot.py collect --episodes 1 --seed 42
# 成功を確認後に増やす
scripts/container.sh scripts/pickup_robot.py collect --episodes 50 --seed 42
scripts/container.sh scripts/pickup_dataset.py
WANDB_MODE=offline scripts/container.sh scripts/train.py
scripts/container.sh scripts/pickup_robot.py infer --model-path checkpoints/pickup_act --episodes 20 --seed 2026 --output data/evaluation
```

推論・教師の実行中はテレオペや他の指令ノードを停止してください。成功判定は実測した物体中心の高さが開始時より5 cm以上高い状態を最後の1秒間保持できたかです。把持接触そのものは判定しません。

失敗エピソードも `result.json` に残します。データ変換は成功エピソードのみを読み、20 Hzの間隔、配列長、関節順序、有限値を検証します。出力先が存在すると上書きせずエラーになります。再収集・再変換では `--output` に新しいパスを指定し、学習設定のdataset_rootも合わせてください。

評価用seedとデータは学習用から分離してください。評価結果は各エピソードの `result.json` に保存されます。推論開始時にpolicyをresetし、教師は実行しません。収集時も推論時もホーム復帰・物体配置は共通の初期化です。

## 検証

```bash
python3 -m unittest discover -s tests -v
scripts/container.sh tests/smoke_ml.py
```

ML smokeは一時フォルダの合成データのみを使用します。物理把持の成功や提出用モデルとして扱わないでください。初回のMLライブラリ読み込みには時間がかかります。

`pixi` で新規環境を構築する場合は `git submodule update --init`、`pixi install` を実行してください。ROS関連の実行は講義コンテナ内で行います。標準タスク `convert/train/infer` は本プロジェクト用に設定済みです。元の `scripts/rosbag2lerobot.py` とその変換設定は講義参照用で、このpickupデータには使用しません。

## 提出

1〜3分の動画に、課題説明、教師データ自動収集、学習済みモデルの実行、未使用配置での成功/失敗を収めます。倍速は明記します。レポートには物体位置の真値使用、エピソード数、評価配置/seed、成功率、接触設定、失敗例を記載します。未実施の実験値は記載しません。

### 把持調整の単発試行

`plan` で解が得られた後、未校正状態でも1回だけ候補動作を試せます。

```bash
scripts/container.sh scripts/pickup_robot.py reset
scripts/container.sh scripts/pickup_robot.py trial --seed 42 --output data/calibration
```

`trial` は1エピソード限定です。結果を確認してから校正完了フラグを設定します。初期の試行では期限超過と、wall時刻に対するシミュレータの遅延が判明しました。制御・記録・保持判定をシミュレーション時刻に変更しています。失敗データは学習に含めません。収集・推論とMLテスト/学習を同時に走らせないでください。
