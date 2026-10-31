# CRANE+ V2 自動把持・ACTプロジェクト

講義の `matsuolab/pai_training` を土台に、Gazebo Sim上の立方体を把持し、4 cm持ち上げて1秒保持する課題を実装します。物体座標はシミュレータの真値を利用します。画像認識は使用しません。

## 現在の状態

- 講義のLeRobotコミットをsubmoduleとして固定済み。
- シミュレーション時刻で20 Hz、関節5個＋物体座標3個→目標関節5個のデータ契約。
- Gazebo Simで物体生成・位置変更、MoveIt IKによる教師、成功判定、成功データ変換、ACT推論を実装。
- 単体テストと、合成データによるLeRobot変換・ACT損失計算・逆伝播・保存/再読込を検証。
- **4 cm立方体の単発把持・持ち上げを目視確認済みです。** 保存済み試行は4 cm・1秒の新条件でも成功を確認。seed 43〜45の3配置と端付近の4配置で教師動作の成功を確認し、候補設定を`calibrated=true`にしています。検証した狭い範囲で収集を始める段階であり、実データ学習と学習済みモデルの成功率は未検証です。

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

`pickup_robot.py` は全モードで既定の設定として `configs/pickup_clearance_candidate.json`（一辺4 cm）を読みます。`--config` を省略しても生成と動作の設定が揃います。旧2.5 cm設定は `--config configs/pickup.json` で明示指定した場合のみ使用します。既に小さいpickup_cubeを生成している場合は、`spawn`ではなく`respawn`で作り直してください。

```bash
# 初回のみ: 赤い4cm立方体を追加（既存物体は削除しません）
scripts/container.sh scripts/pickup_robot.py spawn
# 関節・物体座標・IKサービスの確認
scripts/container.sh scripts/pickup_robot.py diagnose
# 指令を送らず、現在位置に対するIK解だけ確認
scripts/container.sh scripts/pickup_robot.py plan
```

`configs/pickup_clearance_candidate.json` の把持オフセット・手先姿勢・グリッパー開閉値・可動域を調整します。ロボット基準座標の原点はworldの `[0,0,1.015]`、軸は同じ方向という前提です。モデルを移動/回転した場合は座標変換を変更してください。既存の手先フレームはlink4原点のため、指の実際の把持中心とのオフセットが必要です。

初期設定は候補値です。IK成功だけでは経路の衝突や物理把持の成功は保証されません。現在の教師は関節補間を使い、MoveItの経路計画は使いません。候補動作を検証できる状態になったら `calibrated=true` に変更し、まず1件で確認します。実機を接続しないシミュレータで実行してください。

```bash
scripts/container.sh scripts/pickup_robot.py collect --episodes 1 --seed 42
# 成功を確認後に増やす
scripts/container.sh scripts/pickup_robot.py collect --episodes 50 --seed 42
scripts/container.sh scripts/pickup_dataset.py
WANDB_MODE=offline scripts/container.sh scripts/train.py
scripts/container.sh scripts/pickup_robot.py infer --model-path checkpoints/pickup_act --episodes 20 --seed 2026 --output data/evaluation
```

推論・教師の実行中はテレオペや他の指令ノードを停止してください。成功判定は実測した物体中心の高さが開始時より4 cm以上高い状態を最後の1秒間保持できたかです。把持接触そのものは判定しません。

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

グリッパー設定の訂正: 公式pick_and_place例に合わせ、開=-30度、閉=+10度へ変更。従来の閉=-0.7 radは開方向だった。修正後の把持は未検証。詳細はdocs/worklog/2026-10-30.md。

### 手先姿勢のIK比較（動作なし）

` scripts/container.sh scripts/pickup_robot.py scan ` は現在の物体位置で4種類のpitchを比較します。設定ファイルを変更せず、ロボットや物体も動かしません。PASSは各段階のIK成功のみを表し、把持・指先のクリアランス・経路の確認は別途必要です。

### 閉動作中の机との隙間

`python3 scripts/check_gripper_clearance.py` は講義の衝突メッシュからhand/link4の開閉区間全体の最低高さを求めます。ROSやシミュレータに指令を送信しません。`--episode data/calibration/<試行名>` で記録末の姿勢も比較できます。前提と限界は出力およびスクリプトを参照してください。候補設定は `configs/pickup_clearance_candidate.json` に分離し、元設定は保持しています。

### 4 cm立方体の候補設定

提出タスクを把持しやすくするため、候補設定の箱を一辺4 cm・質量20 gに変更しています。元のpickup.jsonは一辺2.5 cmのままです。サイズ変更は既存モデルのresetでは反映されません。Gazebo起動中に次を実行してください。

```bash
scripts/container.sh scripts/pickup_robot.py respawn --config configs/pickup_clearance_candidate.json
scripts/container.sh scripts/pickup_robot.py plan --config configs/pickup_clearance_candidate.json
```

respawnは設定のobject_name（pickup_cube）のモデルを置換します。plan成功後に同じ--configを付けたtrialで確認します。過去のデータは保持します。今後の試行にはconfig.jsonも保存します。対象物の寸法と質量、条件変更の理由は提出レポートに明記してください。大型化後の単発把持は確認済みです。再現性は引き続き検証します。

### 教師動作の撮影

現在の成功条件は4 cm以上を1秒保持です。新しい結果ファイルにはsuccess_criteriaも記録します。過去の5 cm判定結果は書き換えません。

```bash
scripts/container.sh scripts/pickup_robot.py trial --config configs/pickup_clearance_candidate.json --seed 42 --output data/video_trials --wait-for-recording
```

ホーム復帰と箱の再配置後にEnter待ちします。画面録画を開始してからターミナルでEnterを押してください。画面録画自体は別の録画機能で行います。これは教師スクリプトの動作確認動画です。課題のAI活用を示す学習済みモデルの推論評価は後続工程です。

### 初回9件のデータ変換

記録間隔が不適合な1件は元データを残して明示除外します。除外名の誤記や、選択対象にある他の不正記録はエラーになります。

```bash
scripts/container.sh scripts/pickup_dataset.py --episodes data/episodes --exclude-episode collect_1793469615859157802 --output data/lerobot/pickup
```

初回収集から9件・3,291フレームの変換と読み込み検証を完了しています。`--validate-only`を付けると生成せず検査だけ行います。変換成功時は`meta/conversion_manifest.json`に採用元・除外名・元フレームのハッシュを記録します。出力先が存在すると上書きせず停止します。

### 変換後の検証

```bash
scripts/container.sh scripts/verify_pickup_dataset.py
```

元記録のハッシュと全フレームの一致、メタデータ、エピソード境界、LeRobotの学習用バッチ形状を検証します。変換後の時刻は、元の間隔を検証した上で20 Hzのグリッドへ再構成されます。初回データは検証済みで、詳細はdocs/reviews/2026-10-31-converted-dataset.jsonです。まだ学習しておらず、現状のtrain splitは全9件のため、本学習前にエピソード単位の検証データ分離を決めます。

### ACTの短時間学習

初回データはconfigs/pickup_split.jsonで学習7エピソード・検証2エピソードに分けています。正規化統計は学習用だけから計算します。元データセットのエピソード境界を保ち、同じエピソードのフレームが学習と検証の両方に入らない構成です。分割は初回9件の変換manifestに紐付くため、追加収集で再変換した場合は分割を再作成・確認します。

```bash
# 読み込みと分割の確認のみ（学習しません）
scripts/container.sh scripts/train.py --config configs/train_smoke.yaml --validate-only
# CPU上の小型ACTで100ステップの動作確認
scripts/container.sh scripts/train.py --config configs/train_smoke.yaml
```

出力先はcheckpoints/pickup_act_smoke。run_config.jsonに設定と分割、metrics.jsonlに学習損失と検証の最初の行動のMAE[rad]、training_summary.jsonに完了ステップ数と保存モデル再読み込みの結果を保存します。検証時には正解行動をモデルに入力しません。検証MAEはシミュレーションの把持成功率ではありません。この小型・短時間モデルは学習経路の確認用です。既存出力先は上書きしないので、再実行時は設定のoutput_directoryを新しい名前に変更してください。

### 1,000ステップの比較学習

100ステップのモデルは残し、同じ小型構成・分割・seedで初期状態から1,000ステップを学習します。前回checkpointからの再開ではありません。

```bash
scripts/container.sh scripts/train.py --config configs/train_1000.yaml
```

出力先はcheckpoints/pickup_act_1000。前回と同じ25ステップ周期で評価し、完了後に保存モデルの検証MAEと関節制限超過を比較します。

### 学習済みモデルの単発推論

1,000ステップモデルは検証用732観測のオフライン評価で関節制限超過0件。次はシミュレーション内の閉ループ動作を確認します。

```bash
scripts/container.sh scripts/pickup_robot.py infer --model-path checkpoints/pickup_act_1000 --seed 42 --seconds 25 --output data/inference_tests
```

ホーム復帰・箱の再配置後は、教師スクリプトのIK段階を使わずモデルから指令を生成します。既定CPUスレッド数は2。モデル読み込み後に観測を更新・推論をウォームアップします。試行にはpolicy_info.json（モデルのSHA-256など）を保存し、フレームと結果に推論時間を記録します。時間はsim時刻で25秒のため、wall時刻の所要時間はシミュレータ速度に依存します。推論中は教師/テレオペを同時に動かさないでください。モデルの閉ループ把持成功は未確認です。

### 推論時の一時的な遅延

目標周期は20 Hzです。締切を15 ms超過した単発の遅れは記録して次周期を現在時刻から再開します。100 ms超の締切遅延または3回連続超過では停止します（いずれもsim時刻）。追い付きのために指令を連続送信しません。従来の厳密な停止を使う場合はinferに--strict-timingを指定します。result.jsonのtimingで実測fpsと遅延を確認してください。学習データの時刻検査は緩めていません。

保持判定は実測のシミュレーション時刻に基づく4 cm以上・1秒です。観測間隔が変わっても20フレームを1秒とみなさず、保持区間内に200 ms超の観測欠落がある場合は成功にしません。
