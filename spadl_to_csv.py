import pandas as pd
from pathlib import Path
from tqdm import tqdm # 進行状況バー
from socceraction.data.wyscout import PublicWyscoutLoader
import socceraction.spadl as spadl
import warnings

# 警告を非表示
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. 設定
data_dir = Path(r"C:\Users\shiba\wyscout_backup\data")
output_dir = Path(r"C:\Users\shiba\wyscout_backup\csv_data") # CSV保存用フォルダ

# 保存用フォルダがなければ作る
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[処理開始] データディレクトリ: {data_dir}")
print(f"[保存先] {output_dir}")

# 2. Loader初期化
loader = PublicWyscoutLoader(root=str(data_dir), download=False)
comp_id = 524
season_id = 181248

print("試合一覧を取得中...")
matches = loader.games(comp_id, season_id)
print(f"対象: {len(matches)} 試合")

# 3. マスタデータの準備（ID → 名前変換用）
actiontypes = spadl.actiontypes_df()
results = spadl.results_df()
bodyparts = spadl.bodyparts_df()

# 4. 一括変換処理
all_actions = [] # ここに全試合のデータを溜めていく

print("全試合の変換を開始します...")
for idx, match in tqdm(matches.iterrows(), total=len(matches)):
    try:
        game_id = match.game_id
        home_team_id = match.home_team_id
        
        # イベント読み込み
        events = loader.events(game_id)
        
        # SPADL変換
        actions = spadl.wyscout.convert_to_actions(events, home_team_id=home_team_id)
        
        # IDを名前に変換
        actions = actions.merge(actiontypes, how='left', on='type_id')
        actions = actions.merge(results, how='left', on='result_id')
        actions = actions.merge(bodyparts, how='left', on='bodypart_id')
        
        # 【重要】どの試合のデータかわかるように game_id 列を追加
        actions['game_id'] = game_id
        
        # リストに追加
        all_actions.append(actions)
        
    except Exception as e:
        print(f"\n[Error] GameID {game_id} でエラー発生: {e}")
        continue

# 5. 結合してCSV保存
print("\nデータを結合してCSVに書き出し中...")

if all_actions:
    # 1. イベントデータの保存
    df_actions = pd.concat(all_actions, ignore_index=True)
    events_csv = output_dir / "spadl_events_italy.csv"
    df_actions.to_csv(events_csv, index=False)
    
    # 2. 試合データの保存
    matches_csv = output_dir / "spadl_matches_italy.csv"
    matches.to_csv(matches_csv, index=False)
    
    print("-" * 50)
    print("処理完了！")
    print(f"イベントデータ: {events_csv}")
    print(f"試合リスト: {matches_csv}")
    
    # サイズ確認
    size_mb = events_csv.stat().st_size / (1024 * 1024)
    print(f"イベントファイルサイズ: {size_mb:.2f} MB")
    print("★Excel等で中身を確認できます！")
else:
    print("データが作成されませんでした。")