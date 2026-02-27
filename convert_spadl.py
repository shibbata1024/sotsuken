import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import socceraction.spadl as spadl
from socceraction.spadl import wyscout as wyscout_converter

# ==========================================
# 設定: 自動パス設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "spadl_events_all.csv")

def load_json_safe(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return pd.DataFrame(json.load(f))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def process_competition(event_filepath):
    # ファイル名からコンペティション名を抽出
    base_name = os.path.basename(event_filepath)
    comp_name = base_name.replace("events_", "").replace(".json", "")
    
    print(f"\nProcessing Competition: {comp_name}")

    # マッチファイルのパス
    match_filepath = os.path.join(os.path.dirname(event_filepath), f"matches_{comp_name}.json")
    
    if not os.path.exists(match_filepath):
        print(f"  [Skip] マッチファイルなし: {match_filepath}")
        return pd.DataFrame()

    events_df = load_json_safe(event_filepath)
    matches_df = load_json_safe(match_filepath)

    if events_df.empty or matches_df.empty:
        return pd.DataFrame()

    spadl_list = []

    # プログレスバー付きループ
    for idx, match in tqdm(matches_df.iterrows(), total=len(matches_df)):
        match_id = match['wyId']
        match_events = events_df[events_df['matchId'] == match_id].reset_index(drop=True)
        
        if match_events.empty:
            continue

        # ホームチームID特定
        try:
            teams_data = match['teamsData']
            home_team_id = int(next((tid for tid, data in teams_data.items() if data['side'] == 'home'), 0))
        except Exception:
            continue

        try:
            rename_map = {
                'id': 'event_id', 
                'eventId': 'type_id', 
                'subEventId': 'subtype_id',
                'teamId': 'team_id', 
                'playerId': 'player_id', 
                'matchId': 'game_id',
                'matchPeriod': 'period_id'
            }
            match_events = match_events.rename(columns=rename_map)
            
            # eventSec (秒) がある場合、それを milliseconds に変換しておくと安全です
            if 'eventSec' in match_events.columns:
                match_events['milliseconds'] = match_events['eventSec'] * 1000
            else:
                match_events['milliseconds'] = 0

            # ----------------------------------------------------

            # 1. SPADL変換
            actions = wyscout_converter.convert_to_actions(match_events, home_team_id)
            
            # 2. 攻める向きの統一
            actions = spadl.play_left_to_right(actions, home_team_id)
            
            # 3. アクション名の付与
            actions = spadl.add_names(actions)
            
            # メタデータの付与
            actions['game_id'] = match_id
            actions['competition_name'] = comp_name
            
            spadl_list.append(actions)
            
        except Exception as e:
            # エラーが出たら詳細を表示して停止
            print(f"\n[エラー発生] Match ID: {match_id}")
            print(f"エラー内容: {e}")
            raise e 

    if not spadl_list:
        return pd.DataFrame()

    return pd.concat(spadl_list, ignore_index=True)

def main():
    print(f"Search Data Dir: {DATA_DIR}")
    search_path = os.path.join(DATA_DIR, "events_.json")
    event_files = glob.glob(search_path)
    
    if not event_files:
        print("エラー: JSONファイルが見つかりません。")
        return

    all_spadl_data = []

    for event_file in event_files:
        try:
            df_comp = process_competition(event_file)
            if not df_comp.empty:
                all_spadl_data.append(df_comp)
        except Exception:
            print("処理を中断しました。エラーを確認してください。")
            break

    if all_spadl_data:
        print("\n結合中...")
        full_df = pd.concat(all_spadl_data, ignore_index=True)
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"完了: {OUTPUT_FILE}")
    else:
        print("\n警告: SPADLデータを作成できませんでした。")

if __name__ == "__main__":
    main()