import pandas as pd
from pathlib import Path
import json

# 設定: データがあるフォルダ

data_dir = Path(r"C:\Users\shiba\wyscout_backup\data") 
json_file = data_dir / "teams.json"
output_file = data_dir / "teams.csv"

def main():
    if not json_file.exists():
        print(f"エラー: 元データ {json_file} が見つかりません。")
        print("teams.json がどこにあるか確認し、パスを修正してください。")
        return

    print(f"読み込み中: {json_file}")
    
    # JSONの構造によってはエンコーディング指定が必要な場合があります (utf-8 or latin-1)
    try:
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
         with open(json_file, encoding='unicode_escape') as f:
            data = json.load(f)

    # DataFrame化
    df_teams = pd.DataFrame(data)

    # 必要な列を選定・リネーム (Wyscoutの生データ形式に合わせる)
    # 一般的なWyscout public datasetのキー: wyId, name, officialName, area
    if 'wyId' in df_teams.columns:
        df_teams = df_teams.rename(columns={
            'wyId': 'team_id', 
            'officialName': 'official_name',
            'name': 'short_name',
            'area': 'area_data' # areaは辞書型で入っていることが多い
        })
        
        # areaデータから国名を取り出す（必要な場合）
        try:
            df_teams['area_name'] = df_teams['area_data'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
        except:
            pass
            
        # 保存用に列を整理 (spadlの形式に合わせるなら team_id, official_name は必須)
        save_cols = ['team_id', 'official_name', 'short_name', 'area_name']
        # 存在しない列は除外
        save_cols = [c for c in save_cols if c in df_teams.columns]
        
        df_teams[save_cols].to_csv(output_file, index=False)
        print(f"成功: {output_file} を作成しました。")
        print(df_teams[save_cols].head())
        
    else:
        print("エラー: 予期しないJSON形式です。列名を確認してください:")
        print(df_teams.columns)

if __name__ == "__main__":
    main()