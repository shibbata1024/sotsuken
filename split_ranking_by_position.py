import pandas as pd
from pathlib import Path

def main():
    # =========================================================
    # 1. 設定
    # =========================================================
    # データのディレクトリ（環境に合わせて修正してください）
    data_dir = Path(r"C:\Users\shiba\wyscout_backup\csv_data")
    
    ranking_file = data_dir / "xT_ranking_ALL_PLAYERS.csv"
    players_file = data_dir / "players.csv"
    
    print("処理を開始します...")

    # =========================================================
    # 2. データの読み込み
    # =========================================================
    if not ranking_file.exists() or not players_file.exists():
        print(f"エラー: 必要なファイルが見つかりません。\n{ranking_file}\n{players_file}")
        return

    df_ranking = pd.read_csv(ranking_file)
    df_players = pd.read_csv(players_file)
    
    print(f"ランキングデータ: {len(df_ranking)} 件")
    print(f"選手名簿データ: {len(df_players)} 件")

    # =========================================================
    # 3. ポジション情報の結合
    # =========================================================
    # players.csvから必要な情報（IDと役割名）だけ抽出
    # ※ role_name列が存在することを確認してください
    if 'role_name' not in df_players.columns:
        print("エラー: players.csv に 'role_name' 列がありません。")
        return

    # 左結合 (ランキングにいる選手のみ残す)
    merged = df_ranking.merge(
        df_players[['player_id', 'role_name']], 
        on='player_id', 
        how='left'
    )

    # =========================================================
    # 4. ポジションの分類ロジック
    # =========================================================
    def classify_position(role):
        if pd.isna(role): return 'Unknown'
        r = str(role).lower()
        
        if 'keeper' in r: return 'GK'
        if 'defender' in r or 'back' in r: return 'DF'
        if 'midfield' in r or 'winger' in r: return 'MF' # WingをFWにしたい場合はここを修正
        if 'forward' in r or 'striker' in r: return 'FW'
        return 'Other'

    merged['general_position'] = merged['role_name'].apply(classify_position)

    # =========================================================
    # 5. 分割して保存
    # =========================================================
    positions = ['GK', 'DF', 'MF', 'FW']
    
    print("-" * 50)
    for pos in positions:
        # そのポジションの選手だけ抽出
        subset = merged[merged['general_position'] == pos].copy()
        
        # xTの合計(xT_added)で降順ソート
        subset = subset.sort_values('xT_added', ascending=False)
        
        # 保存
        output_path = data_dir / f"xT_ranking_{pos}.csv"
        subset.to_csv(output_path, index=False)
        
        # 結果表示
        count = len(subset)
        top_player = subset.iloc[0]['short_name'] if count > 0 else "なし"
        top_score = subset.iloc[0]['xT_added'] if count > 0 else 0.0
        
        print(f"[{pos}] {count:4d}名 | 保存完了: {output_path.name}")
        if count > 0:
            print(f"    Top: {top_player} (xT: {top_score:.4f})")
    
    print("-" * 50)
    print("すべての処理が完了しました。")

if __name__ == "__main__":
    main()