import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # =========================================================
    # 設定
    # =========================================================
    data_dir = Path(r"C:\Users\shiba\wyscout_backup\csv_data")
    events_path = data_dir / "spadl_events_all.csv"
    grid_path = data_dir / "xT_grid.npy"
    players_path = data_dir / "players.csv"
    teams_path = data_dir / "teams.csv"

    print("全選手・全チームのxTランキング作成を開始します...")
    t0 = time.time()

    # =========================================================
    # 1. データ読み込み
    # =========================================================
    if not events_path.exists() or not grid_path.exists():
        print("エラー: 必要なファイル(events または xT_grid)が見つかりません。")
        return

    # データ量が多いので型を指定してメモリ節約
    events = pd.read_csv(events_path, dtype={'period_id': str})
    xT_grid = np.load(grid_path)
    
    print(f"イベントデータ読み込み完了: {len(events):,} 行")

    # 名簿データの読み込み
    players = pd.read_csv(players_path) if players_path.exists() else None
    teams = pd.read_csv(teams_path) if teams_path.exists() else None

    # =========================================================
    # 2. 攻撃方向の正規化 (一括高速処理)
    # =========================================================
    print("全試合の攻撃方向を正規化しています...")
    
    # 試合・ハーフ・チームごとの平均位置を一括計算
    # transformを使うことで、元の行数と同じ長さのベクトルを作ります
    avg_x_series = events.groupby(['game_id', 'period_id', 'team_id'])['start_x'].transform('mean')
    
    # 反転が必要な行（平均位置が左側）を特定
    mask_flip = avg_x_series < 52.5
    
    # 反転実行 (105 - x, 68 - y)
    # locを使って該当行だけ書き換え
    events.loc[mask_flip, 'start_x'] = 105.0 - events.loc[mask_flip, 'start_x']
    events.loc[mask_flip, 'end_x'] = 105.0 - events.loc[mask_flip, 'end_x']
    events.loc[mask_flip, 'start_y'] = 68.0 - events.loc[mask_flip, 'start_y']
    events.loc[mask_flip, 'end_y'] = 68.0 - events.loc[mask_flip, 'end_y']

    # =========================================================
    # 3. xT計算 (差分法)
    # =========================================================
    print("xT値を計算しています...")
    
    x_bins, y_bins = xT_grid.shape
    w = 105.0 / x_bins
    h = 68.0 / y_bins
    
    # グリッド座標への変換関数 (ベクトル化)
    def get_cell_indices(x_series, y_series):
        c = np.clip(np.floor(x_series / w), 0, x_bins - 1).astype(int)
        r = np.clip(np.floor(y_series / h), 0, y_bins - 1).astype(int)
        return c, r

    # 全データのグリッド特定
    sc, sr = get_cell_indices(events['start_x'], events['start_y'])
    ec, er = get_cell_indices(events['end_x'], events['end_y'])
    
    # xT値の取得
    start_xt = xT_grid[sc, sr]
    end_xt = xT_grid[ec, er]
    
    # 貢献度計算
    # 対象: 成功した(success) 移動系アクション
    target_types = ['pass', 'dribble', 'cross']
    is_valid_move = (events['type_name'].isin(target_types)) & (events['result_name'] == 'success')
    
    events['xT_added'] = 0.0
    # 成功した移動のみ差分を付与
    events.loc[is_valid_move, 'xT_added'] = end_xt[is_valid_move] - start_xt[is_valid_move]

    # =========================================================
    # 4. 集計 (選手ごとのランキング作成)
    # =========================================================
    print("集計中...")
    
    # 選手IDとチームIDでグルーピング
    # (チームIDを含めるのは、チーム名を表示するため。移籍した選手は別レコードになります)
    ranking = events.groupby(['player_id', 'team_id']).agg({
        'xT_added': 'sum',          # xT合計
        'game_id': 'nunique',       # 出場試合数
        'type_name': 'count'        # アクション総数
    }).reset_index()
    
    ranking.rename(columns={'game_id': 'matches', 'type_name': 'actions'}, inplace=True)
    
    # 1試合平均の計算
    ranking['xT_per_match'] = ranking['xT_added'] / ranking['matches']

    # =========================================================
    # 5. 情報の結合 (選手名・チーム名)
    # =========================================================
    # 選手名
    if players is not None:
        # 必要な列だけマージ
        p_cols = ['player_id', 'short_name', 'official_name']
        cols = [c for c in p_cols if c in players.columns]
        ranking = ranking.merge(players[cols], on='player_id', how='left')
    
    # チーム名
    if teams is not None:
        # team_id, official_name, area_name など
        t_cols = ['team_id', 'official_name', 'area_name']
        # 列名重複を防ぐためリネームしてからマージするか、suffxを使う
        # ここではシンプルにマージ
        ranking = ranking.merge(teams, on='team_id', how='left', suffixes=('', '_team'))
        
        # チーム名が見やすいように列整理
        if 'official_name_team' in ranking.columns:
            ranking.rename(columns={'official_name_team': 'team_name'}, inplace=True)
        elif 'official_name' in ranking.columns and 'official_name' in teams.columns:
             # 名前衝突時の処理(pandasの挙動依存)を確認しつつ、ここでは簡易的に
             pass

    # =========================================================
    # 6. 保存と表示
    # =========================================================
    # ソート: xT合計の降順
    ranking = ranking.sort_values('xT_added', ascending=False).reset_index(drop=True)
    
    # 列の並び替え（見やすく）
    disp_cols = ['player_id']
    if 'short_name' in ranking.columns: disp_cols.append('short_name')
    if 'team_name' in ranking.columns: disp_cols.append('team_name')
    elif 'official_name' in ranking.columns and 'team_id' in ranking.columns: 
         # teams.csvのofficial_nameがteam_nameとして残っている場合など
         # ここはデータセットのカラム名次第なので柔軟に
         pass
         
    disp_cols.extend(['xT_added', 'matches', 'xT_per_match', 'actions'])
    
    # 保存
    output_file = data_dir / "xT_ranking_ALL_PLAYERS.csv"
    ranking.to_csv(output_file, index=False)
    
    print("-" * 60)
    print(f"処理完了: {time.time() - t0:.1f} 秒")
    print(f"全選手ランキングを保存しました: {output_file}")
    print("-" * 60)
    print("【xT貢献度 世界ランキング TOP 20】")
    
    # 画面表示用にカラムを絞る
    show_cols = [c for c in disp_cols if c in ranking.columns]
    print(ranking[show_cols].head(20).to_string(index=True))

if __name__ == "__main__":
    main()