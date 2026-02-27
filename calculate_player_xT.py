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
    
    # 対象選手ID (例: 香川真司 = 20550, 本田圭佑 = 14911 など)
    target_player_id = 20550 

    print(f"分析対象 Player ID: {target_player_id}")
    t0 = time.time()

    # =========================================================
    # 1. データ読み込み
    # =========================================================
    if not events_path.exists():
        print("ファイルがありません。")
        return

    events = pd.read_csv(events_path, dtype={'period_id': str})
    
    if not grid_path.exists():
         print("xT_grid.npy がありません。")
         return
    xT_grid = np.load(grid_path)
    
    t1 = time.time()
    print(f"データ読み込み完了: {t1 - t0:.2f}秒")

    # =========================================================
    # 2. 出場試合の特定とフィルタリング
    # =========================================================
    # その選手がプレーした game_id を特定
    played_games = events[events['player_id'] == target_player_id]['game_id'].unique()
    
    if len(played_games) == 0:
        print("この選手が出場した試合はありません。")
        return

    print(f"出場試合数: {len(played_games)} 試合")

    # その試合の「全選手」のデータを抽出する
    relevant_events = events[events['game_id'].isin(played_games)].copy()
    
    print(f"処理対象イベント数: {len(events)} -> {len(relevant_events)} (削減率: {100 - len(relevant_events)/len(events)*100:.1f}%)")

    # =========================================================
    # 3. 攻撃方向の正規化 (修正箇所: transformを使用)
    # =========================================================
    t2 = time.time()
    
    # 【解説】transformを使うと、集計した結果を元の行に合わせて展開してくれます。
    # インデックスがズレる心配がなく、非常に高速です。
    avg_x_series = relevant_events.groupby(['game_id', 'period_id', 'team_id'])['start_x'].transform('mean')
    
    # 平均位置が52.5未満なら反転が必要
    mask_flip = avg_x_series < 52.5

    # 反転実行
    # mask_flipのインデックスはrelevant_eventsと完全に一致するためエラーになりません
    relevant_events.loc[mask_flip, 'start_x'] = 105.0 - relevant_events.loc[mask_flip, 'start_x']
    relevant_events.loc[mask_flip, 'end_x'] = 105.0 - relevant_events.loc[mask_flip, 'end_x']
    relevant_events.loc[mask_flip, 'start_y'] = 68.0 - relevant_events.loc[mask_flip, 'start_y']
    relevant_events.loc[mask_flip, 'end_y'] = 68.0 - relevant_events.loc[mask_flip, 'end_y']

    t3 = time.time()
    print(f"正規化完了: {t3 - t2:.4f}秒")

    # =========================================================
    # 4. 選手本人のxT計算
    # =========================================================
    # ここで初めて本人だけに絞る
    my_events = relevant_events[relevant_events['player_id'] == target_player_id].copy()
    
    if my_events.empty:
        print("正規化後の抽出でデータが空になりました。")
        return

    # グリッド計算の準備
    x_bins, y_bins = xT_grid.shape
    w = 105.0 / x_bins
    h = 68.0 / y_bins
    
    def get_cell(x, y):
        c = np.clip(np.floor(x / w), 0, x_bins - 1).astype(int)
        r = np.clip(np.floor(y / h), 0, y_bins - 1).astype(int)
        return c, r

    sc, sr = get_cell(my_events['start_x'], my_events['start_y'])
    ec, er = get_cell(my_events['end_x'], my_events['end_y'])
    
    # xT付与
    target_types = ['pass', 'dribble', 'cross']
    is_success = (my_events['type_name'].isin(target_types)) & (my_events['result_name'] == 'success')
    
    my_events['xT_added'] = 0.0
    val_diff = xT_grid[ec[is_success], er[is_success]] - xT_grid[sc[is_success], sr[is_success]]
    my_events.loc[is_success, 'xT_added'] = val_diff
    
    total_xt = my_events['xT_added'].sum()
    
    # =========================================================
    # 結果表示
    # =========================================================
    print("-" * 50)
    print(f"【算出結果】")
    print(f"対象選手ID: {target_player_id}")
    print(f"アクション数: {len(my_events)} 回")
    print(f"総xT貢献度: {total_xt:.5f}")
    if len(played_games) > 0:
        print(f"1試合平均xT: {total_xt / len(played_games):.5f}")
    print(f"総計算時間: {time.time() - t0:.4f}秒")
    print("-" * 50)

if __name__ == "__main__":
    main()