import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import socceraction.spadl as spadl
from pathlib import Path
import warnings

def run_analysis(target_name, data_dir):
    # 警告を無視
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # =========================================================
    # 1. 設定とデータ読み込み
    # =========================================================
    x_bins = 16
    y_bins = 12
    n_cells = x_bins * y_bins
    data_dir = Path(data_dir)
    events_path = data_dir / f"spadl_events_{target_name}.csv"
    
    # 出力ファイル名にも target_name を含める
    output_xT_grid_path = data_dir / f"xT_grid_{target_name}.npy"
    output_img_path = data_dir / f"xT_heatmap_{target_name}.png"

    print(f"\n[{target_name}] xTの計算を開始します...")
    print("データを読み込んでいます...")
    
    if not events_path.exists():
        print(f"エラー: データが見つかりません {events_path}")
        return

    events = pd.read_csv(events_path)
    if 'type_name' not in events.columns:
        print("IDを名前に変換しています...")
        events = spadl.add_names(events)

    # =========================================================
    # 2. 攻撃方向の正規化
    # =========================================================
    print("データの前処理（攻撃方向の正規化）中...")
    
    # 高速化のため必要な列のみ
    cols = ['game_id', 'period_id', 'team_id', 'type_name', 'result_name', 'start_x', 'start_y', 'end_x', 'end_y']
    events_subset = events[cols].copy()

    shot_avg = events_subset[events_subset['type_name'] == 'shot'].groupby(
        ['game_id', 'period_id', 'team_id']
    )['start_x'].mean()

    all_avg = events_subset.groupby(['game_id', 'period_id', 'team_id'])['start_x'].mean()
    direction_avg = shot_avg.combine_first(all_avg).reset_index()

    left_attacking = direction_avg[direction_avg['start_x'] < 52.5].copy()
    left_attacking['needs_flip'] = True
    
    # マージ前に不要な列を落とす
    left_attacking = left_attacking.drop(columns=['start_x'], errors='ignore')

    events_merged = events.merge(
        left_attacking, 
        on=['game_id', 'period_id', 'team_id'], 
        how='left'
    )

    mask = events_merged['needs_flip'] == True
    for col in ['start_x', 'end_x']:
        events_merged.loc[mask, col] = 105.0 - events_merged.loc[mask, col]
    for col in ['start_y', 'end_y']:
        events_merged.loc[mask, col] = 68.0 - events_merged.loc[mask, col]

    events = events_merged
    print(f"正規化完了: {mask.sum()} 行を反転しました。")

    # =========================================================
    # 3. 4つの行列（コンポーネント）の準備
    # =========================================================
    print("xT計算に必要な行列を準備しています...")

    # ★修正2: 読み込むファイル名にも target_name を付与する！
    # これをしないと、ドイツの計算にイタリアの確率を使ってしまいます。
    try:
        m_success_path = data_dir / f"move_success_prob_{target_name}.npy"
        t_matrix_path = data_dir / f"transition_matrix_{target_name}.npy"
        
        m_success = np.load(m_success_path)
        T = np.load(t_matrix_path)
        print(f" -> 行列ファイルを読み込みました ({target_name})")
    except FileNotFoundError as e:
        print(f"エラー: 必要な .npy ファイルが見つかりません。\n詳細: {e}")
        return

    # --- B. ゴール決定率 (P_score) & C. シュート選択率 (P_shoot) ---
    shots = events[events['type_name'] == 'shot']
    goals = shots[shots['result_name'] == 'success']
    move_attempts = events[events['type_name'].isin(['pass', 'dribble', 'cross'])]

    shot_counts, _, _ = np.histogram2d(shots['start_x'], shots['start_y'], bins=[x_bins, y_bins], range=[[0, 105], [0, 68]])
    goal_counts, _, _ = np.histogram2d(goals['start_x'], goals['start_y'], bins=[x_bins, y_bins], range=[[0, 105], [0, 68]])
    move_counts, _, _ = np.histogram2d(move_attempts['start_x'], move_attempts['start_y'], bins=[x_bins, y_bins], range=[[0, 105], [0, 68]])

    with np.errstate(divide='ignore', invalid='ignore'):
        p_score = np.nan_to_num(np.divide(goal_counts, shot_counts))

    total_actions = shot_counts + move_counts
    with np.errstate(divide='ignore', invalid='ignore'):
        p_shoot = np.nan_to_num(np.divide(shot_counts, total_actions))

    # =========================================================
    # 4. xTの反復計算 (Iteration)
    # =========================================================
    print("xTの方程式を解いています...")

    p_score_vec = p_score.flatten()
    p_shoot_vec = p_shoot.flatten()
    m_success_vec = m_success.flatten()

    xT = np.zeros(n_cells)

    iterations = 50
    for i in range(iterations):
        shoot_value = p_shoot_vec * p_score_vec
        expected_dest_value = np.dot(T, xT)
        move_value = (1 - p_shoot_vec) * m_success_vec * expected_dest_value
        xT_new = shoot_value + move_value
        
        diff = np.sum(np.abs(xT_new - xT))
        xT = xT_new
        
        if diff < 1e-6:
            print(f"収束しました (Iteration {i+1})")
            break

    xT_grid = xT.reshape(x_bins, y_bins)

    # ★修正3: 保存名にも target_name を付与
    np.save(output_xT_grid_path, xT_grid)
    print(f"xTグリッドを保存しました: {output_xT_grid_path.name}")

    # =========================================================
    # 5. 結果の可視化
    # =========================================================
    plt.figure(figsize=(12, 8))
    max_val = np.max(xT_grid)
    vmax_threshold = max_val * 0.3 

    ax = sns.heatmap(
        xT_grid.T, 
        cmap="OrRd", 
        linewidths=0.1, 
        linecolor='lightgray',
        xticklabels=False, 
        yticklabels=False,
        vmin=0, 
        vmax=vmax_threshold
    )

    ax.invert_yaxis()
    plt.title(f"Expected Threat (xT) Map - {target_name.upper()}", fontsize=15)
    plt.xlabel("Attack Direction ->")
    plt.ylabel("Pitch Width")

    plt.savefig(output_img_path)
    # 自動化のため close に変更
    plt.close() 

    print(f"画像保存先: {output_img_path.name}")
    print("-" * 50)

if __name__ == "__main__":
    run_analysis("germany", r"C:\Users\shiba\wyscout_backup\csv_data")