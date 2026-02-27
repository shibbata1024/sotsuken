import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# socceractionのインポート確認
try:
    import socceraction.spadl as spadl
except ImportError:
    pass

def run_analysis(target_name, data_dir):
    # 警告を無視
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # =========================================================
    # 1. 設定とデータ読み込み
    # =========================================================
    data_dir = Path(data_dir)
    
    # 入力・出力パスの設定 (target_name対応)
    events_path = data_dir / f"spadl_events_{target_name}.csv"
    output_matrix_path = data_dir / f"shot_selection_prob_{target_name}.npy"
    output_img_path = data_dir / f"shot_selection_prob_{target_name}.png"

    print(f"\n[{target_name}] シュート選択率の計算を開始します...")
    print("データを読み込んでいます...")
    
    if not events_path.exists():
        print(f"エラー: データが見つかりません {events_path}")
        return

    events = pd.read_csv(events_path)

    # ID → 名前変換 (必須)
    if 'type_name' not in events.columns:
        print("IDを名前に変換しています...")
        import socceraction.spadl as spadl
        events = spadl.add_names(events)

    # =========================================================
    # 2. 攻撃方向の正規化 (Smart Normalization)
    # =========================================================
    print("データの前処理（攻撃方向の正規化）中...")

    # 高速化のため必要な列のみ抽出
    cols = ['game_id', 'period_id', 'team_id', 'type_name', 'result_name', 'start_x', 'start_y', 'end_x', 'end_y']
    events_subset = events[cols].copy()

    # シュート位置による判定
    shot_avg = events_subset[events_subset['type_name'] == 'shot'].groupby(
        ['game_id', 'period_id', 'team_id']
    )['start_x'].mean()

    # シュートがない場合の保険
    all_avg = events_subset.groupby(['game_id', 'period_id', 'team_id'])['start_x'].mean()
    direction_avg = shot_avg.combine_first(all_avg).reset_index()

    # 左攻め判定 (x < 52.5)
    left_attacking = direction_avg[direction_avg['start_x'] < 52.5].copy()
    left_attacking['needs_flip'] = True
    left_attacking = left_attacking.drop(columns=['start_x'], errors='ignore')

    # マージ
    events_merged = events.merge(
        left_attacking, 
        on=['game_id', 'period_id', 'team_id'], 
        how='left'
    )

    # 反転実行
    mask = events_merged['needs_flip'] == True
    for col in ['start_x', 'end_x']:
        events_merged.loc[mask, col] = 105.0 - events_merged.loc[mask, col]
    for col in ['start_y', 'end_y']:
        events_merged.loc[mask, col] = 68.0 - events_merged.loc[mask, col]

    events = events_merged
    print(f"正規化完了: {mask.sum()} 行を反転しました。")

    # =========================================================
    # 3. シュート選択確率の計算
    # =========================================================
    
    # グリッド設定
    x_bins = 16
    y_bins = 12
    x_edges = np.linspace(0, 105, x_bins + 1)
    y_edges = np.linspace(0, 68, y_bins + 1)

    # アクションの分類
    # シュート: type_name == 'shot' (結果に関わらず「打とうとした」意思をカウント)
    shots = events[events['type_name'] == 'shot']

    # 移動(パス・ドリブル): type_name が pass, dribble, cross など
    moves = events[events['type_name'].isin(['pass', 'dribble', 'cross'])]

    print(f"シュート試行回数: {len(shots)}")
    print(f"移動(パス等)試行回数: {len(moves)}")

    # グリッドごとのカウント
    shot_counts, _, _ = np.histogram2d(
        shots['start_x'], shots['start_y'], bins=[x_edges, y_edges]
    )
    move_counts, _, _ = np.histogram2d(
        moves['start_x'], moves['start_y'], bins=[x_edges, y_edges]
    )

    # 合計アクション数
    total_actions = shot_counts + move_counts

    # 確率計算 (シュート数 / 全アクション数)
    with np.errstate(divide='ignore', invalid='ignore'):
        shot_selection_prob = np.divide(shot_counts, total_actions)
        # アクションがない場所は0で埋める
        shot_selection_prob = np.nan_to_num(shot_selection_prob)

    # =========================================================
    # 4. 保存と可視化
    # =========================================================
    # 行列データを保存 (target_name付き)
    np.save(output_matrix_path, shot_selection_prob)
    print(f"確率行列を保存しました: {output_matrix_path.name}")

    # ヒートマップ表示
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        shot_selection_prob.T, 
        cmap="YlOrRd", 
        linewidths=0.1, 
        linecolor='gray',
        xticklabels=False, 
        yticklabels=False, 
        vmin=0, vmax=0.5 # 50%以上はめったにないのでレンジ調整
    )
    ax.invert_yaxis()
    
    plt.title(f"Shot Selection Probability (P_shoot) - {target_name.upper()}")
    plt.xlabel("Attack Direction ->")
    plt.ylabel("Pitch Width")

    plt.savefig(output_img_path)
    # 自動化のため close に変更
    plt.close()
    
    print(f"画像を保存しました: {output_img_path.name}")
    
    # 数値確認
    print("-" * 30)
    s = shot_counts[15, 6]
    m = move_counts[15, 6]
    p = shot_selection_prob[15, 6]
    print(f"ゴール前(15, 6)の選択率: シュート{int(s)}回 / 移動{int(m)}回 -> {p:.1%}")
    print("-" * 30)

# =========================================================
# 単体テスト用ブロック
# =========================================================
if __name__ == "__main__":
    # 単体で実行した場合のテスト
    run_analysis("england", r"C:\Users\shiba\wyscout_backup\csv_data")