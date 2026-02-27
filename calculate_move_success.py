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
    output_matrix_path = data_dir / f"move_success_prob_{target_name}.npy"
    output_img_path = data_dir / f"move_success_prob_{target_name}.png"

    print(f"\n[{target_name}] 移動成功率(パス/ドリブル)の計算を開始します...")
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

    # 左攻め判定
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
    
    # 成功率は始点(start)が重要だが、念のため終点(end)も反転しておく
    for col in ['start_x', 'end_x']:
        events_merged.loc[mask, col] = 105.0 - events_merged.loc[mask, col]
    for col in ['start_y', 'end_y']:
        events_merged.loc[mask, col] = 68.0 - events_merged.loc[mask, col]

    events = events_merged
    print(f"正規化完了: {mask.sum()} 行を反転しました。")

    # =========================================================
    # 3. 移動成功率 (Move Success Probability) の計算
    # =========================================================

    # グリッド設定 (16x12)
    x_bins = 16
    y_bins = 12
    x_edges = np.linspace(0, 105, x_bins + 1)
    y_edges = np.linspace(0, 68, y_bins + 1)

    # --- 対象アクションの抽出 ---
    # 移動を試みた全アクション (分母)
    move_attempts = events[events['type_name'].isin(['pass', 'dribble', 'cross'])]

    # そのうち成功したアクション (分子)
    move_success = move_attempts[move_attempts['result_name'] == 'success']

    print(f"移動試行回数: {len(move_attempts)}")
    print(f"移動成功回数: {len(move_success)}")

    # --- グリッドごとの集計 ---
    # 試行数 (Attempts)
    attempt_counts, _, _ = np.histogram2d(
        move_attempts['start_x'], move_attempts['start_y'], 
        bins=[x_edges, y_edges]
    )

    # 成功数 (Successes)
    success_counts, _, _ = np.histogram2d(
        move_success['start_x'], move_success['start_y'], 
        bins=[x_edges, y_edges]
    )

    # --- 確率計算 (成功数 / 試行数) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        move_success_prob = np.divide(success_counts, attempt_counts)
        # データがない場所(0/0)はNaNになるので0で埋める
        move_success_prob = np.nan_to_num(move_success_prob)

    # =========================================================
    # 4. 保存と可視化
    # =========================================================

    # 行列データを保存 (後のxT計算で使用)
    np.save(output_matrix_path, move_success_prob)
    print(f"成功率行列を保存しました: {output_matrix_path.name}")

    # ヒートマップ作成
    plt.figure(figsize=(12, 8))

    # 赤(失敗) -> 黄 -> 緑(成功) のグラデーション
    ax = sns.heatmap(
        move_success_prob.T, 
        cmap="RdYlGn", 
        linewidths=0.1, 
        linecolor='gray',
        xticklabels=False, 
        yticklabels=False,
        vmin=0.4, vmax=1.0 # 40%〜100%の範囲で色付け
    )
    ax.invert_yaxis() # y=0を下に

    plt.title(f"Move Success Probability ({target_name.upper()})")
    plt.xlabel("Attack Direction -> (Safe zone vs Risky zone)")
    plt.ylabel("Pitch Width")

    plt.savefig(output_img_path)
    # 自動化のため close
    plt.close()
    
    print(f"ヒートマップ画像を保存しました: {output_img_path.name}")

    # --- 数値確認 (エリアごとの特徴) ---
    print("-" * 30)
    # 自陣 (左側)
    print(f"自陣ビルドアップエリア (4, 6): {move_success_prob[4, 6]:.1%}")
    # 敵陣 (右側)
    print(f"敵陣バイタルエリア (14, 6):   {move_success_prob[14, 6]:.1%}")
    print("-" * 30)

# =========================================================
# 単体テスト用ブロック
# =========================================================
if __name__ == "__main__":
    run_analysis("england", r"C:\Users\shiba\wyscout_backup\csv_data")