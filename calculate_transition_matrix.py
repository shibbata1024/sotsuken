import pandas as pd
import numpy as np
import warnings
from pathlib import Path

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
    
    # 入力・出力パスの設定
    events_path = data_dir / f"spadl_events_{target_name}.csv"
    output_matrix_path = data_dir / f"transition_matrix_{target_name}.npy"

    print(f"\n[{target_name}] 遷移確率行列(T)の計算を開始します...")
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

    # 反転実行 (始点と終点の両方)
    mask = events_merged['needs_flip'] == True
    for col in ['start_x', 'end_x']:
        events_merged.loc[mask, col] = 105.0 - events_merged.loc[mask, col]
    for col in ['start_y', 'end_y']:
        events_merged.loc[mask, col] = 68.0 - events_merged.loc[mask, col]

    events = events_merged
    print(f"正規化完了: {mask.sum()} 行を反転しました。")

    # =========================================================
    # 3. 遷移確率行列 (Transition Matrix) の計算
    # =========================================================
    print("グリッド位置を計算中...")

    # グリッド設定 (16x12)
    x_bins = 16
    y_bins = 12
    w = 105 / x_bins
    h = 68 / y_bins

    # 座標をセルID(0~191)に変換する関数
    def get_cell_id(x, y):
        xi = np.clip(np.floor(x / w), 0, x_bins - 1).astype(int)
        yi = np.clip(np.floor(y / h), 0, y_bins - 1).astype(int)
        return xi * y_bins + yi # ID = x * 12 + y

    events['start_cell'] = get_cell_id(events['start_x'], events['start_y'])
    events['end_cell']   = get_cell_id(events['end_x'], events['end_y'])

    # ---------------------------------------------------------
    # 【重要】成功した移動アクションのみ抽出
    # ---------------------------------------------------------
    move_actions = events[
        events['type_name'].isin(['pass', 'dribble', 'cross']) & 
        (events['result_name'] == 'success')
    ].copy()

    print(f"有効な移動アクション数（成功のみ）: {len(move_actions)}")

    # 集計 (始点ID -> 終点ID の数を数える)
    transitions = move_actions.groupby(['start_cell', 'end_cell']).size().reset_index(name='count')

    # 行列の作成 (192 x 192)
    n_cells = x_bins * y_bins
    transition_matrix = np.zeros((n_cells, n_cells))

    for _, row in transitions.iterrows():
        s = int(row['start_cell'])
        e = int(row['end_cell'])
        transition_matrix[s, e] = row['count']

    # 確率に変換 (行ごとの合計で割る)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        # 合計が0の場所（データなし）は0のままにする
        T = np.divide(transition_matrix, row_sums)
        T = np.nan_to_num(T)

    # =========================================================
    # 4. 保存と確認
    # =========================================================
    np.save(output_matrix_path, T)
    print(f"遷移確率行列を保存しました: {output_matrix_path.name}")
    print(f"行列サイズ: {T.shape}")

    # データの確認 (センターサークル付近からのパス傾向)
    center_id = 8 * y_bins + 6 # x=8, y=6
    if center_id < n_cells:
        probs = T[center_id]
        top_indices = probs.argsort()[-3:][::-1] # 上位3つ

        print("-" * 30)
        print(f"確認: センターサークル(8, 6)からの移動先トップ3")
        for idx in top_indices:
            tx = idx // y_bins
            ty = idx % y_bins
            p = probs[idx]
            if p > 0:
                print(f" -> Grid({tx:2d}, {ty:2d}) へ: {p:.1%} の確率")
        print("-" * 30)

# =========================================================
# 単体テスト用ブロック
# =========================================================
if __name__ == "__main__":
    run_analysis("england", r"C:\Users\shiba\wyscout_backup\csv_data")