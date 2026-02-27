import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import socceraction.spadl as spadl
import warnings

def run_analysis(target_name, data_dir):
    # 警告を無視
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 1. データの読み込み
    data_dir = Path(data_dir)
    events_path = data_dir / f"spadl_events_{target_name}.csv"
    output_matrix_path = data_dir / f"goal_probability_{target_name}.npy"
    output_img_path = data_dir / f"goal_probability_{target_name}.png"

    print(f"\n[{target_name}] データを読み込んでいます...")

    print("データを読み込んでいます...")
    events = pd.read_csv(events_path)
    if 'type_name' not in events.columns:
        print("IDを名前に変換しています...")
        events = spadl.add_names(events)

    # ゴール成功率行列の保存先の指定
    output_matrix_path = data_dir / f"goal_probability_{target_name}.npy"

    # =========================================================
    # 攻撃方向の自動検知と正規化 (Smart Normalization)
    # =========================================================
    # 手順:
    # 1. 「試合ID」「ハーフ(period_id)」「チームID」ごとにグループ化する。
    # 2. そのグループの「シュート(shot)」の平均X座標を計算する。
    # 3. 平均Xが 52.5 より小さい場合、「左に攻めている」と判断し、そのグループの全データを反転させる。

    print("攻撃方向を判定して正規化しています...")

    def normalize_direction(df_group):
        # このグループ（ある試合のあるハーフの片方のチーム）のシュートデータを抽出
        shots = df_group[df_group['type_name'] == 'shot']
        
        # シュートが1本もない場合は、パスなどの平均位置で代替（稀なケースへの保険）
        if len(shots) == 0:
            avg_x = df_group['start_x'].mean()
        else:
            avg_x = shots['start_x'].mean()
        
        # 平均位置が左半分(52.5未満)なら、左攻めとみなして反転
        if avg_x < 52.5:
            # X座標とY座標を反転 (105-x, 68-y)
            df_group['start_x'] = 105.0 - df_group['start_x']
            df_group['end_x'] = 105.0 - df_group['end_x']
            df_group['start_y'] = 68.0 - df_group['start_y']
            df_group['end_y'] = 68.0 - df_group['end_y']
            
        return df_group

    # グループごとに処理を適用 (少し時間がかかります)
    # ※処理高速化のため、必要な列だけを処理対象にする
    cols_to_fix = ['game_id', 'period_id', 'team_id', 'type_name', 'result_name', 'start_x', 'start_y', 'end_x', 'end_y']
    events_subset = events[cols_to_fix].copy()

    # pandasのgroupby + applyは遅いため、ベクトル計算で高速化するアプローチをとります
    # まず、各グループの「攻める向き」を判定したテーブルを作成
    direction_check = events_subset[events_subset['type_name'] == 'shot'].groupby(
        ['game_id', 'period_id', 'team_id']
    )['start_x'].mean().reset_index()

    # 左攻め(avg_x < 52.5)のグループを特定
    left_attacking_groups = direction_check[direction_check['start_x'] < 52.5]

    # マージ用のキーとしてフラグを立てる
    left_attacking_groups['needs_flip'] = True
    left_attacking_groups = left_attacking_groups.drop(columns=['start_x'])

    # 元データに結合
    events_merged = events.merge(
        left_attacking_groups, 
        on=['game_id', 'period_id', 'team_id'], 
        how='left'
    )

    # needs_flip が True の行だけ反転処理
    mask_flip = events_merged['needs_flip'] == True
    print(f"反転対象のイベント数: {mask_flip.sum()} / {len(events_merged)}")

    events_merged.loc[mask_flip, 'start_x'] = 105.0 - events_merged.loc[mask_flip, 'start_x']
    events_merged.loc[mask_flip, 'end_x']   = 105.0 - events_merged.loc[mask_flip, 'end_x']
    events_merged.loc[mask_flip, 'start_y'] = 68.0  - events_merged.loc[mask_flip, 'start_y']
    events_merged.loc[mask_flip, 'end_y']   = 68.0  - events_merged.loc[mask_flip, 'end_y']

    # 正規化完了データを上書き
    events = events_merged

    # =========================================================

    # 3. 再度ヒートマップ作成のための集計
    shots = events[events['type_name'] == 'shot'].copy()
    shots['is_goal'] = (shots['result_name'] == 'success').astype(int)

    # グリッド設定
    x_bins = 16
    y_bins = 12
    x_edges = np.linspace(0, 105, x_bins + 1)
    y_edges = np.linspace(0, 68, y_bins + 1)

    shot_counts, _, _ = np.histogram2d(shots['start_x'], shots['start_y'], bins=[x_edges, y_edges])
    goal_counts, _, _ = np.histogram2d(shots['start_x'], shots['start_y'], bins=[x_edges, y_edges], weights=shots['is_goal'])

    with np.errstate(divide='ignore', invalid='ignore'):
        goal_probability = np.divide(goal_counts, shot_counts)
        goal_probability = np.nan_to_num(goal_probability)

    # 閾値の設定（例: そのマスで最低5本以上のシュートデータがないと信頼しない）
    min_shots_threshold = 3

    with np.errstate(divide='ignore', invalid='ignore'):
        goal_probability = np.divide(goal_counts, shot_counts)
        goal_probability = np.nan_to_num(goal_probability)

    # 【追加】信頼性の低いマス（サンプル不足）を0にする
    goal_probability[shot_counts < min_shots_threshold] = 0

    # ゴール成功率行列の保存
    np.save(output_matrix_path, goal_probability)
    print(f"確率行列を保存しました: {output_matrix_path}")

    # 4. 可視化
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        goal_probability.T, 
        cmap="YlOrRd", 
        linewidths=0.1, 
        linecolor='gray',
        xticklabels=False, 
        yticklabels=False,
        vmin=0, vmax=0.3
    )
    ax.invert_yaxis()

    plt.title(f"Goal Probability of {target_name}")
    plt.xlabel("Attack Direction ->")
    plt.ylabel("Pitch Width")

    output_img = data_dir / f"goal_probability_{target_name}.png"
    plt.savefig(output_img)
    print(f"画像を保存しました: {output_img}")
    # plt.show()


    # 数値データのターミナル出力
    # 数値確認（ゴール前）
    print("\n【参考】ゴール前エリア(右端中央)の決定率:")
    print(f"エリア(15, 6): {goal_probability[15, 6]:.2%}")

    """ print("\n" + "="*65)
    print("【各座標(グリッド)のシュート成績一覧】")
    print(f"X軸: 0(自陣) -> 15(敵陣ゴール前)")
    print(f"Y軸: 0(右) -> 11(左)")
    print("-" * 65)
    print(f"{'Grid (x, y)':<15} | {'Shots':>8} | {'Goals':>8} | {'Prob (%)':>10}")
    print("-" * 65)

    # シュートがあった場所のみ表示（見やすくするため）
    # xは敵陣側（後半）から順に表示したほうがわかりやすいので逆順でループ
    for x in range(x_bins - 1, -1, -1):
        for y in range(y_bins):
            s = shot_counts[x, y]
            g = goal_counts[x, y]
            p = goal_probability[x, y]
            
            # シュートが1本以上ある場合のみ表示
            if s > 0:
                # 確率が高い場所（20%超え）にはマークをつけるなどしても良い
                mark = "(*)" if p > 0.25 else ""
                print(f"({x:2d}, {y:2d}) {mark:<8} | {int(s):8d} | {int(g):8d} | {p:9.1%}")

    print("-" * 65)
    print("(*) = 決定率25%以上のエリア") """

if __name__ == "__main__":
    run_analysis("germany", r"C:\Users\shiba\wyscout_backup\csv_data")