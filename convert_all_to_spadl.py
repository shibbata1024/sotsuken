import pandas as pd
from pathlib import Path
import warnings

# socceractionを使ってIDを名前に変換します
try:
    import socceraction.spadl as spadl
except ImportError:
    print("エラー: socceraction がインストールされていません。")
    exit()

warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================
# 設定
# =========================================================
data_dir = Path(r"C:\Users\shiba\wyscout_backup\csv_data")
output_file = data_dir / "spadl_events_all.csv"

print("フォルダ内の全CSVを探しています...")
all_files = list(data_dir.glob("spadl_events_*.csv"))

# 出力先と同じファイル名が既にあったら除外
if output_file in all_files:
    all_files.remove(output_file)

if not all_files:
    print("エラー: ファイルが見つかりません。")
    exit()

print(f"{len(all_files)} 個のファイルが見つかりました。結合を開始します...")

# =========================================================
# 読み込み & 結合
# =========================================================
df_list = []

# 読み込む列：名前(_name)ではなくID(_id)を指定するのがポイント
use_cols = [
    'game_id', 'period_id', 'team_id', 'player_id',
    'type_id', 'result_id',  # ここを変更
    'start_x', 'start_y', 'end_x', 'end_y'
]

for f in all_files:
    print(f" -> 読み込み中: {f.name}")
    try:
        # まずは指定した列だけで読み込んでみる
        df = pd.read_csv(f, usecols=use_cols)
        df_list.append(df)
    except ValueError:
        # 万が一、古いファイルで 'type_id' がなく 'type_name' しかない場合への保険
        try:
            # 全列読み込んで必要なものだけ残す方式に切り替え
            df = pd.read_csv(f)
            # もしIDがないなら、名前からIDへの逆変換は大変なので、このファイルは詳細確認が必要
            if 'type_id' not in df.columns and 'type_name' in df.columns:
                 print(f"    警告: {f.name} は古い形式(名前のみ)のため、このスクリプトでは扱えません。スキップします。")
                 continue
            df = df[use_cols]
            df_list.append(df)
        except Exception as e:
            print(f"    読み込みエラー (スキップ): {e}")

if not df_list:
    print("結合できるデータがありませんでした。")
    exit()

# 結合
print("データを結合しています...")
combined_events = pd.concat(df_list, ignore_index=True)
print(f"合計イベント数: {len(combined_events):,}")

# =========================================================
# ID -> 名前 の変換 (add_names)
# =========================================================
print("イベントIDを名前に変換しています (0 -> 'pass' 等)...")

# socceractionの便利な関数を使って、type_id -> type_name, result_id -> result_name を列に追加
combined_events = spadl.add_names(combined_events)

# =========================================================
# 保存
# =========================================================
print("一つのファイルに保存しています (数分かかります)...")

# 分析に必要な列だけを選んで保存（ファイルサイズ削減）
final_cols = [
    'game_id', 'period_id', 'team_id', 'player_id',
    'type_name', 'result_name', # 名前が入った列を保存
    'start_x', 'start_y', 'end_x', 'end_y'
]

combined_events[final_cols].to_csv(output_file, index=False)
print(f"完了しました: {output_file}")
print("これで calculate_xt_all.py が動くようになりました！")