import sys
from pathlib import Path

import calculate_goal_prob
import calculate_move_success
import calculate_shot_selection
import calculate_transition_matrix
import calculate_xT


def main():
    data_dir = Path(r"C:\Users\shiba\wyscout_backup\csv_data\spadl_teams")
    
    # 計算したい国・コンペティションのリスト
    targets = [
        "Newcastle_United",
        "Celta_de_Vigo",
        "Espanyol",
        "Deportivo_Alav\u00e9s",
        "Levante",
        "Troyes",
        "Getafe",
        "Borussia_M'gladbach",
        "Huddersfield_Town",
        "Athletic_Club",
        "Atl\u00e9tico_Madrid",
        "Olympique_Lyonnais",
        "PSG",
        "Valencia",
        "Real_Madrid",
        "Barcelona",
        "Las_Palmas",
        "Legan\u00e9s",
        "SPAL",
        "Swansea_City",
        "Olympique_Marseille",
        "Nantes",
        "Nice",
        "Rennes",
        "Strasbourg",
        "Eibar",
        "AFC_Bournemouth",
        "Brighton_&_Hove_Albion",
        "Werder_Bremen",
        "Bayer_Leverkusen",
        "Borussia_Dortmund",
        "Bayern_M\u00fcnchen",
        "Stuttgart",
        "Schalke_04",
        "Milan",
        "Angers",
        "Juventus",
        "Roma",
        "Sassuolo",
        "Burnley",
        "Bordeaux",
        "Hannover_96",
        "Dijon",
        "Hertha_BSC",
        "Wolfsburg",
        "Hamburger_SV",
        "Freiburg",
        "Bologna",
        "Metz",
        "Sampdoria",
        "Chievo",
        "Lazio",
        "Udinese",
        "Internazionale",
        "Leicester_City",
        "West_Ham_United",
        "Stoke_City",
        "Benevento",
        "Saint-\u00c9tienne",
        "Girona",
        "Watford",
        "Hoffenheim",
        "Cagliari",
        "Atalanta",
        "Fiorentina",
        "Everton",
        "West_Bromwich_Albion",
        "Manchester_City",
        "Tottenham_Hotspur",
        "Augsburg",
        "Crystal_Palace",
        "Monaco",
        "Mainz_05",
        "Lille",
        "Eintracht_Frankfurt",
        "Southampton",
        "K\u00f6ln",
        "Liverpool",
        "Chelsea",
        "Manchester_United",
        "Torino",
        "Napoli",
        "Deportivo_La_Coru\u00f1a",
        "RB_Leipzig",
        "Arsenal",
        "Caen",
        "Toulouse",
        "Montpellier",
        "Guingamp",
        "Amiens_SC",
        "Crotone",
        "Hellas_Verona",
        "Genoa",
        "Real_Betis",
        "Real_Sociedad",
        "Sevilla",
        "M\u00e1laga",
        "Villarreal",
    ]

    for competition in targets:
        print("\n" + "="*60)
        print(f"START PROCESSING: {competition.upper()}")
        print("="*60)

        try:
            # 1. ゴール確率 (Goal Probability)
            calculate_goal_prob.run_analysis(competition, data_dir)

            # 2. 移動成功率 (Move Success Probability)
            calculate_move_success.run_analysis(competition, data_dir)

            # 3. シュート選択率 (Shot Selection Probability)
            calculate_shot_selection.run_analysis(competition, data_dir)

            # 4. 遷移確率行列 (Transition Matrix)
            calculate_transition_matrix.run_analysis(competition, data_dir)

            # 5. xT算出 (Expected Threat)
            calculate_xT.run_analysis(competition, data_dir)
            
            print(f"\n成功: {competition} のxT計算が完了しました。")

        except Exception as e:
            print(f"\nエラー発生: {competition} の処理中に問題が起きました。")
            print(f"詳細: {e}")
            print("次の国へ進みます...")
            continue # エラーが出ても止まらず、次の国へ

    print("\n" + "="*60)
    print("全ての処理が終了しました。")

if __name__ == "__main__":
    main()
    