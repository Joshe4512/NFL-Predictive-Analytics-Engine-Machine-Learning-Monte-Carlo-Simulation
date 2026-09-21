"""
NFL Game Predictor & Analytics Pipeline
---------------------------------------
This script extracts historical NFL game data directly from open-source servers,
engineers predictive features (rolling averages), trains a Random Forest Machine
Learning classifier, and generates visual EDA reports. Finally, it performs batch
inference using a Monte Carlo simulation (predict_proba) to generate highly
realistic, variance-adjusted season records.

*Feature Imputation added to dynamically calculate TV Broadcast Networks
based on NFL scheduling logic.

Author: Joshua Evans
Target Roles: Data Science, Product Management, Tech Support (IBM)
"""

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix


# ==========================================
# PHASE 1: DATA EXTRACTION
# ==========================================

def build_dataset(start_year, end_year, filename="nfl_matches.csv"):
    print(f"[*] Fetching real NFL data for {start_year}-{end_year} directly from source...")
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    df = pd.read_csv(url)

    df = df[(df['season'] >= start_year) & (df['season'] <= end_year) & (df['game_type'] != 'PRE')]
    df = df.dropna(subset=['home_score', 'away_score'])

    clean_games = []
    for index, row in df.iterrows():
        date = str(row['gameday'])
        home_team, away_team = row['home_team'], row['away_team']
        home_score, away_score = row['home_score'], row['away_score']

        clean_games.append({
            "date": date, "team": home_team, "opponent": away_team, "venue": "Home",
            "points_for": home_score, "points_against": away_score,
            "result": "W" if home_score > away_score else "L"
        })
        clean_games.append({
            "date": date, "team": away_team, "opponent": home_team, "venue": "Away",
            "points_for": away_score, "points_against": home_score,
            "result": "W" if away_score > home_score else "L"
        })

    final_df = pd.DataFrame(clean_games)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df.to_csv(filename, index=False)
    print(f"[*] Success! Saved {len(final_df)} game records to {filename}")
    return final_df


# ==========================================
# PHASE 2: DATA VISUALIZATION
# ==========================================

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})


def ensure_asset_dir():
    if not os.path.exists("assets"): os.makedirs("assets")


def plot_eda_points(df):
    ensure_asset_dir()
    plt.figure(figsize=(7, 4))
    df['Outcome'] = df['target'].map({1: 'Win', 0: 'Loss'})
    sns.boxplot(x='Outcome', y='points_for', hue='Outcome', data=df, palette={'Win': '#198038', 'Loss': '#da1e28'},
                legend=False)
    plt.title("Game Outcomes vs. Offensive Points Scored", fontweight="bold")
    plt.xlabel("Match Result")
    plt.ylabel("Points Scored in Game")
    plt.tight_layout()
    plt.savefig("assets/points_eda.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    ensure_asset_dir()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Loss', 'Predicted Win'],
                yticklabels=['Actual Loss', 'Actual Win'])
    plt.title("Model Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png")
    plt.close()


def plot_feature_importance(importances, feature_names):
    ensure_asset_dir()
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_df, color="#0f62fe")
    plt.title("Random Forest Feature Importance", fontweight="bold")
    plt.xlabel("Relative Predictive Weight")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png")
    plt.close()


# ==========================================
# PHASE 3: MACHINE LEARNING PIPELINE
# ==========================================

def run_ml_pipeline(df):
    print("[*] Engineering features (Rolling Averages)...")
    df["venue_code"] = df["venue"].astype("category").cat.codes
    df["opp_code"] = df["opponent"].astype("category").cat.codes
    df["target"] = (df["result"] == "W").astype(int)

    df = df.sort_values(by=["team", "date"]).reset_index(drop=True)
    cols = ["points_for", "points_against"]
    for c in cols:
        df[f"{c}_rolling"] = df.groupby("team")[c].transform(lambda x: x.rolling(3, closed='left').mean())

    df_rolling = df.dropna().reset_index(drop=True)

    print("[*] Training Random Forest Classifier...")
    df_rolling = df_rolling.sort_values("date")
    train = df_rolling.iloc[:-100]
    test = df_rolling.iloc[-100:]

    predictors = ["venue_code", "opp_code", "points_for_rolling", "points_against_rolling"]
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train["target"])

    preds = rf.predict(test[predictors])
    precision = precision_score(test["target"], preds)
    print(f"[*] Model Precision Score: {precision * 100:.1f}%")

    plot_eda_points(df_rolling)
    plot_confusion_matrix(test["target"], preds)
    plot_feature_importance(rf.feature_importances_, ["Venue", "Opponent", "Points For (Avg)", "Points Against (Avg)"])

    return rf, df, df_rolling


# ==========================================
# PHASE 4: MONTE CARLO BATCH INFERENCE
# ==========================================

def predict_remaining_season(model, df, df_rolling, season=2026):
    print(f"\n🔮 MONTE CARLO SIMULATION: PREDICTING THE REST OF {season} 🔮")
    print("-" * 65)

    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    schedule = pd.read_csv(url)

    # 1. Calculate current real-world records
    season_start = f"{season}-08-01"
    current_season_games = df[df['date'] >= season_start]
    records = {team: {'W': 0, 'L': 0} for team in df['team'].unique()}
    for _, row in current_season_games.iterrows():
        if row['team'] in records:
            if row['result'] == 'W':
                records[row['team']]['W'] += 1
            else:
                records[row['team']]['L'] += 1

    # 2. Calculate true historical baselines for each team
    baseline_stats = {}
    for team in df['team'].unique():
        team_games = df[df['team'] == team]
        baseline_stats[team] = {'pf': team_games['points_for'].mean(), 'pa': team_games['points_against'].mean()}

    # 3. Load up current 3-game momentum
    current_momentum = {}
    for team in df['team'].unique():
        try:
            latest = df_rolling[df_rolling['team'] == team].iloc[-1]
            current_momentum[team] = {'pf_rolling': latest['points_for_rolling'],
                                      'pa_rolling': latest['points_against_rolling']}
        except:
            current_momentum[team] = {'pf_rolling': 20.0, 'pa_rolling': 20.0}

    # 4. Filter future schedule
    future_games = schedule[
        (schedule['season'] == season) & (schedule['home_score'].isna()) & (schedule['game_type'] != 'PRE')].copy()
    future_games = future_games.sort_values("week")

    print(f"[*] Found {len(future_games)} unplayed games. Running probability simulator...\n")

    predictions_list = []
    team_to_code = dict(zip(df['opponent'], df['opp_code']))

    current_week = 0

    for _, row in future_games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        week = int(row['week'])

        # --- Date & Network Feature Imputation ---
        raw_time = row.get('gametime', 'TBD')
        weekday = row.get('weekday', 'TBD')

        if pd.isna(raw_time):
            formatted_time = "TBD EST"
            network = "TBD"
        else:
            try:
                # Convert 24hr string to 12hr AM/PM
                formatted_time = pd.to_datetime(str(raw_time)).strftime('%I:%M %p').lstrip('0') + " EST"

                # Heuristically predict the TV Network
                if weekday == "Thursday":
                    network = "Prime Video"
                elif weekday == "Monday":
                    network = "ESPN/ABC"
                elif weekday == "Sunday":
                    if "20:" in str(raw_time) or "21:" in str(raw_time):
                        network = "NBC"
                    elif "09:" in str(raw_time) or "10:" in str(raw_time):
                        network = "NFL Network"
                    else:
                        network = "CBS/FOX"
                else:
                    network = "NFL Network"
            except Exception:
                formatted_time = f"{raw_time} EST"
                network = "TBD"
        # ----------------------------------------

        try:
            if week != current_week:
                if current_week != 0:
                    for team in current_momentum:
                        base_pf = baseline_stats[team]['pf']
                        base_pa = baseline_stats[team]['pa']
                        current_momentum[team]['pf_rolling'] += (base_pf - current_momentum[team]['pf_rolling']) * 0.15
                        current_momentum[team]['pa_rolling'] += (base_pa - current_momentum[team]['pa_rolling']) * 0.15

                print(f"\nWeek {week}")
                print("-" * 65)
                current_week = week

            home_stats = current_momentum.get(home_team)
            away_stats = current_momentum.get(away_team)
            home_opp_code = team_to_code.get(away_team, 0)
            away_opp_code = team_to_code.get(home_team, 0)

            home_features = pd.DataFrame([{
                "venue_code": 1, "opp_code": home_opp_code,
                "points_for_rolling": home_stats['pf_rolling'],
                "points_against_rolling": home_stats['pa_rolling']
            }])
            away_features = pd.DataFrame([{
                "venue_code": 0, "opp_code": away_opp_code,
                "points_for_rolling": away_stats['pf_rolling'],
                "points_against_rolling": away_stats['pa_rolling']
            }])

            home_win_prob = model.predict_proba(home_features)[0][1]
            away_win_prob = model.predict_proba(away_features)[0][1]

            total_prob = home_win_prob + away_win_prob
            normalized_home_prob = 0.5 if total_prob == 0 else home_win_prob / total_prob

            dice_roll = random.random()

            if dice_roll <= normalized_home_prob:
                winner, loser = home_team, away_team
                upset_marker = " (UPSET!)" if normalized_home_prob < 0.5 else ""
            else:
                winner, loser = away_team, home_team
                upset_marker = " (UPSET!)" if normalized_home_prob >= 0.5 else ""

            if winner in records: records[winner]['W'] += 1
            if loser in records: records[loser]['L'] += 1

            winner_record = f"{records[winner]['W']}-{records[winner]['L']}"

            home_pct = round(normalized_home_prob * 100, 1)
            away_pct = round((1 - normalized_home_prob) * 100, 1)
            odds_string = f"[{away_team} {away_pct}% | {home_team} {home_pct}%]"

            # Formats Output: team 1 @ team 2 @ time est (Day on Network) -- winner wins (record)
            print(
                f"{away_team} @ {home_team} @ {formatted_time} ({weekday} on {network}) -- {winner} wins ({winner_record}) {odds_string}{upset_marker}")

            predictions_list.append({
                "Week": week,
                "Day": weekday,
                "Time": formatted_time,
                "Network": network,
                "Away Team": away_team,
                "Home Team": home_team,
                "Predicted Winner": winner,
                "Winner Simulated Record": winner_record,
                "Home Win Probability": f"{home_pct}%"
            })
        except Exception as e:
            continue

    results_df = pd.DataFrame(predictions_list)
    filename = f"{season}_season_predictions.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n[*] Complete! Exported all batch predictions to {filename}")


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    csv_filename = "nfl_matches.csv"

    if os.path.exists(csv_filename):
        print(f"[*] Deleting old {csv_filename} to fetch the latest stats...")
        os.remove(csv_filename)

    df = build_dataset(2022, 2026, csv_filename)
    trained_model, raw_data, rolling_data = run_ml_pipeline(df)
    predict_remaining_season(trained_model, raw_data, rolling_data, season=2026)