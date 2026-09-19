
#MLB win-probability project
#===========================

#Project that:
##  1. downloads free MLB game data,
##  2. converts physical/game ideas into numerical features,
##  3. trains a small neural network,
##  4. compares it with logistic regression, and
##  5. reports out-of-sample performance.



# 1. Imports and experiment settings


from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEASONS = [2023, 2024, 2025]
MIN_GAMES = 20
TEST_FRACTION = 0.20
RANDOM_SEED = 7

API = "https://statsapi.mlb.com/api"
CACHE = Path("mlb_cache_student_project")
OUTPUT = Path("mlb_results")


# Every difference is defined so that positive means an advantage for the
# home team.  This makes coefficients and model behavior easier to interpret.
FEATURES = [
    "home_field",
    "starter_era_advantage",
    "starter_whip_advantage",
    "starter_kbb_advantage",
    "offense_rpg_advantage",
    "offense_ops_advantage",
    "bullpen_era_advantage",
    "bullpen_fatigue_advantage",
    "home_record_advantage",
    "park_run_factor",
    "rest_advantage",
    "travel_advantage",
]


# %% 2. Small numerical and data-access functions

def safe_divide(numerator, denominator, default=0.0):
    """Division with a physically reasonable value when no sample exists."""
    return numerator / denominator if denominator else default


def innings_to_outs(innings):
    """MLB writes 5 innings + 2 outs as 5.2, not as a decimal number."""
    whole, _, fraction = str(innings or "0.0").partition(".")
    return 3 * int(whole) + int(fraction or 0)


def distance_miles(point_1, point_2):
    """Great-circle distance using the haversine equation."""
    if point_1 is None or point_2 is None:
        return 0.0

    radius = 3958.8
    lat_1, lon_1 = np.radians(point_1)
    lat_2, lon_2 = np.radians(point_2)
    dlat = lat_2 - lat_1
    dlon = lon_2 - lon_1

    a = np.sin(dlat / 2) ** 2
    a += np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def download_json(url, cache_file, parameters=None, refresh=False):
    """Download once, then use the local cached copy on later runs."""
    cache_file = CACHE / cache_file
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    response = requests.get(url, params=parameters, timeout=30)
    response.raise_for_status()
    data = response.json()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    time.sleep(0.05)  # Be polite to the free server.
    return data


def get_schedule(season, refresh=False):
    data = download_json(
        f"{API}/v1/schedule",
        f"schedule_{season}.json",
        {
            "sportId": 1,
            "startDate": f"{season}-03-15",
            "endDate": f"{season}-11-10",
            "hydrate": "venue,probablePitcher",
        },
        refresh=refresh,
    )
    games = [game for day in data.get("dates", []) for game in day.get("games", [])]
    final_states = {"Final", "Game Over", "Completed Early"}
    return [g for g in games if g["status"]["detailedState"] in final_states]


def get_games_on_date(date_text):
    """Retrieve the live schedule for exactly the date selected by the user."""
    data = download_json(
        f"{API}/v1/schedule",
        f"daily_schedules/{date_text}.json",
        {
            "sportId": 1,
            "date": date_text,
            "hydrate": "venue,probablePitcher",
        },
        refresh=True,
    )
    return [game for day in data.get("dates", []) for game in day.get("games", [])]


def get_game_feed(game_id):
    return download_json(
        f"{API}/v1.1/game/{game_id}/feed/live",
        f"games/{game_id}.json",
    )


def get_venue_coordinates(venue_id):
    data = download_json(
        f"{API}/v1/venues/{venue_id}",
        f"venues/{venue_id}.json",
    )
    venue = data["venues"][0]
    coordinates = venue.get("location", {}).get("defaultCoordinates", {})
    if "latitude" not in coordinates or "longitude" not in coordinates:
        return None
    return float(coordinates["latitude"]), float(coordinates["longitude"])


# %% 3. State variables

def new_batting_state():
    return {
        "games": 0, "runs": 0, "ab": 0, "hits": 0,
        "doubles": 0, "triples": 0, "hr": 0,
        "walks": 0, "hbp": 0, "sac_flies": 0,
    }


def new_pitching_state():
    return {"outs": 0, "earned_runs": 0, "hits": 0, "walks": 0, "strikeouts": 0}


def new_team_state():
    return {
        "batting": new_batting_state(),
        "bullpen": new_pitching_state(),
        "bullpen_history": [],
        "last_date": None,
        "last_location": None,
        "home_games": 0,
        "home_wins": 0,
    }


def era(pitching):
    return safe_divide(27 * pitching["earned_runs"], pitching["outs"], 4.50)


def whip(pitching):
    numerator = 3 * (pitching["walks"] + pitching["hits"])
    return safe_divide(numerator, pitching["outs"], 1.30)


def kbb(pitching):
    return safe_divide(pitching["strikeouts"], pitching["walks"], 2.50)


def runs_per_game(batting):
    return safe_divide(batting["runs"], batting["games"], 4.50)


def ops(batting):
    singles = batting["hits"] - batting["doubles"] - batting["triples"] - batting["hr"]
    total_bases = singles + 2 * batting["doubles"] + 3 * batting["triples"] + 4 * batting["hr"]

    obp_denominator = batting["ab"] + batting["walks"] + batting["hbp"] + batting["sac_flies"]
    obp = safe_divide(batting["hits"] + batting["walks"] + batting["hbp"], obp_denominator, 0.310)
    slugging = safe_divide(total_bases, batting["ab"], 0.400)
    return obp + slugging


def add_pitching_line(state, line, sign=1):
    state["outs"] += sign * innings_to_outs(line.get("inningsPitched"))
    state["earned_runs"] += sign * float(line.get("earnedRuns", 0) or 0)
    state["hits"] += sign * float(line.get("hits", 0) or 0)
    state["walks"] += sign * float(line.get("baseOnBalls", 0) or 0)
    state["strikeouts"] += sign * float(line.get("strikeOuts", 0) or 0)


# %% 4. Extract one completed game

def extract_game(feed):
    game_data = feed["gameData"]
    box = feed["liveData"]["boxscore"]["teams"]

    record = {
        "date": datetime.strptime(game_data["datetime"]["officialDate"], "%Y-%m-%d").date(),
        "venue_id": int(game_data["venue"]["id"]),
        "home_id": int(game_data["teams"]["home"]["id"]),
        "away_id": int(game_data["teams"]["away"]["id"]),
        "home_name": game_data["teams"]["home"]["name"],
        "away_name": game_data["teams"]["away"]["name"],
    }

    for side in ["home", "away"]:
        side_box = box[side]
        pitcher_ids = side_box.get("pitchers", [])
        starter_id = int(pitcher_ids[0])
        starter = side_box["players"][f"ID{starter_id}"]["stats"]["pitching"]

        record[f"{side}_starter_id"] = starter_id
        record[f"{side}_starter"] = starter
        record[f"{side}_batting"] = side_box["teamStats"]["batting"]
        record[f"{side}_pitching"] = side_box["teamStats"]["pitching"]
        record[f"{side}_runs"] = int(side_box["teamStats"]["batting"].get("runs", 0))

    return record


def extract_scheduled_game(scheduled_game):
    """Extract information known before an upcoming game begins."""
    teams = scheduled_game["teams"]
    return {
        "game_id": int(scheduled_game["gamePk"]),
        "date": datetime.strptime(scheduled_game["officialDate"], "%Y-%m-%d").date(),
        "venue_id": int(scheduled_game["venue"]["id"]),
        "home_id": int(teams["home"]["team"]["id"]),
        "away_id": int(teams["away"]["team"]["id"]),
        "home_name": teams["home"]["team"]["name"],
        "away_name": teams["away"]["team"]["name"],
        "home_starter_id": teams["home"].get("probablePitcher", {}).get("id"),
        "away_starter_id": teams["away"].get("probablePitcher", {}).get("id"),
    }


# %% 5. Convert each game into a pregame feature vector

def starter_summary(pitcher_state):
    # Use league-average priors until the pitcher has thrown 15 innings.
    if pitcher_state is None or pitcher_state["outs"] < 45:
        return 4.50, 1.30, 2.50
    return era(pitcher_state), whip(pitcher_state), kbb(pitcher_state)


def make_feature_row(game, team_states, pitcher_states, park_runs, league_runs, location):
    home = team_states[game["home_id"]]
    away = team_states[game["away_id"]]

    home_starter = starter_summary(pitcher_states.get(game["home_starter_id"]))
    away_starter = starter_summary(pitcher_states.get(game["away_starter_id"]))

    # Bullpen workload accumulated during the previous three calendar days.
    home_recent_outs = sum(
        outs for day, outs in home["bullpen_history"]
        if 0 < (game["date"] - day).days <= 3
    )
    away_recent_outs = sum(
        outs for day, outs in away["bullpen_history"]
        if 0 < (game["date"] - day).days <= 3
    )

    home_rest = (game["date"] - home["last_date"]).days - 1 if home["last_date"] else 3
    away_rest = (game["date"] - away["last_date"]).days - 1 if away["last_date"] else 3

    league_mean = np.mean(league_runs) if league_runs else 9.0
    park_mean = np.mean(park_runs[game["venue_id"]]) if park_runs[game["venue_id"]] else league_mean

    row = {
        "date": game["date"].isoformat(),
        "away_team": game["away_name"],
        "home_team": game["home_name"],
        "home_field": 1.0,
        # ERA and WHIP are reversed because smaller values are better.
        "starter_era_advantage": away_starter[0] - home_starter[0],
        "starter_whip_advantage": away_starter[1] - home_starter[1],
        "starter_kbb_advantage": home_starter[2] - away_starter[2],
        "offense_rpg_advantage": runs_per_game(home["batting"]) - runs_per_game(away["batting"]),
        "offense_ops_advantage": ops(home["batting"]) - ops(away["batting"]),
        "bullpen_era_advantage": era(away["bullpen"]) - era(home["bullpen"]),
        # A tired away bullpen is an advantage for the home team.
        "bullpen_fatigue_advantage": away_recent_outs - home_recent_outs,
        "home_record_advantage": safe_divide(home["home_wins"], home["home_games"], 0.54) - 0.50,
        "park_run_factor": safe_divide(park_mean, league_mean, 1.0),
        "rest_advantage": np.clip(home_rest, 0, 7) - np.clip(away_rest, 0, 7),
        # More away-team travel than home-team travel is positive.
        "travel_advantage": (
            distance_miles(away["last_location"], location)
            - distance_miles(home["last_location"], location)
        ) / 1000.0,
        "home_games_before": home["batting"]["games"],
        "away_games_before": away["batting"]["games"],
    }

    # Completed historical games have a known answer. Upcoming games do not.
    if "home_runs" in game:
        row["home_win"] = int(game["home_runs"] > game["away_runs"])
    if "game_id" in game:
        row["game_id"] = game["game_id"]
    return row


def update_states(game, team_states, pitcher_states, park_runs, league_runs, location):
    """Update only after constructing the row: this prevents data leakage."""
    for side in ["home", "away"]:
        team = team_states[game[f"{side}_id"]]
        batting_line = game[f"{side}_batting"]

        mapping = {
            "runs": "runs", "ab": "atBats", "hits": "hits",
            "doubles": "doubles", "triples": "triples", "hr": "homeRuns",
            "walks": "baseOnBalls", "hbp": "hitByPitch", "sac_flies": "sacFlies",
        }
        team["batting"]["games"] += 1
        for state_name, api_name in mapping.items():
            team["batting"][state_name] += float(batting_line.get(api_name, 0) or 0)

        starter_line = game[f"{side}_starter"]
        add_pitching_line(pitcher_states[game[f"{side}_starter_id"]], starter_line)

        bullpen_line = new_pitching_state()
        add_pitching_line(bullpen_line, game[f"{side}_pitching"])
        add_pitching_line(bullpen_line, starter_line, sign=-1)
        for key in bullpen_line:
            team["bullpen"][key] += bullpen_line[key]

        team["bullpen_history"].append((game["date"], bullpen_line["outs"]))
        team["last_date"] = game["date"]
        team["last_location"] = location

    home = team_states[game["home_id"]]
    home["home_games"] += 1
    home["home_wins"] += int(game["home_runs"] > game["away_runs"])

    total_runs = game["home_runs"] + game["away_runs"]
    park_runs[game["venue_id"]].append(total_runs)
    league_runs.append(total_runs)


def build_dataset(seasons, prediction_date):
    target_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()
    team_states = defaultdict(new_team_state)
    pitcher_states = defaultdict(new_pitching_state)
    park_runs = defaultdict(list)
    league_runs = []
    rows = []

    scheduled_games = []
    for season in seasons:
        # Refresh the target season because its schedule and probable pitchers
        # can change during the season.
        scheduled_games.extend(get_schedule(season, refresh=(season == target_date.year)))
    scheduled_games.sort(key=lambda game: game["gameDate"])

    # Strictly exclude the selected date and everything after it. This lets a
    # historical date be tested without allowing the model to see the answer.
    prior_games = [
        game for game in scheduled_games
        if datetime.strptime(game["officialDate"], "%Y-%m-%d").date() < target_date
    ]

    for index, scheduled_game in enumerate(prior_games, start=1):
        try:
            game = extract_game(get_game_feed(scheduled_game["gamePk"]))
            location = get_venue_coordinates(game["venue_id"])
            row = make_feature_row(
                game, team_states, pitcher_states, park_runs, league_runs, location
            )
            rows.append(row)
            update_states(
                game, team_states, pitcher_states, park_runs, league_runs, location
            )
        except (KeyError, ValueError, requests.RequestException) as error:
            print(f"Skipping game {scheduled_game['gamePk']}: {error}")

        if index % 100 == 0:
            print(f"Processed {index} of {len(prior_games)} games")

    data = pd.DataFrame(rows)
    enough_history = (
        (data["home_games_before"] >= MIN_GAMES)
        & (data["away_games_before"] >= MIN_GAMES)
    )
    data = data.loc[enough_history].reset_index(drop=True)

    # Create feature rows for the requested slate without updating any state.
    # Those games therefore contain no information from their own results.
    slate_rows = []
    for scheduled_game in get_games_on_date(prediction_date):
        try:
            game = extract_scheduled_game(scheduled_game)
            location = get_venue_coordinates(game["venue_id"])
            slate_rows.append(
                make_feature_row(
                    game, team_states, pitcher_states, park_runs, league_runs, location
                )
            )
        except (KeyError, ValueError, requests.RequestException) as error:
            print(f"Skipping scheduled game {scheduled_game.get('gamePk')}: {error}")

    return data, pd.DataFrame(slate_rows)


# %% 6. Model experiment

def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, prediction),
        "log_loss": log_loss(y_test, probability),
        "brier_score": brier_score_loss(y_test, probability),
        "roc_auc": roc_auc_score(y_test, probability),
    }
    return result, probability, model


def run_experiment(data):
    # Chronological splitting imitates predicting genuinely future games.
    split = int((1 - TEST_FRACTION) * len(data))
    train = data.iloc[:split]
    test = data.iloc[split:]

    x_train, y_train = train[FEATURES], train["home_win"]
    x_test, y_test = test[FEATURES], test["home_win"]

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_SEED),
    )

    neural_network = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(12,),
            activation="relu",
            alpha=1.0,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_SEED,
        ),
    )

    results = []
    predictions = test[["date", "away_team", "home_team", "home_win"]].copy()

    for name, model in [("Logistic regression", logistic), ("Neural network", neural_network)]:
        result, probability, fitted_model = evaluate_model(
            name, model, x_train, y_train, x_test, y_test
        )
        results.append(result)
        predictions[name] = probability

    # Evaluation above uses an untouched future test set. After measuring the
    # model, refit the neural network on every completed game available before
    # the requested date so the live prediction uses all legitimate evidence.
    final_model = clone(neural_network)
    final_model.fit(data[FEATURES], data["home_win"])

    return pd.DataFrame(results), predictions, final_model


def predict_winners(model, slate):
    """Convert home-win probabilities into team names and confidence values."""
    if slate.empty:
        return slate

    output = slate[["game_id", "date", "away_team", "home_team"]].copy()
    home_probability = model.predict_proba(slate[FEATURES])[:, 1]
    output["predicted_winner"] = np.where(
        home_probability >= 0.5,
        output["home_team"],
        output["away_team"],
    )
    output["win_probability"] = np.where(
        home_probability >= 0.5,
        home_probability,
        1.0 - home_probability,
    )
    output["home_win_probability"] = home_probability
    return output


# %% 7. Run the complete calculation

def main():
    CACHE.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)

    prediction_date = input("Enter the game date (YYYY-MM-DD): ").strip()
    try:
        target_year = datetime.strptime(prediction_date, "%Y-%m-%d").year
    except ValueError:
        print("Invalid date. Use the format YYYY-MM-DD, for example 2026-09-19.")
        return

    if target_year < min(SEASONS):
        print(f"Choose {min(SEASONS)} or later so enough training data are available.")
        return

    seasons = [season for season in SEASONS if season <= target_year]
    if target_year not in seasons:
        seasons.append(target_year)

    print(f"Building leakage-safe data using games before {prediction_date}...")
    data, slate = build_dataset(seasons, prediction_date)
    data.to_csv(OUTPUT / "mlb_feature_data.csv", index=False)

    print(f"\nUsable games: {len(data):,}")
    print(f"Features: {len(FEATURES)}")

    results, predictions, final_model = run_experiment(data)
    results.to_csv(OUTPUT / "model_results.csv", index=False)
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)

    daily_predictions = predict_winners(final_model, slate)
    daily_path = OUTPUT / f"predictions_{prediction_date}.csv"
    daily_predictions.to_csv(daily_path, index=False)

    print("\nOut-of-sample results")
    print(results.round(4).to_string(index=False))
    print("\nLower log loss and Brier score are better.")

    if daily_predictions.empty:
        print(f"\nNo MLB games were found for {prediction_date}.")
    else:
        display = daily_predictions[
            ["away_team", "home_team", "predicted_winner", "win_probability"]
        ].copy()
        display["win_probability"] = (100 * display["win_probability"]).round(1).astype(str) + "%"
        print(f"\nPredicted winners for {prediction_date}")
        print(display.to_string(index=False))

    print(f"\nPredictions saved to {daily_path}")
    print("All files were saved in the mlb_results folder.")


if __name__ == "__main__":
    main()

