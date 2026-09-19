###
#"""
#Simple, single-file MLB betting starter.

#What it does:
#  1. Pulls finished games from the MLB Stats API for a season (free, no key).
#  2. Builds one feature per team: rolling win% going into that game.
#  3. Trains a small neural net to predict home-team win probability.
#  4. Predicts today's games and lets you compare to sportsbook odds.
#  5. Logs bets you decide to make to a CSV.

# Run it directly:
#    python simple_mlb.py

## This is a starting point, not the full model — no pitchers, no park
#factors, no Elo. Just team win% -> neural net -> probability.
#"""

from __future__ import annotations

import csv
import datetime as dt
import os
from collections import defaultdict

import csv
import datetime as dt
import os
from collections import defaultdict

import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

STATS_API = "https://statsapi.mlb.com/api/v1"
try:
    BETS_CSV = os.path.join(os.path.dirname(__file__), "simple_bets.csv")
except NameError:
    BETS_CSV = "simple_bets.csv"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_finished_games(season: int) -> list[dict]:
    """All completed regular-season games for a season, in date order."""
    resp = requests.get(
        f"{STATS_API}/schedule",
        params={
            "sportId": 1,
            "season": season,
            "gameType": "R",
            "fields": (
                "dates,date,games,gamePk,status,codedGameState,"
                "teams,home,away,team,id,name,score"
            ),
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    games = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            if g["status"].get("codedGameState") != "F":
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            if home.get("score") is None or away.get("score") is None:
                continue
            games.append(
                {
                    "date": day["date"],
                    "home_id": home["team"]["id"],
                    "home_name": home["team"]["name"],
                    "away_id": away["team"]["id"],
                    "away_name": away["team"]["name"],
                    "home_won": home["score"] > away["score"],
                }
            )
    return games


def get_todays_games() -> list[dict]:
    today = dt.date.today().isoformat()
    resp = requests.get(
        f"{STATS_API}/schedule",
        params={"sportId": 1, "date": today},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    games = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            games.append(
                {
                    "game_pk": g["gamePk"],
                    "home_id": home["team"]["id"],
                    "home_name": home["team"]["name"],
                    "away_id": away["team"]["id"],
                    "away_name": away["team"]["name"],
                }
            )
    return games


# --------------------------------------------------------------------------- #
# Features: rolling win% per team, computed walk-forward so a game never sees
# its own result or future results.
# --------------------------------------------------------------------------- #
def build_training_data(games: list[dict]) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    wins = defaultdict(int)
    played = defaultdict(int)

    def win_pct(team_id: int) -> float:
        return wins[team_id] / played[team_id] if played[team_id] else 0.5

    X, y = [], []
    for g in games:
        home_wp = win_pct(g["home_id"])
        away_wp = win_pct(g["away_id"])
        X.append([home_wp, away_wp, home_wp - away_wp])
        y.append(1 if g["home_won"] else 0)

        played[g["home_id"]] += 1
        played[g["away_id"]] += 1
        if g["home_won"]:
            wins[g["home_id"]] += 1
        else:
            wins[g["away_id"]] += 1

    current_win_pct = {team_id: win_pct(team_id) for team_id in played}
    return np.array(X), np.array(y), current_win_pct


def train_model(X: np.ndarray, y: np.ndarray) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        alpha=1.0,
        max_iter=2000,
        early_stopping=True,
        random_state=0,
    )
    model.fit(X, y)
    return model


# --------------------------------------------------------------------------- #
# Odds math
# --------------------------------------------------------------------------- #
def american_to_prob(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds: float) -> float:
    if odds < 0:
        return 1 + 100 / -odds
    return 1 + odds / 100


def kelly_stake(model_prob: float, odds: float, fraction: float = 0.25, bankroll: float = 1000.0) -> float:
    """Fractional Kelly stake in dollars. 0 if there's no edge."""
    dec = american_to_decimal(odds)
    b = dec - 1
    edge = model_prob * dec - 1
    if edge <= 0:
        return 0.0
    f = edge / b
    return round(bankroll * f * fraction, 2)


# --------------------------------------------------------------------------- #
# Bet log
# --------------------------------------------------------------------------- #
def log_bet(date: str, team: str, opponent: str, odds: float, stake: float, model_prob: float) -> None:
    is_new = not os.path.exists(BETS_CSV)
    with open(BETS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["date", "team", "opponent", "odds", "stake", "model_prob", "result", "profit"])
        writer.writerow([date, team, opponent, odds, stake, round(model_prob, 4), "pending", ""])
    print(f"Logged: {team} vs {opponent} @ {odds:+.0f}, stake ${stake:.2f} -> {BETS_CSV}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    train_season = dt.date.today().year - 1
    print(f"Pulling {train_season} season results...")
    games = get_finished_games(train_season)
    print(f"{len(games)} finished games.")

    X, y, win_pct = build_training_data(games)
    model = train_model(X, y)
    print("Model trained.\n")

    todays = get_todays_games()
    if not todays:
        print("No games scheduled today.")
        return

    print(f"Today's games ({dt.date.today().isoformat()}):\n")
    predictions = []
    for g in todays:
        home_wp = win_pct.get(g["home_id"], 0.5)
        away_wp = win_pct.get(g["away_id"], 0.5)
        features = np.array([[home_wp, away_wp, home_wp - away_wp]])
        home_prob = model.predict_proba(features)[0][1]
        predictions.append({**g, "home_prob": home_prob})
        print(
            f"  {g['away_name']:>22} @ {g['home_name']:<22}  "
            f"home win prob: {home_prob:.1%}"
        )

    print(
        "\nTo check for value, enter each team's American moneyline odds "
        "and compare to the probabilities above."
    )
    print("Example, in a Python shell:")
    print("    from simple_mlb import american_to_prob, kelly_stake, log_bet")
    print("    american_to_prob(-150)   # sportsbook implied probability")
    print("    kelly_stake(0.62, -150)  # suggested stake if you have an edge")
    print('    log_bet("2026-04-01", "Yankees", "Red Sox", -150, 25.0, 0.62)')


if __name__ == "__main__":
    main()
