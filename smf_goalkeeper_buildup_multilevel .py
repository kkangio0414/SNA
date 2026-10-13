from __future__ import annotations

import json
import hashlib
import importlib.metadata
import gzip
import math
import os
import pickle
import re
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

TARGET_COMPETITION_NAME = "UEFA Euro"
TARGET_SEASON_NAME = "2024"
TARGET_GENDER = "male"
TARGET_COMPETITION_ID = 55
TARGET_SEASON_ID = 282
ANALYSIS_VERSION = "2026-08-12-incomplete-receipt-control-fix"

DEFENSIVE_THIRD_X = 40.0
HALFWAY_X = 60.0
FINAL_THIRD_X = 80.0
FORWARD_PASS_MIN_X_GAIN = 10.0
LONG_PASS_THRESHOLD = 30.0
EARLY_PASS_LIMIT = 3
INITIAL_BUILDUP_ACTIONS = 5
POST_DISTRIBUTION_WINDOW_SECONDS = 10.0
MIN_SEQUENCE_TEAM_EVENTS = 2
EDGE_BOOTSTRAP_ITERATIONS = 200
EDGE_STABILITY_THRESHOLD = 0.95
MAP_GRADIENT_NORM_THRESHOLD = 1e-3

MAX_RETRIES = 4
RETRY_BASE_SECONDS = 2.0
FORCE_REFRESH_CACHE = False
EXPORT_JSON_CACHE = True

SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    Path("/content/smf_goalkeeper_buildup_focused")
    if Path("/content").exists()
    else (
        SOURCE_DIRECTORY
        if SOURCE_DIRECTORY.name == "smf_goalkeeper_buildup_focused"
        else SOURCE_DIRECTORY / "smf_goalkeeper_buildup_focused"
    )
)
OUTPUT_ROOT = Path(os.environ.get("SMF_GK_BUILDUP_OUTPUT_DIR", DEFAULT_ROOT))
LEGACY_OUTPUT_ROOT = SOURCE_DIRECTORY / "smf_goalkeeper_buildup_multilevel"
CACHE_DIR = (
    LEGACY_OUTPUT_ROOT / "cache"
    if (LEGACY_OUTPUT_ROOT / "cache").exists()
    else OUTPUT_ROOT / "cache"
)
EVENT_CACHE_DIR = CACHE_DIR / "events"
LINEUP_CACHE_DIR = CACHE_DIR / "lineups"
TABLE_DIR = OUTPUT_ROOT / "tables"
MODEL_DIR = OUTPUT_ROOT / "models"
FIGURE_DIR = OUTPUT_ROOT / "figures"
SUPPLEMENT_FIGURE_DIR = FIGURE_DIR / "supplementary"
JSON_CACHE_DIR = OUTPUT_ROOT / "cache_json"

for directory in [
    OUTPUT_ROOT,
    CACHE_DIR,
    EVENT_CACHE_DIR,
    LINEUP_CACHE_DIR,
    TABLE_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    SUPPLEMENT_FIGURE_DIR,
    JSON_CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


OUTCOME_COLUMNS = [
    "defensive_third_exit",
    "entry_x60",
    "controlled_final_third_entry",
    "shot_created",
    "lost_within_10s",
    "opponent_shot_after_loss_10s",
]

CORE_MODEL_OUTCOMES = [
    "controlled_final_third_entry",
    "lost_within_10s",
]

MODEL_NUMERIC_PREDICTORS = [
    "pressure_anchor_share",
    "stage_strength",
    "sequence_start_minute",
    "global_efficiency",
    "long_distribution_share",
    "cb_fb_distribution_share",
]

# Pressure is deliberately kept on its natural scale.  A one-unit change in
# this predictor is a 0.5 increase in pressure_anchor_share (no exposed anchor
# action -> one of two anchor actions exposed).  Unlike z scores, this unit is
# identical in the full, LOO-eligible, and restricted sensitivity samples.
PRESSURE_MODEL_TERM = "pressure_per_half_unit"

LAYER_NETWORK_PREDICTORS = [
    "global_efficiency",
    "long_distribution_share",
    "cb_fb_distribution_share",
]

MODEL_CATEGORICAL_PREDICTORS = [
    "half",
    "score_state",
    "sequence_origin",
]


# ---------------------------------------------------------------------
# 2. Data loading and generic helpers
# ---------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or float(denominator) == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def retry_call(function, *args, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            last_error = error
            if attempt == MAX_RETRIES:
                break
            sleep_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"Request failed ({attempt}/{MAX_RETRIES}): {error}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{function.__name__} failed after retries") from last_error


def fit_vb_deterministic(model):
    """Fit a variational Bayes model from fixed, non-random starting values."""
    parameter_count = model.k_fep + model.k_vcp + model.k_vc
    initial_mean = np.zeros(parameter_count)
    initial_sd = np.full(parameter_count, np.exp(-0.5))
    return model.fit_vb(mean=initial_mean, sd=initial_sd)


def resolve_target_competition() -> tuple[int, int, pd.DataFrame]:
    competitions = retry_call(sb.competitions)
    target = competitions.loc[
        competitions["competition_name"].eq(TARGET_COMPETITION_NAME)
        & competitions["season_name"].astype(str).eq(TARGET_SEASON_NAME)
        & competitions["competition_gender"].eq(TARGET_GENDER)
    ].copy()

    if target.empty:
        raise ValueError(
            f"{TARGET_COMPETITION_NAME} {TARGET_SEASON_NAME} was not found in StatsBomb Open Data."
        )

    row = target.iloc[0]
    return int(row["competition_id"]), int(row["season_id"]), competitions


def load_match_events(match_id: int) -> pd.DataFrame:
    cache_file = EVENT_CACHE_DIR / f"{match_id}.pkl"
    json_cache_file = JSON_CACHE_DIR / "events" / f"{match_id}.json.gz"
    if not FORCE_REFRESH_CACHE:
        if json_cache_file.exists():
            try:
                return pd.read_json(
                    json_cache_file,
                    orient="table",
                    compression="gzip",
                )
            except Exception as error:
                print(f"JSON event cache unreadable for {match_id}; trying pickle: {error}")
        if cache_file.exists():
            return pd.read_pickle(cache_file)

    events = retry_call(sb.events, match_id=int(match_id))
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError(f"Match {match_id}: empty events")

    events.to_pickle(cache_file)
    return events


def load_match_lineups(match_id: int) -> dict[str, pd.DataFrame]:
    cache_file = LINEUP_CACHE_DIR / f"{match_id}.pkl"
    json_cache_file = JSON_CACHE_DIR / "lineups" / f"{match_id}.json.gz"
    if not FORCE_REFRESH_CACHE:
        if json_cache_file.exists():
            try:
                with gzip.open(json_cache_file, "rt", encoding="utf-8") as file:
                    payload = json.load(file)
                return {
                    team_name: pd.DataFrame(records)
                    for team_name, records in payload.items()
                }
            except Exception as error:
                print(f"JSON lineup cache unreadable for {match_id}; trying pickle: {error}")
        if cache_file.exists():
            with cache_file.open("rb") as file:
                return pickle.load(file)

    lineups = retry_call(sb.lineups, match_id=int(match_id))
    if not isinstance(lineups, dict) or not lineups:
        raise ValueError(f"Match {match_id}: empty lineups")

    with cache_file.open("wb") as file:
        pickle.dump(lineups, file)
    return lineups


def export_json_reproducibility_cache() -> None:
    """Export pickle caches as gzip JSON for pandas-version-independent reuse."""
    if not EXPORT_JSON_CACHE:
        return

    event_output_dir = JSON_CACHE_DIR / "events"
    lineup_output_dir = JSON_CACHE_DIR / "lineups"
    event_output_dir.mkdir(parents=True, exist_ok=True)
    lineup_output_dir.mkdir(parents=True, exist_ok=True)

    for cache_file in EVENT_CACHE_DIR.glob("*.pkl"):
        output_file = event_output_dir / f"{cache_file.stem}.json.gz"
        if output_file.exists() and output_file.stat().st_mtime >= cache_file.stat().st_mtime:
            continue
        events = pd.read_pickle(cache_file)
        events.to_json(
            output_file,
            orient="table",
            force_ascii=False,
            compression="gzip",
            index=False,
        )

    for cache_file in LINEUP_CACHE_DIR.glob("*.pkl"):
        output_file = lineup_output_dir / f"{cache_file.stem}.json.gz"
        if output_file.exists() and output_file.stat().st_mtime >= cache_file.stat().st_mtime:
            continue
        with cache_file.open("rb") as file:
            lineups = pickle.load(file)
        payload = {
            team_name: dataframe.to_dict(orient="records")
            for team_name, dataframe in lineups.items()
        }
        with gzip.open(output_file, "wt", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, default=str)


def clock_to_minutes(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float) and np.isnan(value):
        return float("nan")

    parts = str(value).strip().split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
        if len(parts) == 3:
            return float(parts[0]) * 60.0 + float(parts[1]) + float(parts[2]) / 60.0
    except ValueError:
        return float("nan")
    return float("nan")


def prepare_event_time(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    minutes = pd.to_numeric(result.get("minute", 0), errors="coerce").fillna(0)
    seconds = pd.to_numeric(result.get("second", 0), errors="coerce").fillna(0)
    result["_event_minute"] = minutes + seconds / 60.0
    result["_event_second"] = result["_event_minute"] * 60.0
    result["_event_index"] = pd.to_numeric(
        result.get("index", np.arange(len(result))),
        errors="coerce",
    ).fillna(pd.Series(np.arange(len(result)), index=result.index)).astype(int)
    return result


def get_match_end_minute(events: pd.DataFrame) -> float:
    if "_event_minute" not in events.columns:
        events = prepare_event_time(events)
    regulation = events.loc[events["period"].isin([1, 2, 3, 4])]
    match_end = pd.to_numeric(regulation["_event_minute"], errors="coerce").max()
    return max(float(match_end), 90.0) if not pd.isna(match_end) else 90.0


def location_x(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 1:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def location_y(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        try:
            return float(value[1])
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def pass_end_x(row: pd.Series | Any) -> float:
    end_x = location_x(getattr(row, "pass_end_location", None))
    if not pd.isna(end_x):
        return end_x
    return location_x(getattr(row, "location", None))


def controlled_event_x(row: pd.Series | Any) -> float:
    """Return an x-coordinate only when the team has controlled the ball."""
    event_type = str(getattr(row, "type", ""))
    if event_type == "Pass":
        if pd.isna(getattr(row, "pass_outcome", None)):
            return location_x(getattr(row, "pass_end_location", None))
        return float("nan")
    if event_type == "Carry":
        carry_end = location_x(getattr(row, "carry_end_location", None))
        return carry_end if not pd.isna(carry_end) else location_x(getattr(row, "location", None))
    if event_type in {"Ball Receipt*", "Ball Receipt"}:
        receipt_outcome = getattr(row, "ball_receipt_outcome", None)
        if pd.isna(receipt_outcome):
            return location_x(getattr(row, "location", None))
        return float("nan")
    if event_type == "Dribble":
        return location_x(getattr(row, "location", None))
    return float("nan")


def first_controlled_entry_second(
    team_events: pd.DataFrame,
    threshold_x: float,
) -> float:
    """Return the first time controlled team possession reaches a pitch threshold."""
    values = team_events.apply(controlled_event_x, axis=1)
    reached = team_events.loc[values.ge(threshold_x)]
    if reached.empty:
        return float("nan")
    return float(reached["_event_second"].min())


def first_controlled_entry_index(
    team_events: pd.DataFrame,
    threshold_x: float,
) -> float:
    """Return the first event index where controlled possession reaches a threshold."""
    values = team_events.apply(controlled_event_x, axis=1)
    reached = team_events.loc[values.ge(threshold_x)]
    if reached.empty:
        return float("nan")
    return float(reached["_event_index"].min())


def display_player_name(row: pd.Series | Any) -> str:
    nickname = getattr(row, "player_nickname", None)
    if nickname is not None and not pd.isna(nickname) and str(nickname).strip():
        return str(nickname).strip()
    name = getattr(row, "player_name", None)
    return str(name).strip() if name is not None and not pd.isna(name) else "Unknown goalkeeper"


def get_match_context(match_row: Any, team_name: str) -> dict[str, Any]:
    is_home = str(match_row.home_team) == team_name
    opponent = str(match_row.away_team) if is_home else str(match_row.home_team)
    goals_for = int(match_row.home_score) if is_home else int(match_row.away_score)
    goals_against = int(match_row.away_score) if is_home else int(match_row.home_score)
    result = "win" if goals_for > goals_against else "loss" if goals_for < goals_against else "draw"
    return {
        "match_id": int(match_row.match_id),
        "match_date": str(match_row.match_date),
        "competition_stage": str(match_row.competition_stage),
        "team": team_name,
        "opponent": opponent,
        "venue": "home" if is_home else "away",
        "goals_for": goals_for,
        "goals_against": goals_against,
        "match_result": result,
    }


# ---------------------------------------------------------------------
# 3. Goalkeeper stints and match-level context
# ---------------------------------------------------------------------

def extract_goalkeeper_stints(
    lineup_df: pd.DataFrame | None,
    team_name: str,
    match_end_minute: float,
) -> list[dict[str, Any]]:
    if lineup_df is None or lineup_df.empty or "positions" not in lineup_df.columns:
        return []

    stints: list[dict[str, Any]] = []
    for row in lineup_df.itertuples(index=False):
        player_id = getattr(row, "player_id", None)
        if player_id is None or pd.isna(player_id):
            continue

        for position in getattr(row, "positions", []) or []:
            if not isinstance(position, dict):
                continue
            if position.get("position") != "Goalkeeper" and position.get("position_id") != 1:
                continue

            start = clock_to_minutes(position.get("from"))
            end = clock_to_minutes(position.get("to"))
            if pd.isna(start):
                start = 0.0
            if pd.isna(end):
                end = match_end_minute
            if end <= start:
                continue

            stints.append(
                {
                    "team": team_name,
                    "goalkeeper_id": int(player_id),
                    "goalkeeper": display_player_name(row),
                    "goalkeeper_full_name": str(getattr(row, "player_name", display_player_name(row))),
                    "start_minute": float(start),
                    "end_minute": float(end),
                    "stint_minutes": float(end - start),
                    "started_match": bool(start == 0.0),
                }
            )

    return sorted(stints, key=lambda item: (item["start_minute"], item["goalkeeper_id"]))


def fallback_goalkeeper_stints_from_events(
    events: pd.DataFrame,
    team_name: str,
    match_end_minute: float,
) -> list[dict[str, Any]]:
    goalkeepers = events.loc[
        events.get("team").eq(team_name)
        & events.get("position").eq("Goalkeeper")
        & events.get("player_id").notna()
    ][["player_id", "player"]].drop_duplicates("player_id")

    return [
        {
            "team": team_name,
            "goalkeeper_id": int(row.player_id),
            "goalkeeper": str(row.player),
            "goalkeeper_full_name": str(row.player),
            "start_minute": 0.0,
            "end_minute": float(match_end_minute),
            "stint_minutes": float(match_end_minute),
            "started_match": True,
        }
        for row in goalkeepers.itertuples(index=False)
    ]


STAGE_ORDER = {
    "Group Stage": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Quarter Final": 3,
    "Semi-finals": 4,
    "Semi Final": 4,
    "Final": 5,
}


def build_team_strength_lookup(matches: pd.DataFrame) -> dict[str, int]:
    strength: dict[str, int] = {}
    for row in matches.itertuples(index=False):
        stage_score = STAGE_ORDER.get(str(row.competition_stage), 0)
        for team_name in [str(row.home_team), str(row.away_team)]:
            strength[team_name] = max(strength.get(team_name, 0), stage_score)
    return strength


def player_position_lookup(events: pd.DataFrame) -> dict[int, str]:
    positions = events.loc[
        events["player_id"].notna() & events["position"].notna(),
        ["player_id", "position"],
    ].drop_duplicates("player_id")
    return {int(row.player_id): str(row.position) for row in positions.itertuples(index=False)}


# ---------------------------------------------------------------------
# 4. Sequence definitions, independent variables and outcomes
# ---------------------------------------------------------------------

def score_state_at_second(events: pd.DataFrame, team_name: str, second: float) -> str:
    shot_goals = events.loc[
        events["type"].eq("Shot")
        & events["shot_outcome"].eq("Goal")
        & events["_event_second"].lt(second)
    ]
    own_goals_against = events.loc[
        events["type"].eq("Own Goal Against")
        & events["_event_second"].lt(second)
    ]
    goals_for = int(shot_goals["team"].eq(team_name).sum()) + int(
        (~own_goals_against["team"].eq(team_name)).sum()
    )
    goals_against = int((~shot_goals["team"].eq(team_name)).sum()) + int(
        own_goals_against["team"].eq(team_name).sum()
    )
    if goals_for > goals_against:
        return "winning"
    if goals_for < goals_against:
        return "losing"
    return "drawing"


def pressure_level(pressure_share: float) -> str:
    if pd.isna(pressure_share) or pressure_share <= 0:
        return "none"
    if pressure_share >= 1.0:
        return "high"
    return "low"


def pressure_level_display_label(level: str) -> str:
    labels = {
        "none": "No observed anchor pressure",
        "low": "One of two anchor actions pressured",
        "high": "Pressure on all observed anchor actions",
    }
    return labels.get(str(level).lower(), str(level).replace("_", " ").title())


def time_layer_label(minute: float, period: int) -> str:
    if period == 1:
        return "00-15" if minute <= 15 else "16-30" if minute <= 30 else "31-45"
    if period == 2:
        return "46-60" if minute <= 60 else "61-75" if minute <= 75 else "76-90"
    if period == 3:
        return "91-105"
    return "106-120"


def infer_sequence_origin(
    events: pd.DataFrame,
    sequence: pd.DataFrame,
    team_name: str,
    sequence_start_second: float,
) -> str:
    play_patterns = set(sequence.get("play_pattern", pd.Series(dtype=str)).dropna())
    if "From Goal Kick" in play_patterns:
        return "goal_kick"

    previous_window = events.loc[
        events["_event_second"].ge(sequence_start_second - POST_DISTRIBUTION_WINDOW_SECONDS)
        & events["_event_second"].lt(sequence_start_second)
        & ~events["team"].eq(team_name)
    ]
    if previous_window["type"].eq("Shot").any():
        return "distribution_after_opponent_shot"

    if "Goal Keeper" in set(sequence.get("type", pd.Series(dtype=str)).dropna()):
        return "other_goalkeeper_action"
    if "From Throw In" in play_patterns:
        return "throw_in"
    if "From Free Kick" in play_patterns:
        return "free_kick"
    return "open_play_recovery"


def is_back_line_position(position: str | float | None) -> bool:
    if position is None or pd.isna(position):
        return False
    text = str(position).lower()
    tokens = ["center back", "left back", "right back", "wing back", "full back"]
    return any(token in text for token in tokens)


def build_sequence_network_metrics(
    completed_passes: pd.DataFrame,
    goalkeeper_id: int,
) -> dict[str, Any]:
    graph = nx.DiGraph()
    if not completed_passes.empty:
        edge_table = (
            completed_passes.groupby(["player_id", "pass_recipient_id"], as_index=False)
            .agg(weight=("id", "size"))
            .rename(columns={"player_id": "source_id", "pass_recipient_id": "target_id"})
        )
        for edge in edge_table.itertuples(index=False):
            graph.add_edge(
                int(edge.source_id),
                int(edge.target_id),
                weight=int(edge.weight),
                distance=1.0 / float(edge.weight),
            )
    graph.add_node(int(goalkeeper_id))

    if graph.number_of_nodes() > 1 and graph.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
        density = float(nx.density(graph))
        global_efficiency = float(nx.global_efficiency(graph.to_undirected()))
    else:
        betweenness = {int(goalkeeper_id): 0.0}
        density = 0.0
        global_efficiency = 0.0

    gk_id = int(goalkeeper_id)
    return {
        "weighted_out_strength": int(graph.out_degree(gk_id, weight="weight")),
        "weighted_in_strength": int(graph.in_degree(gk_id, weight="weight")),
        "out_degree": int(graph.out_degree(gk_id)),
        "in_degree": int(graph.in_degree(gk_id)),
        "betweenness": float(betweenness.get(gk_id, 0.0)),
        "network_density": density,
        "global_efficiency": global_efficiency,
        "network_nodes": int(graph.number_of_nodes()),
        "network_edges": int(graph.number_of_edges()),
    }


def calculate_distribution_metrics(
    gk_passes: pd.DataFrame,
    gk_completed: pd.DataFrame,
    position_lookup: dict[int, str],
) -> dict[str, Any]:
    attempts = int(len(gk_passes))
    completed = int(len(gk_completed))

    pass_lengths = pd.to_numeric(gk_passes["pass_length"], errors="coerce")
    long_attempts = int(pass_lengths.ge(LONG_PASS_THRESHOLD).sum())
    forward_attempts = 0
    if not gk_passes.empty:
        start_x = gk_passes["location"].map(location_x)
        end_x = gk_passes.apply(pass_end_x, axis=1)
        forward_attempts = int((end_x - start_x).ge(FORWARD_PASS_MIN_X_GAIN).sum())

    receiver_counts = gk_completed.groupby("pass_recipient_id").size().sort_values(ascending=False)
    receiver_diversity = int(receiver_counts.size)
    if completed > 0:
        shares = receiver_counts / completed
        receiver_hhi = float(np.square(shares).sum())
        receiver_entropy = float(-np.sum(shares * np.log(shares)) / np.log(receiver_diversity)) if receiver_diversity > 1 else 0.0
        top_receiver_id = int(receiver_counts.index[0])
        top_receiver_passes = int(receiver_counts.iloc[0])
        top_receiver_share = safe_divide(top_receiver_passes, completed)
        back_line_passes = int(
            sum(is_back_line_position(position_lookup.get(int(receiver_id))) * int(count) for receiver_id, count in receiver_counts.items())
        )
    else:
        receiver_hhi = float("nan")
        receiver_entropy = float("nan")
        top_receiver_id = float("nan")
        top_receiver_passes = 0
        top_receiver_share = float("nan")
        back_line_passes = 0

    return {
        "gk_pass_attempts": attempts,
        "gk_completed_passes": completed,
        "gk_completion_pct": safe_divide(completed, attempts),
        "long_distribution_attempts": long_attempts,
        "long_distribution_share": safe_divide(long_attempts, attempts),
        "forward_pass_attempts": forward_attempts,
        "forward_pass_share": safe_divide(forward_attempts, attempts),
        "receiver_diversity": receiver_diversity,
        "receiver_entropy": receiver_entropy,
        "receiver_hhi": receiver_hhi,
        "top_receiver_id": top_receiver_id,
        "top_receiver_passes": top_receiver_passes,
        "top_receiver_share": top_receiver_share,
        "cb_fb_distribution_passes": back_line_passes,
        "cb_fb_distribution_share": safe_divide(back_line_passes, completed),
    }


def extract_goalkeeper_buildup_sequences(
    events: pd.DataFrame,
    match_row: Any,
    team_name: str,
    goalkeeper_stint: dict[str, Any],
    team_strength: dict[str, int],
) -> pd.DataFrame:
    goalkeeper_id = int(goalkeeper_stint["goalkeeper_id"])
    start_minute = float(goalkeeper_stint["start_minute"])
    end_minute = float(goalkeeper_stint["end_minute"])
    context = get_match_context(match_row, team_name)
    positions = player_position_lookup(events)

    match_events = events.loc[events["period"].isin([1, 2, 3, 4])].copy()
    stint_events = match_events.loc[
        match_events["_event_minute"].ge(start_minute)
        & match_events["_event_minute"].lt(end_minute)
    ].copy()
    team_possessions = stint_events.loc[stint_events["possession_team"].eq(team_name)].copy()

    rows: list[dict[str, Any]] = []
    sort_columns = [column for column in ["period", "index", "_event_second"] if column in events.columns]

    for possession_id, sequence in team_possessions.groupby("possession", dropna=True):
        sequence = sequence.sort_values(sort_columns).copy()
        team_events = sequence.loc[sequence["team"].eq(team_name)].copy()
        if len(team_events) < MIN_SEQUENCE_TEAM_EVENTS:
            continue

        team_passes = team_events.loc[team_events["type"].eq("Pass")].copy()
        early_passes = team_passes.head(EARLY_PASS_LIMIT)
        first_team_event = team_events.iloc[0]
        first_x_values = team_events["location"].map(location_x).dropna()
        first_x = float(first_x_values.iloc[0]) if not first_x_values.empty else float("nan")

        gk_in_sequence = bool(team_events["player_id"].eq(goalkeeper_id).any())
        gk_starts_sequence = bool(first_team_event.get("player_id") == goalkeeper_id)
        gk_early_pass = bool(early_passes["player_id"].eq(goalkeeper_id).any())
        defensive_start = bool(not pd.isna(first_x) and first_x <= DEFENSIVE_THIRD_X)
        from_goal_kick = bool("From Goal Kick" in set(team_events["play_pattern"].dropna()))

        if not gk_in_sequence or not (
            gk_starts_sequence
            or gk_early_pass
            or (from_goal_kick and gk_in_sequence)
            or defensive_start
        ):
            continue

        sequence_start_second = float(sequence["_event_second"].min())
        sequence_end_second = float(sequence["_event_second"].max())

        gk_events = team_events.loc[team_events["player_id"].eq(goalkeeper_id)].copy()
        gk_passes = team_passes.loc[team_passes["player_id"].eq(goalkeeper_id)].copy()
        if gk_passes.empty:
            continue
        first_gk_distribution_second = float(gk_passes["_event_second"].min())
        first_gk_distribution_index = int(gk_passes["_event_index"].min())

        post_distribution_events = team_events.loc[
            team_events["_event_index"].ge(first_gk_distribution_index)
        ].sort_values(sort_columns).copy()
        pre_distribution_events = team_events.loc[
            team_events["_event_index"].lt(first_gk_distribution_index)
        ].copy()
        initial_build_events = post_distribution_events.head(INITIAL_BUILDUP_ACTIONS).copy()
        initial_build_end_index = (
            int(initial_build_events["_event_index"].max())
            if not initial_build_events.empty
            else first_gk_distribution_index
        )
        pressure_anchor_events = post_distribution_events.head(2).copy()
        pressure_anchor_count = int(len(pressure_anchor_events))
        pressure_anchor_events_under_pressure = int(
            pressure_anchor_events["under_pressure"].fillna(False).astype(bool).sum()
        )
        pressure_anchor_share = safe_divide(
            pressure_anchor_events_under_pressure,
            pressure_anchor_count,
        )
        any_pressure_anchor = int(pressure_anchor_events_under_pressure > 0)

        defensive_exit_eligible = (
            defensive_start
            and pd.isna(
                first_controlled_entry_index(
                    pre_distribution_events,
                    DEFENSIVE_THIRD_X + 1e-6,
                )
            )
        )
        defensive_exit_index = (
            first_controlled_entry_index(
                post_distribution_events,
                DEFENSIVE_THIRD_X + 1e-6,
            )
            if defensive_exit_eligible
            else float("nan")
        )
        entry_x60_index = first_controlled_entry_index(
            post_distribution_events,
            HALFWAY_X,
        )
        controlled_final_third_index = first_controlled_entry_index(
            post_distribution_events,
            FINAL_THIRD_X,
        )
        shots = post_distribution_events.loc[
            post_distribution_events["type"].eq("Shot")
        ].copy()
        first_shot_index = (
            float(shots["_event_index"].min()) if not shots.empty else float("nan")
        )

        after_distribution = match_events.loc[
            match_events["_event_index"].gt(first_gk_distribution_index)
            & match_events["_event_second"].le(first_gk_distribution_second + POST_DISTRIBUTION_WINDOW_SECONDS)
        ].copy()
        loss_events = after_distribution.loc[
            after_distribution["possession_team"].notna()
            & ~after_distribution["possession_team"].eq(team_name)
        ]
        lost_within_10s = bool(not loss_events.empty)
        first_loss_index = (
            float(loss_events["_event_index"].min())
            if lost_within_10s
            else float("nan")
        )
        opponent_shot_after_loss = False
        if lost_within_10s:
            opponent_window = match_events.loc[
                match_events["_event_index"].ge(first_loss_index)
                & match_events["_event_second"].le(
                    float(loss_events["_event_second"].min())
                    + POST_DISTRIBUTION_WINDOW_SECONDS
                )
                & ~match_events["team"].eq(team_name)
            ]
            opponent_shot_after_loss = bool(opponent_window["type"].eq("Shot").any())

        outcome_indices = [
            value
            for value in [
                defensive_exit_index,
                controlled_final_third_index,
                first_loss_index,
                first_shot_index,
            ]
            if not pd.isna(value) and value >= first_gk_distribution_index
        ]
        outcome_censor_index = min(outcome_indices) if outcome_indices else float("inf")
        network_cutoff_index = min(initial_build_end_index, outcome_censor_index)
        network_events = post_distribution_events.loc[
            post_distribution_events["_event_index"].le(network_cutoff_index)
        ].copy()
        pressure_events = int(
            network_events["under_pressure"].fillna(False).astype(bool).sum()
        )
        pressure_share = safe_divide(pressure_events, len(network_events))
        network_passes = network_events.loc[network_events["type"].eq("Pass")].copy()
        network_completed_passes = network_passes.loc[
            network_passes["pass_outcome"].isna()
            & network_passes["player_id"].notna()
            & network_passes["pass_recipient_id"].notna()
        ].copy()
        if not network_completed_passes.empty:
            network_completed_passes["player_id"] = network_completed_passes["player_id"].astype(int)
            network_completed_passes["pass_recipient_id"] = network_completed_passes["pass_recipient_id"].astype(int)

        gk_network_passes = network_passes.loc[
            network_passes["player_id"].eq(goalkeeper_id)
        ].copy()
        gk_completed = gk_network_passes.loc[
            gk_network_passes["pass_outcome"].isna()
            & gk_network_passes["pass_recipient_id"].notna()
        ].copy()
        if not gk_completed.empty:
            gk_completed["pass_recipient_id"] = gk_completed["pass_recipient_id"].astype(int)

        controlled_x = post_distribution_events.apply(
            controlled_event_x,
            axis=1,
        ).dropna()
        controlled_max_x = float(controlled_x.max()) if not controlled_x.empty else float("nan")
        initial_build_end_minute = float(
            initial_build_events["_event_minute"].max()
        ) if not initial_build_events.empty else first_gk_distribution_second / 60.0
        cutoff_events = match_events.loc[
            match_events["_event_index"].eq(network_cutoff_index)
        ]
        network_cutoff_minute = (
            float(cutoff_events["_event_minute"].min())
            if not cutoff_events.empty
            else initial_build_end_minute
        )
        sequence_id = f"{int(context['match_id'])}_{team_name}_{int(possession_id)}_{goalkeeper_id}"

        row = {
            **context,
            "buildup_sequence_id": sequence_id,
            "possession": int(possession_id),
            "goalkeeper_id": goalkeeper_id,
            "goalkeeper": goalkeeper_stint["goalkeeper"],
            "goalkeeper_full_name": goalkeeper_stint["goalkeeper_full_name"],
            "stint_start_minute": start_minute,
            "stint_end_minute": end_minute,
            "sequence_start_minute": sequence_start_second / 60.0,
            "sequence_end_minute": sequence_end_second / 60.0,
            "time_layer": time_layer_label(
                sequence_start_second / 60.0,
                int(first_team_event["period"]),
            ),
            "period": int(first_team_event["period"]),
            "duration_seconds": sequence_end_second - sequence_start_second,
            "first_gk_distribution_minute": first_gk_distribution_second / 60.0,
            "first_gk_distribution_event_index": first_gk_distribution_index,
            "initial_build_end_minute": initial_build_end_minute,
            "initial_build_end_event_index": initial_build_end_index,
            "network_cutoff_minute": network_cutoff_minute,
            "network_cutoff_event_index": network_cutoff_index,
            "half": "first_half" if int(first_team_event["period"]) == 1 else "second_half_or_extra",
            "score_state": score_state_at_second(match_events, team_name, sequence_start_second),
            "opponent_strength": team_strength.get(context["opponent"], 0),
            "stage_strength": STAGE_ORDER.get(context["competition_stage"], 0),
            "sequence_origin": infer_sequence_origin(match_events, team_events, team_name, sequence_start_second),
            "pressure_events": pressure_events,
            "pressure_share": pressure_share,
            "pressure_level": pressure_level(pressure_anchor_share),
            "pressure_anchor_events": pressure_anchor_count,
            "pressure_anchor_under_pressure_events": pressure_anchor_events_under_pressure,
            "pressure_anchor_share": pressure_anchor_share,
            "any_pressure_anchor": any_pressure_anchor,
            "gk_starts_sequence": gk_starts_sequence,
            "gk_early_pass": gk_early_pass,
            "defensive_start": defensive_start,
            "from_goal_kick": from_goal_kick,
            "first_x": first_x,
            "controlled_max_x": controlled_max_x,
            "defensive_third_exit": (
                int(not pd.isna(defensive_exit_index))
                if defensive_exit_eligible
                else pd.NA
            ),
            "entry_x60": int(not pd.isna(entry_x60_index)),
            "controlled_final_third_entry": int(
                not pd.isna(controlled_final_third_index)
            ),
            "final_third_entry": int(not pd.isna(controlled_final_third_index)),
            "shot_created": int(not shots.empty),
            "lost_within_10s": int(lost_within_10s),
            "opponent_shot_after_loss_10s": int(opponent_shot_after_loss),
            "team_events": int(len(team_events)),
            "team_passes": int(len(team_passes)),
            "initial_build_events": int(len(initial_build_events)),
            "network_events": int(len(network_events)),
            "network_passes": int(len(network_passes)),
            "team_completed_passes": int(len(network_completed_passes)),
            "defensive_third_exit_eligible": int(defensive_exit_eligible),
        }
        row.update(build_sequence_network_metrics(network_completed_passes, goalkeeper_id))
        row.update(calculate_distribution_metrics(gk_network_passes, gk_completed, positions))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 5. Dataset construction and summaries
# ---------------------------------------------------------------------

def build_sequence_dataset(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_strength = build_team_strength_lookup(matches)
    sequence_frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []

    for match_row in tqdm(
        matches.sort_values("match_date").itertuples(index=False),
        total=len(matches),
        desc="Build sequence dataset",
    ):
        match_id = int(match_row.match_id)
        try:
            events = prepare_event_time(load_match_events(match_id))
            lineups = load_match_lineups(match_id)
            match_end = get_match_end_minute(events)

            for team_name in [str(match_row.home_team), str(match_row.away_team)]:
                stints = extract_goalkeeper_stints(lineups.get(team_name), team_name, match_end)
                if not stints:
                    stints = fallback_goalkeeper_stints_from_events(events, team_name, match_end)
                if not stints:
                    errors.append({"match_id": match_id, "team": team_name, "error": "No goalkeeper stint"})
                    continue

                for stint in stints:
                    sequences = extract_goalkeeper_buildup_sequences(
                        events=events,
                        match_row=match_row,
                        team_name=team_name,
                        goalkeeper_stint=stint,
                        team_strength=team_strength,
                    )
                    if not sequences.empty:
                        sequence_frames.append(sequences)
        except Exception as error:
            errors.append({"match_id": match_id, "team": None, "error": repr(error)})
            print(f"Match {match_id} failed: {error}")

    dataset = pd.concat(sequence_frames, ignore_index=True) if sequence_frames else pd.DataFrame()
    return dataset, pd.DataFrame(errors)


def summarize_context_layers(sequence_dataset: pd.DataFrame) -> pd.DataFrame:
    if sequence_dataset.empty:
        return pd.DataFrame()

    group_columns = [
        "goalkeeper_id",
        "goalkeeper",
        "team",
        "pressure_level",
        "half",
        "score_state",
        "sequence_origin",
        "competition_stage",
        "opponent_strength",
    ]
    rows: list[dict[str, Any]] = []
    for group_key, group in sequence_dataset.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=False))
        row["sequences"] = int(len(group))
        for outcome in OUTCOME_COLUMNS:
            row[f"{outcome}_rate"] = float(group[outcome].mean())
        for variable in [
            "weighted_out_strength",
            "betweenness",
            "receiver_diversity",
            "receiver_entropy",
            "receiver_hhi",
            "network_density",
            "global_efficiency",
            "forward_pass_share",
            "long_distribution_share",
            "top_receiver_share",
            "cb_fb_distribution_share",
        ]:
            row[f"{variable}_mean"] = float(pd.to_numeric(group[variable], errors="coerce").mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["team", "goalkeeper", "pressure_level", "score_state"])


def aggregate_layer_network_metrics(
    layer_passes: pd.DataFrame,
    goalkeeper_id: int,
) -> dict[str, Any]:
    if layer_passes.empty:
        return {
            "weighted_out_strength": 0.0,
            "betweenness": 0.0,
            "receiver_diversity": float("nan"),
            "receiver_entropy": float("nan"),
            "receiver_hhi": float("nan"),
            "network_density": 0.0,
            "global_efficiency": 0.0,
            "layer_completed_pass_count": 0,
            "layer_node_count": 1,
            "long_distribution_share": float("nan"),
            "cb_fb_distribution_share": float("nan"),
            "top_receiver_share": float("nan"),
            "layer_gk_pass_attempts": 0,
            "layer_gk_completed_passes": 0,
            "layer_entropy_eligible": False,
        }

    completed = layer_passes.loc[
        layer_passes["pass_outcome"].isna()
        & layer_passes["pass_recipient_id"].notna()
    ].copy()
    if not completed.empty:
        completed["player_id"] = completed["player_id"].astype(int)
        completed["pass_recipient_id"] = completed["pass_recipient_id"].astype(int)
    gk_attempts = layer_passes.loc[layer_passes["player_id"].eq(goalkeeper_id)].copy()
    gk_completed = gk_attempts.loc[
        gk_attempts["pass_outcome"].isna()
        & gk_attempts["pass_recipient_id"].notna()
    ].copy()
    if not gk_completed.empty:
        gk_completed["pass_recipient_id"] = gk_completed["pass_recipient_id"].astype(int)
    position_lookup = {
        int(row.pass_recipient_id): str(row.pass_recipient_position)
        for row in layer_passes[["pass_recipient_id", "pass_recipient_position"]]
        .dropna(subset=["pass_recipient_id"])
        .drop_duplicates("pass_recipient_id")
        .itertuples(index=False)
    }
    network_metrics = build_sequence_network_metrics(completed, goalkeeper_id)
    distribution_metrics = calculate_distribution_metrics(
        gk_attempts,
        gk_completed,
        position_lookup,
    )
    entropy_eligible = (
        len(gk_attempts) >= 10
        and int(distribution_metrics["receiver_diversity"]) >= 2
    )
    if not entropy_eligible:
        distribution_metrics["receiver_entropy"] = float("nan")
        distribution_metrics["receiver_hhi"] = float("nan")
    return {
        "weighted_out_strength": float(network_metrics["weighted_out_strength"]),
        "betweenness": float(network_metrics["betweenness"]),
        "receiver_diversity": float(distribution_metrics["receiver_diversity"]),
        "receiver_entropy": float(distribution_metrics["receiver_entropy"]),
        "receiver_hhi": float(distribution_metrics["receiver_hhi"]),
        "network_density": float(network_metrics["network_density"]),
        "global_efficiency": float(network_metrics["global_efficiency"]),
        "layer_completed_pass_count": int(len(completed)),
        "layer_node_count": int(network_metrics["network_nodes"]),
        "long_distribution_share": float(distribution_metrics["long_distribution_share"]),
        "cb_fb_distribution_share": float(distribution_metrics["cb_fb_distribution_share"]),
        "top_receiver_share": float(distribution_metrics["top_receiver_share"]),
        "layer_gk_pass_attempts": int(len(gk_attempts)),
        "layer_gk_completed_passes": int(len(gk_completed)),
        "layer_entropy_eligible": bool(entropy_eligible),
    }


def build_goalkeeper_match_pressure_layers(
    sequence_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sequence_dataset.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_initial_passes = completed_sequence_passes(
        sequence_dataset,
        completed_only=False,
    )
    group_columns = [
        "match_id",
        "team",
        "opponent",
        "goalkeeper_id",
        "goalkeeper",
        "pressure_level",
    ]
    layer_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    for group_key, group in sequence_dataset.groupby(group_columns, dropna=False):
        match_id, team, opponent, goalkeeper_id, goalkeeper, pressure = group_key
        sequence_ids = sorted(
            group["buildup_sequence_id"].astype(str).unique()
        )
        layer_passes = all_initial_passes.loc[
            all_initial_passes["buildup_sequence_id"].isin(sequence_ids)
        ].copy()
        layer_id = f"{int(match_id)}_{int(goalkeeper_id)}_{pressure}"
        metrics = aggregate_layer_network_metrics(layer_passes, int(goalkeeper_id))
        metrics["weighted_out_strength_per_sequence"] = safe_divide(
            metrics["weighted_out_strength"],
            len(group),
        )
        layer_row = {
            "layer_id": layer_id,
            "match_id": int(match_id),
            "team": team,
            "opponent": opponent,
            "goalkeeper_id": int(goalkeeper_id),
            "goalkeeper": goalkeeper,
            "pressure_level": pressure,
            "sequences": int(len(group)),
            "opponent_strength": float(group["opponent_strength"].iloc[0]),
            "stage_strength": float(group["stage_strength"].iloc[0]),
            "sequence_start_minute": float(group["sequence_start_minute"].mean()),
            "defensive_third_exit_rate": float(group["defensive_third_exit"].mean()),
            "controlled_final_third_entry_rate": float(group["controlled_final_third_entry"].mean()),
            "shot_created_rate": float(group["shot_created"].mean()),
            "lost_within_10s_rate": float(group["lost_within_10s"].mean()),
            "opponent_shot_after_loss_10s_rate": float(group["opponent_shot_after_loss_10s"].mean()),
            **metrics,
        }
        layer_rows.append(layer_row)

        for sequence_id in sequence_ids:
            remaining_passes = layer_passes.loc[
                ~layer_passes["buildup_sequence_id"].eq(sequence_id)
            ].copy()
            remaining_sequence_count = int(len(sequence_ids) - 1)
            loo_metrics = aggregate_layer_network_metrics(
                remaining_passes,
                int(goalkeeper_id),
            )
            layer_model_eligible = (
                remaining_sequence_count >= 3
                and loo_metrics["layer_gk_pass_attempts"] >= 3
            )
            leave_one_out_rows.append(
                {
                    "buildup_sequence_id": sequence_id,
                    "layer_id": layer_id,
                    "layer_sequences_excluding": remaining_sequence_count,
                    "layer_model_eligible": bool(layer_model_eligible),
                    **{f"loo_{name}": value for name, value in loo_metrics.items()},
                }
            )

    layer_dataframe = pd.DataFrame(layer_rows).sort_values(
        ["team", "goalkeeper", "match_id", "pressure_level"],
    )
    leave_one_out_dataframe = pd.DataFrame(leave_one_out_rows)
    return layer_dataframe, leave_one_out_dataframe


def build_past_only_network_metrics(sequence_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build temporally preceding, same-match/layer network metrics.

    The focal sequence and every later sequence are excluded.  Three preceding
    sequences and three preceding goalkeeper pass attempts are required, which
    mirrors the primary LOO eligibility rule while enforcing temporal order.
    """
    if sequence_dataset.empty:
        return pd.DataFrame()
    all_passes = completed_sequence_passes(sequence_dataset, completed_only=False)
    rows: list[dict[str, Any]] = []
    group_columns = ["match_id", "team", "goalkeeper_id", "pressure_level"]
    ordered = sequence_dataset.sort_values(
        [*group_columns, "first_gk_distribution_event_index", "buildup_sequence_id"]
    )
    for (_, _, goalkeeper_id, _), group in ordered.groupby(group_columns, dropna=False):
        prior_ids: list[str] = []
        for sequence in group.itertuples(index=False):
            prior_passes = all_passes.loc[
                all_passes["buildup_sequence_id"].astype(str).isin(prior_ids)
            ].copy()
            metrics = aggregate_layer_network_metrics(prior_passes, int(goalkeeper_id))
            eligible = len(prior_ids) >= 3 and metrics["layer_gk_pass_attempts"] >= 3
            rows.append({
                "buildup_sequence_id": str(sequence.buildup_sequence_id),
                "past_layer_sequences": len(prior_ids),
                "past_layer_model_eligible": bool(eligible),
                **{f"past_{name}": value for name, value in metrics.items()},
            })
            prior_ids.append(str(sequence.buildup_sequence_id))
    return pd.DataFrame(rows)


def save_top_receiver_share_audit(
    sequence_dataset: pd.DataFrame,
    layer_dataset: pd.DataFrame,
) -> Path | None:
    """Contrast short-sequence averages with correctly pooled layer shares."""
    if sequence_dataset.empty or layer_dataset.empty:
        return None
    keys = ["match_id", "team", "goalkeeper_id", "pressure_level"]
    sequence_summary = (
        sequence_dataset.groupby(keys, dropna=False)
        .agg(
            sequences=("buildup_sequence_id", "size"),
            sequence_mean_top_receiver_share=("top_receiver_share", "mean"),
            sequence_gk_completed_passes=("gk_completed_passes", "sum"),
        )
        .reset_index()
    )
    audit = sequence_summary.merge(
        layer_dataset[keys + ["top_receiver_share", "layer_gk_completed_passes"]],
        on=keys,
        how="left",
    ).rename(columns={"top_receiver_share": "pooled_layer_top_receiver_share"})
    audit["sequence_mean_minus_pooled"] = (
        audit["sequence_mean_top_receiver_share"] - audit["pooled_layer_top_receiver_share"]
    )
    audit["interpretation"] = np.where(
        audit["sequence_mean_top_receiver_share"].ge(0.995)
        & audit["pooled_layer_top_receiver_share"].lt(0.995),
        "sequence-level averaging artifact",
        "inspect recipient linkage/true concentration",
    )
    output = TABLE_DIR / "top_receiver_share_aggregation_audit.csv"
    audit.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_model_ready_dataset(
    sequence_dataset: pd.DataFrame,
    layer_dataset: pd.DataFrame | None = None,
    leave_one_out_dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    model_columns = [
        "buildup_sequence_id",
        "match_id",
        "team",
        "opponent",
        "goalkeeper_id",
        "goalkeeper",
        "pressure_share",
        "pressure_anchor_events",
        "pressure_anchor_under_pressure_events",
        "pressure_anchor_share",
        "any_pressure_anchor",
        "pressure_level",
        "period",
        "half",
        "score_state",
        "opponent_strength",
        "stage_strength",
        "sequence_origin",
        "gk_starts_sequence",
        "from_goal_kick",
        "sequence_start_minute",
        "duration_seconds",
        "defensive_start",
        "defensive_third_exit_eligible",
        "weighted_out_strength",
        "betweenness",
        "receiver_diversity",
        "receiver_entropy",
        "receiver_hhi",
        "network_density",
        "global_efficiency",
        "forward_pass_share",
        "long_distribution_share",
        "top_receiver_share",
        "cb_fb_distribution_share",
        *OUTCOME_COLUMNS,
    ]
    available_columns = [column for column in model_columns if column in sequence_dataset.columns]
    model_data = sequence_dataset[available_columns].copy()
    if layer_dataset is not None and not layer_dataset.empty:
        layer_predictors = [
            "match_id",
            "team",
            "goalkeeper_id",
            "pressure_level",
            "layer_id",
            "sequences",
        ]
        model_data = model_data.merge(
            layer_dataset[layer_predictors],
            on=["match_id", "team", "goalkeeper_id", "pressure_level"],
            how="left",
            suffixes=("_sequence", "_layer"),
        )
        if "sequences" in model_data.columns:
            model_data = model_data.rename(columns={"sequences": "layer_sequences"})
    if leave_one_out_dataset is not None and not leave_one_out_dataset.empty:
        model_data = model_data.merge(
            leave_one_out_dataset,
            on="buildup_sequence_id",
            how="left",
        )
        for column in MODEL_NUMERIC_PREDICTORS:
            loo_column = f"loo_{column}"
            if loo_column in model_data.columns:
                model_data[column] = model_data[loo_column]
        if "loo_layer_entropy_eligible" in model_data.columns:
            model_data["entropy_model_eligible"] = model_data[
                "loo_layer_entropy_eligible"
            ].fillna(False).astype(bool)
    if "layer_id_x" in model_data.columns:
        model_data["layer_id"] = model_data["layer_id_x"]
    elif "layer_id_y" in model_data.columns:
        model_data["layer_id"] = model_data["layer_id_y"]
    for column in OUTCOME_COLUMNS:
        if column in model_data.columns:
            model_data[column] = pd.to_numeric(
                model_data[column],
                errors="coerce",
            ).astype("Int64")
    return model_data


def save_descriptive_result_tables(sequence_dataset: pd.DataFrame) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if sequence_dataset.empty:
        return outputs

    overview = pd.DataFrame(
        [
            {"metric": "matches", "value": int(sequence_dataset["match_id"].nunique())},
            {"metric": "teams", "value": int(sequence_dataset["team"].nunique())},
            {"metric": "goalkeepers", "value": int(sequence_dataset["goalkeeper_id"].nunique())},
            {"metric": "sequences", "value": int(len(sequence_dataset))},
            {"metric": "mean_sequence_duration_seconds", "value": float(sequence_dataset["duration_seconds"].mean())},
            {"metric": "controlled_final_third_entry_rate", "value": float(sequence_dataset["controlled_final_third_entry"].mean())},
            {"metric": "shot_created_rate", "value": float(sequence_dataset["shot_created"].mean())},
            {"metric": "lost_within_10s_rate", "value": float(sequence_dataset["lost_within_10s"].mean())},
        ]
    )
    outputs["sample_overview"] = TABLE_DIR / "result_01_sample_overview.csv"
    overview.to_csv(outputs["sample_overview"], index=False, encoding="utf-8-sig")

    pressure = (
        sequence_dataset.groupby("pressure_level")
        .agg(
            sequences=("buildup_sequence_id", "size"),
            defensive_third_exit_rate=("defensive_third_exit", "mean"),
            entry_x60_rate=("entry_x60", "mean"),
            controlled_final_third_entry_rate=("controlled_final_third_entry", "mean"),
            shot_created_rate=("shot_created", "mean"),
            lost_within_10s_rate=("lost_within_10s", "mean"),
            opponent_shot_after_loss_10s_rate=("opponent_shot_after_loss_10s", "mean"),
            mean_global_efficiency=("global_efficiency", "mean"),
            mean_receiver_entropy=("receiver_entropy", "mean"),
            mean_long_distribution_share=("long_distribution_share", "mean"),
        )
        .reset_index()
    )
    outputs["pressure_descriptives"] = TABLE_DIR / "result_02_pressure_descriptives.csv"
    pressure.to_csv(outputs["pressure_descriptives"], index=False, encoding="utf-8-sig")

    origins = (
        sequence_dataset.groupby("sequence_origin")
        .agg(
            sequences=("buildup_sequence_id", "size"),
            defensive_third_exit_rate=("defensive_third_exit", "mean"),
            controlled_final_third_entry_rate=("controlled_final_third_entry", "mean"),
            shot_created_rate=("shot_created", "mean"),
            lost_within_10s_rate=("lost_within_10s", "mean"),
            mean_weighted_out_strength=("weighted_out_strength", "mean"),
            mean_network_density=("network_density", "mean"),
            mean_global_efficiency=("global_efficiency", "mean"),
        )
        .sort_values("sequences", ascending=False)
        .reset_index()
    )
    outputs["origin_descriptives"] = TABLE_DIR / "result_03_sequence_origin_descriptives.csv"
    origins.to_csv(outputs["origin_descriptives"], index=False, encoding="utf-8-sig")

    goalkeepers = (
        sequence_dataset.groupby(["goalkeeper_id", "goalkeeper", "team"])
        .agg(
            sequences=("buildup_sequence_id", "size"),
            controlled_final_third_entry_rate=("controlled_final_third_entry", "mean"),
            shot_created_rate=("shot_created", "mean"),
            lost_within_10s_rate=("lost_within_10s", "mean"),
            weighted_out_strength_mean=("weighted_out_strength", "mean"),
            betweenness_mean=("betweenness", "mean"),
            receiver_entropy_mean=("receiver_entropy", "mean"),
            receiver_hhi_mean=("receiver_hhi", "mean"),
            network_density_mean=("network_density", "mean"),
            global_efficiency_mean=("global_efficiency", "mean"),
            long_distribution_share_mean=("long_distribution_share", "mean"),
            cb_fb_distribution_share_mean=("cb_fb_distribution_share", "mean"),
        )
        .sort_values("sequences", ascending=False)
        .reset_index()
    )
    outputs["goalkeeper_summary"] = TABLE_DIR / "result_04_goalkeeper_summary.csv"
    goalkeepers.to_csv(outputs["goalkeeper_summary"], index=False, encoding="utf-8-sig")

    quartile_source = sequence_dataset.copy()
    quartile_source["global_efficiency_quartile"] = pd.qcut(
        quartile_source["global_efficiency"].rank(method="first"),
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )
    quartiles = (
        quartile_source.groupby("global_efficiency_quartile", observed=False)
        .agg(
            sequences=("buildup_sequence_id", "size"),
            mean_global_efficiency=("global_efficiency", "mean"),
            controlled_final_third_entry_rate=("controlled_final_third_entry", "mean"),
            shot_created_rate=("shot_created", "mean"),
            lost_within_10s_rate=("lost_within_10s", "mean"),
        )
        .reset_index()
    )
    outputs["network_quartiles"] = TABLE_DIR / "result_05_network_efficiency_quartiles.csv"
    quartiles.to_csv(outputs["network_quartiles"], index=False, encoding="utf-8-sig")

    return outputs


def normal_ci(series: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, float("nan"), float("nan")
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def save_table_1_sample_characteristics(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None
    gk_counts = sequence_dataset.groupby(["goalkeeper_id", "goalkeeper", "team"]).size()
    rows: list[dict[str, Any]] = [
        {"section": "sample", "category": "matches", "n": int(sequence_dataset["match_id"].nunique()), "value": float("nan")},
        {"section": "sample", "category": "teams", "n": int(sequence_dataset["team"].nunique()), "value": float("nan")},
        {"section": "sample", "category": "goalkeepers", "n": int(sequence_dataset["goalkeeper_id"].nunique()), "value": float("nan")},
        {"section": "sample", "category": "sequences", "n": int(len(sequence_dataset)), "value": float("nan")},
        {"section": "sample", "category": "median_sequences_per_goalkeeper", "n": int(len(gk_counts)), "value": float(gk_counts.median())},
    ]
    for section, column in [
        ("pressure_level", "pressure_level"),
        ("score_state", "score_state"),
        ("sequence_origin", "sequence_origin"),
    ]:
        for category, count in sequence_dataset[column].value_counts(dropna=False).items():
            rows.append({"section": section, "category": str(category), "n": int(count), "value": float("nan")})
    for outcome in OUTCOME_COLUMNS:
        outcome_values = pd.to_numeric(sequence_dataset[outcome], errors="coerce")
        eligible_n = int(outcome_values.notna().sum())
        events = int(outcome_values.sum())
        rate = safe_divide(events, eligible_n)
        rows.append(
            {
                "section": "outcomes",
                "category": outcome,
                "n": eligible_n,
                "events": events,
                "eligible_n": eligible_n,
                "rate": rate,
                "value": rate,
            }
        )
    for column in ["team_completed_passes", "network_nodes", "network_edges"]:
        mean, low, high = normal_ci(sequence_dataset[column])
        rows.append({"section": "sequence_characteristics", "category": column, "n": int(sequence_dataset[column].notna().sum()), "value": mean, "ci_low": low, "ci_high": high})

    table = pd.DataFrame(rows)
    output = TABLE_DIR / "table_1_sample_and_sequence_characteristics.csv"
    table.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_table_2_pressure_network_characteristics(layer_dataset: pd.DataFrame) -> Path | None:
    if layer_dataset.empty:
        return None
    layer_dataset = layer_dataset.loc[layer_dataset["sequences"].ge(4)].copy()
    if layer_dataset.empty:
        return None
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        smf = None
    variables = {
        "weighted_out_strength_per_sequence": "goalkeeper_completed_pass_involvement_per_sequence",
        "long_distribution_share": "long_distribution_share",
        "cb_fb_distribution_share": "centreback_fullback_distribution_share",
        "global_efficiency": "global_efficiency",
        "controlled_final_third_entry_rate": "progression_success_rate",
    }
    rows: list[dict[str, Any]] = []
    reference: dict[str, float] = {}
    adjusted_differences: dict[tuple[str, str], tuple[float, float, float]] = {}
    adjusted_data = layer_dataset.loc[
        layer_dataset["pressure_level"].isin(["low", "none"])
    ].copy()
    adjusted_data["pressure_level"] = pd.Categorical(
        adjusted_data["pressure_level"],
        categories=["low", "none"],
    )
    if smf is not None:
        for variable in variables:
            analysis_data = adjusted_data.dropna(subset=[variable]).copy()
            if analysis_data["pressure_level"].nunique() < 2:
                continue
            formula = (
                f"{variable} ~ C(pressure_level) + stage_strength "
                "+ sequence_start_minute + C(goalkeeper_id)"
            )
            try:
                # Goalkeeper fixed effects can create unit-leverage rows in this
                # small layer-level sample, for which HC3 variances diverge.
                result = smf.ols(formula, data=analysis_data).fit()
            except Exception:
                continue
            for pressure in ["none"]:
                term = f"C(pressure_level)[T.{pressure}]"
                if term not in result.params.index:
                    continue
                adjusted_differences[(variable, pressure)] = (
                    float(result.params[term]),
                    float(result.conf_int().loc[term, 0]),
                    float(result.conf_int().loc[term, 1]),
                )
    for variable in variables:
        reference[variable] = float(layer_dataset.loc[layer_dataset["pressure_level"].eq("low"), variable].mean())
    for pressure, group in layer_dataset.groupby("pressure_level", dropna=False):
        for variable, label in variables.items():
            mean, low, high = normal_ci(group[variable])
            rows.append(
                {
                    "pressure_level": pressure,
                    "metric": label,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "unadjusted_difference_vs_low": mean - reference.get(variable, float("nan")),
                    "adjusted_difference_vs_low": (
                        0.0 if pressure == "low" else adjusted_differences.get((variable, str(pressure)), (float("nan"), float("nan"), float("nan")))[0]
                    ),
                    "adjusted_ci_low": (
                        0.0 if pressure == "low" else adjusted_differences.get((variable, str(pressure)), (float("nan"), float("nan"), float("nan")))[1]
                    ),
                    "adjusted_ci_high": (
                        0.0 if pressure == "low" else adjusted_differences.get((variable, str(pressure)), (float("nan"), float("nan"), float("nan")))[2]
                    ),
                }
            )
    output = TABLE_DIR / "table_2_pressure_network_distribution_characteristics.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_all_goalkeeper_pressure_differences(layer_dataset: pd.DataFrame) -> Path | None:
    if layer_dataset.empty:
        return None
    metrics = [
        "receiver_entropy",
        "global_efficiency",
        "long_distribution_share",
        "cb_fb_distribution_share",
        "controlled_final_third_entry_rate",
        "lost_within_10s_rate",
    ]
    grouped = (
        layer_dataset.groupby(["goalkeeper_id", "goalkeeper", "team", "pressure_level"])
        .agg(
            **{metric: (metric, "mean") for metric in metrics},
            total_sequences=("sequences", "sum"),
        )
        .reset_index()
    )
    low = grouped.loc[grouped["pressure_level"].eq("low")].set_index(["goalkeeper_id", "goalkeeper", "team"])
    high = grouped.loc[grouped["pressure_level"].eq("high")].set_index(["goalkeeper_id", "goalkeeper", "team"])
    shared = low.index.intersection(high.index)
    if shared.empty:
        return None
    result = pd.DataFrame(index=shared).reset_index()
    result["low_pressure_total_sequences"] = low.loc[shared, "total_sequences"].to_numpy()
    result["high_pressure_total_sequences"] = high.loc[shared, "total_sequences"].to_numpy()
    for metric in metrics:
        result[f"low_{metric}"] = low.loc[shared, metric].to_numpy()
        result[f"high_{metric}"] = high.loc[shared, metric].to_numpy()
        result[f"high_minus_low_{metric}"] = high.loc[shared, metric].to_numpy() - low.loc[shared, metric].to_numpy()
    output = TABLE_DIR / "all_goalkeeper_low_high_pressure_differences.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_table_3_main_multilevel_models(
    model_results: pd.DataFrame,
    pressure_only_results: pd.DataFrame | None = None,
) -> Path | None:
    if model_results.empty and (pressure_only_results is None or pressure_only_results.empty):
        return None
    main_outcomes = CORE_MODEL_OUTCOMES
    focus_terms = [
        PRESSURE_MODEL_TERM,
        "z_long_distribution_share",
        "z_cb_fb_distribution_share",
        "z_global_efficiency",
        "C(score_state)[T.losing]",
        "C(score_state)[T.winning]",
    ]
    table = model_results.loc[
        model_results["outcome"].isin(main_outcomes)
        & model_results["term"].isin(focus_terms)
    ].copy()
    table.insert(0, "model_scope", "network_augmented_LOO_eligible")
    if pressure_only_results is not None and not pressure_only_results.empty:
        pressure_table = pressure_only_results.loc[
            pressure_only_results["outcome"].isin(main_outcomes)
            & pressure_only_results["term"].eq(PRESSURE_MODEL_TERM)
        ].copy()
        pressure_table.insert(0, "model_scope", "pressure_only_full_sequence_sample")
        table = pd.concat([pressure_table, table], ignore_index=True, sort=False)
    output = TABLE_DIR / "table_3_main_multilevel_models.csv"
    table.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def fit_pressure_network_interaction_models(model_data: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return pd.DataFrame()

    data = prepare_model_data(model_data)
    # Pressure uses a fixed natural unit: 0.5 share = one-anchor exposure.
    outcomes = CORE_MODEL_OUTCOMES
    network_terms = [
        "long_distribution_share",
    ]
    covariates = [
        "C(half)",
        "C(score_state)",
        "C(sequence_origin)",
        "z_stage_strength",
        "z_sequence_start_minute",
    ]
    variance_components = {
        "team": "0 + C(team)",
        "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)",
        "pressure_layer": "0 + C(layer_id)",
    }
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        outcome_data = data.dropna(
            subset=[outcome, *[f"z_{column}" for column in LAYER_NETWORK_PREDICTORS]],
        ).copy()
        if "layer_model_eligible" in outcome_data.columns:
            outcome_data = outcome_data.loc[
                outcome_data["layer_model_eligible"].fillna(False)
            ].copy()
        if outcome == "defensive_third_exit" and "defensive_third_exit_eligible" in outcome_data.columns:
            outcome_data = outcome_data.loc[outcome_data["defensive_third_exit_eligible"].eq(1)].copy()
        if outcome_data.empty or outcome_data[outcome].nunique() < 2:
            continue
        for network_term in network_terms:
            standardized_term = f"z_{network_term}"
            other_network_main_effects = [
                f"z_{term}" for term in LAYER_NETWORK_PREDICTORS
                if term != network_term
            ]
            formula = (
                f"{outcome} ~ {PRESSURE_MODEL_TERM} * {standardized_term} + "
                + " + ".join([*other_network_main_effects, *covariates])
            )
            try:
                model = BinomialBayesMixedGLM.from_formula(formula, variance_components, outcome_data)
                result = fit_vb_deterministic(model)
            except Exception as error:
                rows.append({"outcome": outcome, "network_term": network_term, "term": "model_error", "error": repr(error)})
                continue
            for term, mean, sd in zip(model.exog_names, result.fe_mean, result.fe_sd, strict=False):
                if ":" not in term:
                    continue
                ci_low = float(mean - 1.96 * sd)
                ci_high = float(mean + 1.96 * sd)
                rows.append(
                    {
                        "outcome": outcome,
                        "network_term": network_term,
                        "term": term,
                        "log_odds": float(mean),
                        "posterior_sd": float(sd),
                        "odds_ratio": float(math.exp(mean)),
                        "odds_ratio_ci_low": float(math.exp(ci_low)),
                        "odds_ratio_ci_high": float(math.exp(ci_high)),
                    }
                )
            coefficient_mean = dict(zip(model.exog_names, result.fe_mean, strict=False))
            coefficient_sd = dict(zip(model.exog_names, result.fe_sd, strict=False))
            random_generator = np.random.default_rng(42)
            simulations = {
                name: random_generator.normal(coefficient_mean[name], coefficient_sd[name], 2000)
                for name in coefficient_mean
            }
            for pressure_value, pressure_label in [
                (0.00, "No pressure-anchor exposure (0.00)"),
                (0.50, "One-anchor exposure (0.50)"),
            ]:
                pressure_half_units = pressure_value / 0.5
                probability_by_z: dict[int, np.ndarray] = {}
                for z_value in [-1, 1]:
                    linear_predictor = simulations.get("Intercept", np.zeros(2000)).copy()
                    linear_predictor += pressure_half_units * simulations.get(
                        PRESSURE_MODEL_TERM,
                        0.0,
                    )
                    linear_predictor += z_value * simulations.get(standardized_term, 0.0)
                    interaction_term = f"{PRESSURE_MODEL_TERM}:{standardized_term}"
                    alternate_interaction_term = f"{standardized_term}:{PRESSURE_MODEL_TERM}"
                    linear_predictor += pressure_half_units * z_value * simulations.get(
                        interaction_term,
                        simulations.get(alternate_interaction_term, 0.0),
                    )
                    probability_by_z[z_value] = 1.0 / (1.0 + np.exp(-linear_predictor))
                    values = probability_by_z[z_value]
                    prediction_rows.append(
                        {
                            "outcome": outcome,
                            "network_term": network_term,
                            "pressure_level": pressure_label,
                            "pressure_half_units": pressure_half_units,
                            "pressure_value": pressure_value,
                            "pressure_effect_unit": "0.5 pressure-anchor-share increase",
                            "prediction_in_observed_model_range": True,
                            "network_z": z_value,
                            "adjusted_probability": float(np.mean(values)),
                            "credible_interval_low": float(np.quantile(values, 0.025)),
                            "credible_interval_high": float(np.quantile(values, 0.975)),
                        }
                    )
                marginal_effect = probability_by_z[1] - probability_by_z[-1]
                prediction_rows.append(
                    {
                        "outcome": outcome,
                        "network_term": network_term,
                        "pressure_level": pressure_label,
                        "pressure_half_units": pressure_half_units,
                        "pressure_value": pressure_value,
                        "pressure_effect_unit": "0.5 pressure-anchor-share increase",
                        "prediction_in_observed_model_range": True,
                        "network_z": "AME_plus1_minus1",
                        "adjusted_probability": float(np.mean(marginal_effect)),
                        "credible_interval_low": float(np.quantile(marginal_effect, 0.025)),
                        "credible_interval_high": float(np.quantile(marginal_effect, 0.975)),
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(TABLE_DIR / "table_4_pressure_network_interactions.csv", index=False, encoding="utf-8-sig")
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(
            TABLE_DIR / "table_4_adjusted_predictions_and_ame.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return table


def save_table_4_key_interaction_and_probabilities(
    interaction_results: pd.DataFrame,
) -> Path | None:
    """Create the combined interaction and adjusted-probability result table."""
    prediction_path = TABLE_DIR / "table_4_adjusted_predictions_and_ame.csv"
    if interaction_results.empty and not prediction_path.exists():
        return None

    sections: list[pd.DataFrame] = []
    if not interaction_results.empty:
        interaction = interaction_results.loc[
            interaction_results["outcome"].isin(CORE_MODEL_OUTCOMES)
            & interaction_results["network_term"].eq("long_distribution_share")
        ].copy()
        if not interaction.empty:
            interaction.insert(0, "table_section", "pressure_x_long_distribution_interaction")
            sections.append(interaction)
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        predictions = predictions.loc[
            predictions["outcome"].isin(CORE_MODEL_OUTCOMES)
            & predictions["network_term"].eq("long_distribution_share")
        ].copy()
        if not predictions.empty:
            predictions.insert(0, "table_section", "adjusted_probabilities_and_ame")
            sections.append(predictions)
    if not sections:
        return None
    output = TABLE_DIR / "table_4_key_interaction_and_adjusted_probabilities.csv"
    pd.concat(sections, ignore_index=True, sort=False).to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    return output


def fit_pressure_only_multilevel_models(sequence_dataset: pd.DataFrame) -> pd.DataFrame:
    """Estimate early pressure exposure on all sequences without LOO-network eligibility."""
    if sequence_dataset.empty:
        return pd.DataFrame()
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return pd.DataFrame()

    data = sequence_dataset.copy()
    numeric_columns = ["pressure_anchor_share", "stage_strength", "sequence_start_minute"]
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if column != "pressure_anchor_share":
            values = values.fillna(float(values.mean()))
        data[column] = values
        if column == "pressure_anchor_share":
            data[PRESSURE_MODEL_TERM] = values / 0.5
        else:
            standard_deviation = float(values.std(ddof=0))
            data[f"z_{column}"] = (
                (values - float(values.mean())) / standard_deviation
                if standard_deviation > 0 else 0.0
            )
    for column in MODEL_CATEGORICAL_PREDICTORS:
        data[column] = data[column].fillna("unknown").astype(str)

    predictors = [
        PRESSURE_MODEL_TERM,
        "z_stage_strength",
        "z_sequence_start_minute",
        *[f"C({column})" for column in MODEL_CATEGORICAL_PREDICTORS],
    ]
    # The pressure-only model deliberately omits pressure-layer and all LOO
    # network terms, allowing pressure-anchor shares 0, 0.5 and 1.0 to enter.
    variance_components = {
        "team": "0 + C(team)",
        "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)",
    }
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for outcome in CORE_MODEL_OUTCOMES:
        outcome_data = data.dropna(subset=[outcome, PRESSURE_MODEL_TERM]).copy()
        if outcome_data.empty or outcome_data[outcome].nunique() < 2:
            continue
        formula = f"{outcome} ~ " + " + ".join(predictors)
        try:
            model = BinomialBayesMixedGLM.from_formula(formula, variance_components, outcome_data)
            result = fit_vb_deterministic(model)
            (MODEL_DIR / f"{outcome}_pressure_only_full_sample_summary.txt").write_text(
                str(result.summary()), encoding="utf-8"
            )
            for term, mean, sd in zip(model.exog_names, result.fe_mean, result.fe_sd, strict=False):
                ci_low, ci_high = float(mean - 1.96 * sd), float(mean + 1.96 * sd)
                rows.append({
                    "outcome": outcome,
                    "term": term,
                    "sequences": int(len(outcome_data)),
                    "pressure_anchor_share_values": "|".join(
                        f"{value:g}" for value in sorted(outcome_data["pressure_anchor_share"].dropna().unique())
                    ),
                    "log_odds": float(mean),
                    "posterior_sd": float(sd),
                    "odds_ratio": float(math.exp(mean)),
                    "odds_ratio_ci_low": float(math.exp(ci_low)),
                    "odds_ratio_ci_high": float(math.exp(ci_high)),
                    "estimation": "variational Bayes",
                    "random_intercepts": "team|match|goalkeeper",
                })
            optim_retvals = getattr(result, "optim_retvals", {}) or {}
            diagnostics.append({
                "outcome": outcome,
                "sequences": int(len(outcome_data)),
                "converged": bool(optim_retvals.get("success", True)),
                "message": str(optim_retvals.get("message", "")),
            })
        except Exception as error:
            rows.append({"outcome": outcome, "term": "model_error", "error": repr(error)})
    result_table = pd.DataFrame(rows)
    result_table.to_csv(MODEL_DIR / "pressure_only_full_sample_fixed_effect_odds_ratios.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(MODEL_DIR / "pressure_only_full_sample_diagnostics.csv", index=False, encoding="utf-8-sig")
    return result_table


def save_table_5_sensitivity_robustness(model_data: pd.DataFrame) -> Path | None:
    """Repeat the primary Bayesian multilevel model under one-at-a-time restrictions."""
    if model_data.empty:
        return None
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None

    # Standardize once on the same model-ready data as the primary model, then
    # subset. This prevents scenario-specific changes of predictor scale.
    data = prepare_model_data(model_data)
    scenarios = [
        ("loo_eligibility_3_sequences_3_gk_passes", pd.Series(True, index=data.index), "controlled_final_third_entry", 3, 3),
        ("both_anchor_actions_observed", data["pressure_anchor_events"].eq(2), "controlled_final_third_entry", 3, 3),
        ("goal_kicks_only", data["from_goal_kick"].eq(1), "controlled_final_third_entry", 3, 3),
        ("goalkeeper_first_sequences_only", data["gk_starts_sequence"].eq(1), "controlled_final_third_entry", 3, 3),
        ("regulation_time_only", data["period"].le(2), "controlled_final_third_entry", 3, 3),
        ("layer_eligibility_4_sequences_4_gk_passes", pd.Series(True, index=data.index), "controlled_final_third_entry", 4, 4),
        ("layer_eligibility_5_sequences_5_gk_passes", pd.Series(True, index=data.index), "controlled_final_third_entry", 5, 5),
    ]
    predictors = [PRESSURE_MODEL_TERM] + [
        f"z_{column}" for column in MODEL_NUMERIC_PREDICTORS
        if column != "pressure_anchor_share"
    ] + [
        f"C({column})" for column in MODEL_CATEGORICAL_PREDICTORS
    ]
    variance_components = {
        "team": "0 + C(team)",
        "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)",
        "pressure_layer": "0 + C(layer_id)",
    }
    rows: list[dict[str, Any]] = []
    for name, scenario_mask, outcome, minimum_sequences, minimum_gk_passes in scenarios:
        scenario_data = data.loc[scenario_mask.fillna(False)].copy()
        scenario_data = scenario_data.dropna(
            subset=[outcome, *[f"z_{column}" for column in LAYER_NETWORK_PREDICTORS]],
        )
        scenario_data = scenario_data.loc[
            scenario_data["layer_sequences_excluding"].ge(minimum_sequences)
            & scenario_data["loo_layer_gk_pass_attempts"].ge(minimum_gk_passes)
        ].copy()
        base_row = {
            "scenario": name,
            "outcome": outcome,
            "minimum_layer_sequences_excluding": minimum_sequences,
            "minimum_layer_gk_pass_attempts": minimum_gk_passes,
            "sequences": int(len(scenario_data)),
            "goalkeepers": int(scenario_data["goalkeeper_id"].nunique()) if not scenario_data.empty else 0,
            "pressure_values": "|".join(
                f"{value:g}" for value in sorted(scenario_data["pressure_anchor_share"].dropna().unique())
            ),
            "outcome_rate": float(scenario_data[outcome].mean()) if not scenario_data.empty else float("nan"),
            "model": "Bayesian multilevel logistic (VB)",
            "random_intercepts": "team|match|goalkeeper|pressure_layer",
        }
        if len(scenario_data) < 30:
            base_row["status"] = "not_estimable_sample_below_30"
            rows.append(base_row)
            continue
        if scenario_data["pressure_anchor_share"].nunique() < 2:
            base_row["status"] = "not_estimable_no_pressure_contrast"
            rows.append(base_row)
            continue
        if scenario_data[outcome].nunique() < 2:
            base_row["status"] = "not_estimable_no_outcome_variation"
            rows.append(base_row)
            continue
        try:
            formula = f"{outcome} ~ " + " + ".join(predictors)
            model = BinomialBayesMixedGLM.from_formula(
                formula,
                variance_components,
                scenario_data,
            )
            result = fit_vb_deterministic(model)
            pressure_index = model.exog_names.index(PRESSURE_MODEL_TERM)
            mean = float(result.fe_mean[pressure_index])
            sd = float(result.fe_sd[pressure_index])
            base_row.update(
                {
                    "status": "estimated",
                    "pressure_odds_ratio_per_0_5_unit": float(math.exp(mean)),
                    "credible_interval_low": float(math.exp(mean - 1.96 * sd)),
                    "credible_interval_high": float(math.exp(mean + 1.96 * sd)),
                    "interval_method": "variational_posterior_95pct",
                }
            )
        except Exception as error:
            base_row["status"] = "model_error"
            base_row["model_error"] = repr(error)
        rows.append(base_row)
    output = TABLE_DIR / "table_5_sensitivity_robustness.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_global_efficiency_size_sensitivity(model_data: pd.DataFrame) -> Path | None:
    """Check whether global efficiency is explained by layer pass volume."""
    required = {
        "global_efficiency",
        "loo_layer_completed_pass_count",
        "loo_layer_node_count",
    }
    if model_data.empty or not required.issubset(model_data.columns):
        return None

    correlation_rows: list[dict[str, Any]] = []
    for size_term in ["loo_layer_completed_pass_count", "loo_layer_node_count"]:
        pair = model_data[["global_efficiency", size_term]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        correlation_rows.append(
            {
                "size_measure": size_term,
                "n": int(len(pair)),
                "pearson_r": float(pair["global_efficiency"].corr(pair[size_term], method="pearson")),
                "spearman_rho": float(pair["global_efficiency"].corr(pair[size_term], method="spearman")),
            }
        )
    pd.DataFrame(correlation_rows).to_csv(
        TABLE_DIR / "global_efficiency_layer_size_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None
    data = prepare_model_data(model_data)
    size_terms = ["loo_layer_completed_pass_count", "loo_layer_node_count"]
    for size_term in size_terms:
        values = pd.to_numeric(data[size_term], errors="coerce")
        size_sd = float(values.std(ddof=0))
        data[f"z_{size_term}"] = (
            (values - float(values.mean())) / size_sd if size_sd > 0 else 0.0
        )
    predictors = [PRESSURE_MODEL_TERM] + [
        f"z_{column}" for column in MODEL_NUMERIC_PREDICTORS
        if column != "pressure_anchor_share"
    ] + [
        "z_loo_layer_completed_pass_count",
        "z_loo_layer_node_count",
        *[f"C({column})" for column in MODEL_CATEGORICAL_PREDICTORS],
    ]
    variance_components = {
        "team": "0 + C(team)",
        "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)",
        "pressure_layer": "0 + C(layer_id)",
    }
    rows: list[dict[str, Any]] = []
    for outcome in ["controlled_final_third_entry", "lost_within_10s"]:
        outcome_data = data.dropna(
            subset=[outcome, *[f"z_{column}" for column in LAYER_NETWORK_PREDICTORS], "z_loo_layer_completed_pass_count", "z_loo_layer_node_count"]
        ).copy()
        if "layer_model_eligible" in outcome_data.columns:
            outcome_data = outcome_data.loc[outcome_data["layer_model_eligible"].fillna(False)]
        if outcome_data.empty or outcome_data[outcome].nunique() < 2:
            continue
        try:
            model = BinomialBayesMixedGLM.from_formula(
                f"{outcome} ~ " + " + ".join(predictors),
                variance_components,
                outcome_data,
            )
            result = fit_vb_deterministic(model)
            for term in [PRESSURE_MODEL_TERM, "z_global_efficiency", "z_long_distribution_share", "z_loo_layer_completed_pass_count", "z_loo_layer_node_count"]:
                index = model.exog_names.index(term)
                mean = float(result.fe_mean[index])
                sd = float(result.fe_sd[index])
                rows.append({
                    "outcome": outcome,
                    "term": term,
                    "sequences": int(len(outcome_data)),
                    "odds_ratio": float(math.exp(mean)),
                    "odds_ratio_ci_low": float(math.exp(mean - 1.96 * sd)),
                    "odds_ratio_ci_high": float(math.exp(mean + 1.96 * sd)),
                    "estimation": "variational Bayes; additionally adjusted for LOO completed-pass and node counts",
                })
        except Exception as error:
            rows.append({"outcome": outcome, "term": "model_error", "error": repr(error)})
    output = TABLE_DIR / "global_efficiency_pass_and_node_count_sensitivity.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_alternative_estimation_sensitivity(model_data: pd.DataFrame) -> Path | None:
    """Refit both primary models and their interaction by Laplace/MAP."""
    if model_data.empty:
        return None
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None
    data = prepare_model_data(model_data)
    predictors = [PRESSURE_MODEL_TERM] + [
        f"z_{column}" for column in MODEL_NUMERIC_PREDICTORS
        if column != "pressure_anchor_share"
    ] + [
        f"C({column})" for column in MODEL_CATEGORICAL_PREDICTORS
    ]
    variance_components = {
        "team": "0 + C(team)",
        "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)",
        "pressure_layer": "0 + C(layer_id)",
    }
    output = TABLE_DIR / "primary_and_interaction_alternative_estimation.csv"
    rows: list[dict[str, Any]] = []
    for outcome in CORE_MODEL_OUTCOMES:
        outcome_data = data.dropna(subset=[outcome, *[f"z_{column}" for column in LAYER_NETWORK_PREDICTORS]]).copy()
        if "layer_model_eligible" in outcome_data.columns:
            outcome_data = outcome_data.loc[outcome_data["layer_model_eligible"].fillna(False)].copy()
        model_specs = {
            "primary": f"{outcome} ~ " + " + ".join(predictors),
            "pressure_x_long_distribution": (
                f"{outcome} ~ {PRESSURE_MODEL_TERM} * z_long_distribution_share + "
                "z_global_efficiency + z_cb_fb_distribution_share + z_stage_strength + "
                "z_sequence_start_minute + C(half) + C(score_state) + C(sequence_origin)"
            ),
        }
        for specification, formula in model_specs.items():
            try:
                model = BinomialBayesMixedGLM.from_formula(formula, variance_components, outcome_data)
                result = model.fit_map()
                optim_retvals = getattr(result, "optim_retvals", {}) or {}
                optimizer_success = bool(optim_retvals.get("success", False))
                optimizer_message = str(optim_retvals.get("message", ""))
                gradient = optim_retvals.get("jac")
                gradient_norm = (
                    float(np.linalg.norm(np.asarray(gradient, dtype=float)))
                    if gradient is not None else float("nan")
                )
                acceptable = (
                    optimizer_success
                    and np.isfinite(gradient_norm)
                    and gradient_norm <= MAP_GRADIENT_NORM_THRESHOLD
                )
                focus_terms = [
                    PRESSURE_MODEL_TERM,
                    "z_global_efficiency",
                    "z_long_distribution_share",
                    "z_cb_fb_distribution_share",
                    f"{PRESSURE_MODEL_TERM}:z_long_distribution_share",
                ]
                for term in focus_terms:
                    if term not in model.exog_names:
                        continue
                    index = model.exog_names.index(term)
                    mean, sd = float(result.fe_mean[index]), float(result.fe_sd[index])
                    rows.append({
                        "outcome": outcome,
                        "model_specification": specification,
                        "term": term,
                        "sequences": int(len(outcome_data)),
                        "odds_ratio": float(math.exp(mean)),
                        "odds_ratio_ci_low": float(math.exp(mean - 1.96 * sd)),
                        "odds_ratio_ci_high": float(math.exp(mean + 1.96 * sd)),
                        "estimation": "Laplace approximation at posterior mode (fit_map)",
                        "converged": acceptable,
                        "optimizer_success": optimizer_success,
                        "optimizer_message": optimizer_message,
                        "gradient_norm": gradient_norm,
                    })
            except Exception as error:
                rows.append({"outcome": outcome, "model_specification": specification, "term": "model_error", "error": repr(error)})
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_past_only_network_sensitivity(
    model_data: pd.DataFrame,
    past_metrics: pd.DataFrame,
) -> Path | None:
    """Refit the primary model using only networks observed before each sequence."""
    if model_data.empty or past_metrics.empty:
        return None
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None
    data = model_data.merge(past_metrics, on="buildup_sequence_id", how="left")
    for column in LAYER_NETWORK_PREDICTORS:
        data[column] = data[f"past_{column}"]
    data["layer_model_eligible"] = data["past_layer_model_eligible"].fillna(False)
    data = prepare_model_data(data)
    predictors = [PRESSURE_MODEL_TERM, *[f"z_{x}" for x in LAYER_NETWORK_PREDICTORS],
                  "z_stage_strength", "z_sequence_start_minute",
                  *[f"C({x})" for x in MODEL_CATEGORICAL_PREDICTORS]]
    variance_components = {
        "team": "0 + C(team)", "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)", "pressure_layer": "0 + C(layer_id)",
    }
    rows: list[dict[str, Any]] = []
    for outcome in CORE_MODEL_OUTCOMES:
        subset = data.loc[data["layer_model_eligible"]].dropna(
            subset=[outcome, *[f"z_{x}" for x in LAYER_NETWORK_PREDICTORS]]
        ).copy()
        if subset.empty or subset[outcome].nunique() < 2:
            continue
        try:
            model = BinomialBayesMixedGLM.from_formula(
                f"{outcome} ~ " + " + ".join(predictors), variance_components, subset
            )
            result = fit_vb_deterministic(model)
            for term in [PRESSURE_MODEL_TERM, *[f"z_{x}" for x in LAYER_NETWORK_PREDICTORS]]:
                index = model.exog_names.index(term)
                mean, sd = float(result.fe_mean[index]), float(result.fe_sd[index])
                rows.append({"outcome": outcome, "term": term, "sequences": len(subset),
                             "odds_ratio": math.exp(mean),
                             "odds_ratio_ci_low": math.exp(mean - 1.96 * sd),
                             "odds_ratio_ci_high": math.exp(mean + 1.96 * sd),
                             "network_definition": "same-match same-layer sequences strictly preceding focal sequence"})
        except Exception as error:
            rows.append({"outcome": outcome, "term": "model_error", "error": repr(error)})
    output = TABLE_DIR / "past_only_network_sensitivity.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_gee_estimation_sensitivity(model_data: pd.DataFrame) -> Path | None:
    """GEE check clustered by match with team fixed effects only.

    Goalkeeper fixed effects are intentionally excluded to avoid an over-specified
    population-average sensitivity model. Match dependence is handled by the
    exchangeable working correlation and sandwich-robust covariance estimator.
    """
    if model_data.empty:
        return None
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    data = prepare_model_data(model_data)
    data = data.loc[data["layer_model_eligible"].fillna(False)].copy()
    predictors = [PRESSURE_MODEL_TERM, *[f"z_{x}" for x in LAYER_NETWORK_PREDICTORS],
                  "z_stage_strength", "z_sequence_start_minute", "C(half)",
                  "C(score_state)", "C(sequence_origin)", "C(team)"]
    rows: list[dict[str, Any]] = []
    for outcome in CORE_MODEL_OUTCOMES:
        subset = data.dropna(subset=[outcome, *[f"z_{x}" for x in LAYER_NETWORK_PREDICTORS]]).copy()
        try:
            result = smf.gee(f"{outcome} ~ " + " + ".join(predictors), "match_id", subset,
                             family=sm.families.Binomial(),
                             cov_struct=sm.cov_struct.Exchangeable()).fit()
            for term in [PRESSURE_MODEL_TERM, *[f"z_{x}" for x in LAYER_NETWORK_PREDICTORS]]:
                mean, se = float(result.params[term]), float(result.bse[term])
                rows.append({"outcome": outcome, "term": term, "sequences": len(subset),
                             "matches": int(subset["match_id"].nunique()),
                             "odds_ratio": math.exp(mean),
                             "odds_ratio_ci_low": math.exp(mean - 1.96 * se),
                             "odds_ratio_ci_high": math.exp(mean + 1.96 * se),
                             "converged": bool(getattr(result, "converged", True)),
                             "estimation": "GEE exchangeable; match-cluster robust SE; team fixed effects"})
        except Exception as error:
            rows.append({"outcome": outcome, "term": "model_error", "error": repr(error)})
    output = TABLE_DIR / "frequentist_gee_estimation_sensitivity.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


# ---------------------------------------------------------------------
# 6. Research figures
# ---------------------------------------------------------------------

def clean_rate_axis(axis: plt.Axes, ylabel: str = "Sequence success rate") -> None:
    axis.set_ylim(0, 1)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#d8dee9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_pressure_outcome_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None

    pressure_order = ["none", "low", "high"]
    outcome_map = {
        "controlled_final_third_entry": "Controlled final-third entry",
        "shot_created": "Shot created",
        "lost_within_10s": "Lost within 10s",
        "opponent_shot_after_loss_10s": "Opponent shot after loss",
    }
    available = [column for column in outcome_map if column in sequence_dataset.columns]
    grouped = (
        sequence_dataset.groupby("pressure_level")[available]
        .mean()
        .reindex(pressure_order)
        .dropna(how="all")
    )
    if grouped.empty:
        return None
    counts = sequence_dataset.groupby("pressure_level").size().reindex(grouped.index)

    def wilson_interval(proportion: float, sample_size: int) -> tuple[float, float]:
        if sample_size <= 0:
            return float("nan"), float("nan")
        z_value = 1.96
        denominator = 1 + z_value**2 / sample_size
        center = (proportion + z_value**2 / (2 * sample_size)) / denominator
        radius = z_value * math.sqrt(
            proportion * (1 - proportion) / sample_size
            + z_value**2 / (4 * sample_size**2)
        ) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)

    fig, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(grouped.index))
    width = 0.18
    colors = ["#2f6f9f", "#2f9f6f", "#d9822b", "#b33f62"]

    for offset, column in enumerate(available):
        positions = x + (offset - (len(available) - 1) / 2) * width
        values = grouped[column]
        intervals = [
            wilson_interval(float(value), int(counts.iloc[index]))
            for index, value in enumerate(values)
        ]
        errors = np.array([
            [float(value) - interval[0] for value, interval in zip(values, intervals)],
            [interval[1] - float(value) for value, interval in zip(values, intervals)],
        ])
        axis.bar(
            positions,
            grouped[column],
            width=width,
            label=outcome_map[column],
            color=colors[offset],
        )
        axis.errorbar(positions, values, yerr=errors, fmt="none", ecolor="#344054", capsize=2, linewidth=0.8)

    axis.set_title("Goalkeeper build-up outcomes by opponent pressure")
    axis.set_xlabel("Pressure level")
    axis.set_xticks(x)
    axis.set_xticklabels([
        f"{pressure_level_display_label(str(value))}\n(n={int(counts.loc[value])})"
        for value in grouped.index
    ])
    clean_rate_axis(axis, ylabel="Sequence outcome rate")
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()

    output = FIGURE_DIR / "01_pressure_outcome_rates.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_study_flow_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None

    counts = [
        ("StatsBomb Open Data", f"{TARGET_COMPETITION_NAME} {TARGET_SEASON_NAME}"),
        ("Match sample", f"{sequence_dataset['match_id'].nunique():,} matches | {sequence_dataset['team'].nunique():,} teams"),
        ("Goalkeeper distribution sequences", f"{len(sequence_dataset):,} sequences with a goalkeeper pass"),
        ("Initial network window", f"Goalkeeper distribution plus first {INITIAL_BUILDUP_ACTIONS - 1} subsequent team actions"),
        ("Outcome-censored network", "Stop after first controlled progression, loss, or shot"),
        ("Multilevel analysis", "Team, match, goalkeeper, and pressure-layer random intercepts"),
    ]
    fig, axis = plt.subplots(figsize=(8.5, 10.5))
    axis.axis("off")
    y_positions = np.linspace(0.88, 0.12, len(counts))
    for index, ((title, detail), y_position) in enumerate(zip(counts, y_positions, strict=False)):
        box = plt.Rectangle(
            (0.12, y_position - 0.055),
            0.76,
            0.11,
            facecolor="#f3f6f8",
            edgecolor="#2f6f9f",
            linewidth=1.6,
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(0.5, y_position + 0.018, title, ha="center", va="center", fontsize=11, weight="bold", transform=axis.transAxes)
        axis.text(0.5, y_position - 0.022, detail, ha="center", va="center", fontsize=9, transform=axis.transAxes)
        if index < len(counts) - 1:
            axis.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.065),
                xytext=(0.5, y_position - 0.065),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#2f3a45"},
                xycoords=axis.transAxes,
                textcoords=axis.transAxes,
            )
    axis.set_title("Study design and analysis flow", fontsize=14, weight="bold", pad=18)
    output = FIGURE_DIR / "00_study_design_flow.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_sequence_definition_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None
    criteria = [
        ("Goalkeeper starts possession", int(sequence_dataset["gk_starts_sequence"].sum())),
        (f"Goalkeeper in first {EARLY_PASS_LIMIT} passes", int(sequence_dataset["gk_early_pass"].sum())),
        ("Goal kick sequence", int(sequence_dataset["from_goal_kick"].sum())),
        ("Defensive start", int(sequence_dataset["defensive_start"].sum())),
    ]
    labels = [item[0] for item in criteria]
    values = [item[1] for item in criteria]

    fig, axis = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    axis.barh(y, values, color=["#2f6f9f", "#426a5a", "#d9822b", "#7b5ea7"])
    axis.set_title("Sequence inclusion criteria observed in the dataset")
    axis.set_xlabel("Sequences meeting criterion")
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.grid(axis="x", color="#d8dee9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for index, value in enumerate(values):
        axis.text(value, index, f" {value:,}", va="center", fontsize=9)
    fig.tight_layout()

    output = FIGURE_DIR / "06_sequence_inclusion_criteria.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def completed_sequence_passes(
    sequence_rows: pd.DataFrame,
    completed_only: bool = True,
) -> pd.DataFrame:
    if sequence_rows.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for match_id, match_sequences in sequence_rows.groupby("match_id"):
        events = prepare_event_time(load_match_events(int(match_id)))
        lineups = load_match_lineups(int(match_id))
        positions = player_position_lookup(events)
        for sequence in match_sequences.itertuples(index=False):
            lineup_df = lineups.get(str(sequence.team))
            display_name_lookup: dict[int, str] = {}
            if lineup_df is not None and not lineup_df.empty:
                for lineup_row in lineup_df.itertuples(index=False):
                    player_id = getattr(lineup_row, "player_id", None)
                    if player_id is not None and not pd.isna(player_id):
                        display_name_lookup[int(player_id)] = display_player_name(
                            lineup_row,
                        )
            distribution_index = int(sequence.first_gk_distribution_event_index)
            cutoff_index = int(sequence.network_cutoff_event_index)
            pass_mask = (
                events["possession"].eq(int(sequence.possession))
                & events["team"].eq(str(sequence.team))
                & events["type"].eq("Pass")
                & events["player_id"].notna()
                & events["_event_index"].ge(distribution_index)
                & events["_event_index"].le(cutoff_index)
            )
            if completed_only:
                pass_mask &= (
                    events["pass_outcome"].isna()
                    & events["pass_recipient_id"].notna()
                )
            passes = events.loc[pass_mask].copy()
            if passes.empty:
                continue
            passes["buildup_sequence_id"] = sequence.buildup_sequence_id
            passes["goalkeeper_id"] = int(sequence.goalkeeper_id)
            passes["goalkeeper"] = str(sequence.goalkeeper)
            passes["pressure_level"] = str(sequence.pressure_level)
            passes["defensive_start"] = int(sequence.defensive_start)
            passes["defensive_third_exit"] = sequence.defensive_third_exit
            passes["controlled_final_third_entry"] = int(
                sequence.controlled_final_third_entry
            )
            passes["time_layer"] = str(sequence.time_layer)
            passes["player_position"] = passes["player_id"].map(
                lambda player_id: positions.get(int(player_id), "")
            )
            passes["pass_recipient_position"] = passes["pass_recipient_id"].map(
                lambda player_id: positions.get(int(player_id), "")
                if not pd.isna(player_id)
                else ""
            )
            passes["player_display_name"] = passes["player_id"].map(
                lambda player_id: display_name_lookup.get(
                    int(player_id),
                    str(passes.loc[passes["player_id"].eq(player_id), "player"].iloc[0]),
                )
            )
            passes["pass_recipient_display_name"] = passes[
                "pass_recipient_id"
            ].map(
                lambda player_id: (
                    display_name_lookup.get(
                        int(player_id),
                        str(
                            passes.loc[
                                passes["pass_recipient_id"].eq(player_id),
                                "pass_recipient",
                            ].iloc[0]
                        ),
                    )
                    if not pd.isna(player_id)
                    else "Unknown recipient"
                )
            )
            frames.append(passes)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["player_id"] = result["player_id"].astype(int)
    result["pass_recipient_id"] = result["pass_recipient_id"].astype("Int64")
    result["source_x"] = result["location"].map(location_x)
    result["source_y"] = result["location"].map(location_y)
    result["target_x"] = result["pass_end_location"].map(location_x)
    result["target_y"] = result["pass_end_location"].map(location_y)
    return result


def aggregate_player_network(passes: pd.DataFrame) -> tuple[nx.DiGraph, dict[int, tuple[float, float]], dict[int, str]]:
    graph = nx.DiGraph()
    positions: dict[int, tuple[float, float]] = {}
    labels: dict[int, str] = {}

    if passes.empty:
        return graph, positions, labels

    edge_table = (
        passes.groupby(["player_id", "pass_recipient_id"], as_index=False)
        .agg(
            weight=("id", "size"),
            source_name=("player_display_name", "first"),
            target_name=("pass_recipient_display_name", "first"),
        )
        .sort_values("weight", ascending=False)
    )
    for edge in edge_table.itertuples(index=False):
        graph.add_edge(int(edge.player_id), int(edge.pass_recipient_id), weight=int(edge.weight))
        labels[int(edge.player_id)] = str(edge.source_name)
        labels[int(edge.pass_recipient_id)] = str(edge.target_name)

    source_positions = passes.groupby("player_id").agg(x=("source_x", "mean"), y=("source_y", "mean"))
    target_positions = passes.groupby("pass_recipient_id").agg(x=("target_x", "mean"), y=("target_y", "mean"))
    for node in graph.nodes:
        candidate_positions: list[tuple[float, float]] = []
        if node in source_positions.index:
            row = source_positions.loc[node]
            candidate_positions.append((float(row.x), float(row.y)))
        if node in target_positions.index:
            row = target_positions.loc[node]
            candidate_positions.append((float(row.x), float(row.y)))
        valid = [(x, y) for x, y in candidate_positions if not pd.isna(x) and not pd.isna(y)]
        if valid:
            positions[int(node)] = (float(np.mean([x for x, _ in valid])), float(np.mean([y for _, y in valid])))
        else:
            positions[int(node)] = (60.0, 40.0)

    return graph, positions, labels


def short_player_label(name: str) -> str:
    parts = [part for part in str(name).replace("-", " ").split() if part]
    if not parts:
        return "Unknown"
    if len(parts) == 1:
        return parts[0][:12]
    if parts[0].endswith("."):
        return f"{parts[0]} {parts[1]}"[:14]
    return f"{parts[0][0]}. {parts[1]}"[:14]


def safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:90] if text else "unknown"


def draw_pitch_background(axis: plt.Axes) -> None:
    axis.set_xlim(0, 120)
    axis.set_ylim(0, 80)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.add_patch(plt.Rectangle((0, 0), 120, 80, facecolor="#edf3ee", edgecolor="#7a8b7d", linewidth=1.2))
    axis.plot([60, 60], [0, 80], color="#9aa89d", linewidth=1.0)
    axis.plot([40, 40], [0, 80], color="#c3cec6", linewidth=0.8, linestyle="--")
    axis.plot([80, 80], [0, 80], color="#c3cec6", linewidth=0.8, linestyle="--")
    axis.add_patch(plt.Rectangle((0, 18), 18, 44, fill=False, edgecolor="#9aa89d", linewidth=1.0))
    axis.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, edgecolor="#9aa89d", linewidth=1.0))


def draw_player_network(
    axis: plt.Axes,
    passes: pd.DataFrame,
    goalkeeper_id: int,
    title: str,
    top_edges: int = 24,
) -> None:
    draw_pitch_background(axis)
    graph, positions, labels = aggregate_player_network(passes)
    axis.set_title(title, fontsize=11, weight="bold")
    if graph.number_of_edges() == 0:
        axis.text(60, 40, "No completed passes", ha="center", va="center", fontsize=11)
        return

    edge_weights = nx.get_edge_attributes(graph, "weight")
    selected_edges = sorted(edge_weights, key=edge_weights.get, reverse=True)[:top_edges]
    max_weight = max(edge_weights.values()) if edge_weights else 1
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(graph.nodes)
    for edge in selected_edges:
        subgraph.add_edge(edge[0], edge[1], weight=edge_weights[edge])

    strengths = {
        node: graph.in_degree(node, weight="weight") + graph.out_degree(node, weight="weight")
        for node in graph.nodes
    }
    node_sizes = [420 + 130 * math.sqrt(float(strengths.get(node, 0))) for node in subgraph.nodes]
    node_colors = ["#b33f62" if int(node) == int(goalkeeper_id) else "#f7fbff" for node in subgraph.nodes]
    node_edges = ["#7a1235" if int(node) == int(goalkeeper_id) else "#2f6f9f" for node in subgraph.nodes]
    widths = [0.8 + 5.8 * edge_weights[edge] / max_weight for edge in selected_edges]

    nx.draw_networkx_edges(
        subgraph,
        positions,
        edgelist=selected_edges,
        width=widths,
        edge_color="#315f72",
        alpha=0.72,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        ax=axis,
    )
    nx.draw_networkx_nodes(
        subgraph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_edges,
        linewidths=1.4,
        ax=axis,
    )
    plot_labels = {node: short_player_label(labels.get(int(node), str(node))) for node in subgraph.nodes}
    nx.draw_networkx_labels(subgraph, positions, labels=plot_labels, font_size=7, ax=axis)
    axis.text(2, 3, f"Sequences: {passes['buildup_sequence_id'].nunique():,} | Passes: {len(passes):,}", fontsize=8, color="#344054")


def edge_rates_with_stability(passes: pd.DataFrame) -> pd.DataFrame:
    if passes.empty:
        return pd.DataFrame()
    sequence_ids = passes["buildup_sequence_id"].drop_duplicates().to_numpy()
    sequence_count = len(sequence_ids)
    edge_table = (
        passes.groupby(["player_id", "pass_recipient_id"], as_index=False)
        .agg(
            pass_count=("id", "size"),
            edge_sequences=("buildup_sequence_id", "nunique"),
        )
    )
    edge_table["edge_rate_per_sequence"] = edge_table["pass_count"] / sequence_count
    random_generator = np.random.default_rng(42)
    edge_sequence_sets = {
        (int(source), int(target)): set(group["buildup_sequence_id"])
        for (source, target), group in passes.groupby(["player_id", "pass_recipient_id"])
    }
    stability: list[float] = []
    for source, target in edge_table[["player_id", "pass_recipient_id"]].itertuples(index=False):
        edge_sequences = edge_sequence_sets[(int(source), int(target))]
        retained = 0
        for _ in range(EDGE_BOOTSTRAP_ITERATIONS):
            sample = random_generator.choice(sequence_ids, size=sequence_count, replace=True)
            if any(sequence_id in edge_sequences for sequence_id in sample):
                retained += 1
        stability.append(retained / EDGE_BOOTSTRAP_ITERATIONS)
    edge_table["bootstrap_stability"] = stability
    return edge_table.sort_values("edge_rate_per_sequence", ascending=False)


def pooled_player_layout(passes: pd.DataFrame) -> tuple[dict[int, tuple[float, float]], dict[int, str]]:
    _, positions, labels = aggregate_player_network(passes)
    return positions, labels


def draw_standardized_player_network(
    axis: plt.Axes,
    passes: pd.DataFrame,
    goalkeeper_id: int,
    title: str,
    fixed_positions: dict[int, tuple[float, float]],
    fixed_labels: dict[int, str],
    global_edge_rate: float,
    global_node_rate: float,
    top_edges: int = 20,
) -> None:
    draw_pitch_background(axis)
    axis.set_title(title, fontsize=11, weight="bold")
    if passes.empty:
        axis.text(60, 40, "No completed pre-outcome passes", ha="center", va="center", fontsize=10)
        return

    edge_table = edge_rates_with_stability(passes)
    stable_edges = edge_table.loc[
        edge_table["bootstrap_stability"].ge(EDGE_STABILITY_THRESHOLD)
    ].copy()
    if stable_edges.empty:
        stable_edges = edge_table.head(top_edges).copy()
    stable_edges = stable_edges.head(top_edges)

    sequence_count = int(passes["buildup_sequence_id"].nunique())
    node_strength = pd.concat(
        [
            passes.groupby("player_id").size().rename("count").rename_axis("node").reset_index(),
            passes.groupby("pass_recipient_id").size().rename("count").rename_axis("node").reset_index(),
        ],
        ignore_index=True,
    ).groupby("node", as_index=False)["count"].sum()
    node_strength["rate"] = node_strength["count"] / sequence_count

    graph = nx.DiGraph()
    for edge in stable_edges.itertuples(index=False):
        graph.add_edge(int(edge.player_id), int(edge.pass_recipient_id), rate=float(edge.edge_rate_per_sequence))
    graph.add_nodes_from(node_strength["node"].astype(int))
    positions = {node: fixed_positions.get(int(node), (60.0, 40.0)) for node in graph.nodes}
    node_rate_lookup = node_strength.set_index("node")["rate"].to_dict()
    node_sizes = [360 + 900 * float(node_rate_lookup.get(node, 0.0)) / max(global_node_rate, 1e-9) for node in graph.nodes]
    widths = [0.8 + 6.0 * float(graph.edges[edge]["rate"]) / max(global_edge_rate, 1e-9) for edge in graph.edges]
    node_colors = ["#b33f62" if int(node) == int(goalkeeper_id) else "#f7fbff" for node in graph.nodes]
    node_edges = ["#7a1235" if int(node) == int(goalkeeper_id) else "#2f6f9f" for node in graph.nodes]

    nx.draw_networkx_edges(
        graph,
        positions,
        width=widths,
        edge_color="#315f72",
        alpha=0.78,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_edges,
        linewidths=1.3,
        ax=axis,
    )
    label_nodes = set(graph.nodes)
    labels = {node: short_player_label(fixed_labels.get(int(node), str(node))) for node in label_nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=axis)
    axis.text(
        2,
        3,
        f"Sequences: {sequence_count:,} | edge width: passes per sequence | stable edges: >= {EDGE_STABILITY_THRESHOLD:.0%}",
        fontsize=7.5,
        color="#344054",
    )


def position_role(position: str) -> str:
    text = str(position).lower()
    if "goalkeeper" in text:
        return "GK"
    if "center back" in text:
        return "CB"
    if any(token in text for token in ["left back", "right back", "wing back", "full back"]):
        return "FB"
    if "defensive midfield" in text:
        return "DM"
    if any(token in text for token in ["attacking midfield", "second striker"]):
        return "AM"
    if any(token in text for token in ["center midfield", "central midfield"]):
        return "CM"
    if any(token in text for token in ["winger", "forward", "striker"]):
        return "FW"
    return "Other"


ROLE_POSITIONS = {
    "GK": (8, 40),
    "CB": (30, 40),
    "FB": (35, 18),
    "DM": (48, 40),
    "CM": (62, 40),
    "AM": (78, 40),
    "FW": (98, 40),
    "Other": (58, 65),
}


def draw_role_network(
    axis: plt.Axes,
    passes: pd.DataFrame,
    title: str,
    global_edge_rate: float,
) -> None:
    draw_pitch_background(axis)
    axis.set_title(title, fontsize=11, weight="bold")
    if passes.empty:
        axis.text(60, 40, "No completed pre-outcome passes", ha="center", va="center")
        return
    role_passes = passes.copy()
    role_passes["source_role"] = role_passes["player_position"].map(position_role)
    role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
    sequence_count = int(role_passes["buildup_sequence_id"].nunique())
    edges = (
        role_passes.groupby(["source_role", "target_role"], as_index=False)
        .agg(pass_count=("id", "size"))
    )
    edges["rate"] = edges["pass_count"] / sequence_count
    graph = nx.DiGraph()
    for edge in edges.itertuples(index=False):
        graph.add_edge(edge.source_role, edge.target_role, rate=float(edge.rate))
    widths = [0.8 + 6.0 * graph.edges[edge]["rate"] / max(global_edge_rate, 1e-9) for edge in graph.edges]
    node_sizes = [1200 if node == "GK" else 900 for node in graph.nodes]
    node_colors = ["#b33f62" if node == "GK" else "#f7fbff" for node in graph.nodes]
    nx.draw_networkx_edges(graph, ROLE_POSITIONS, width=widths, edge_color="#315f72", arrows=True, arrowsize=16, ax=axis)
    nx.draw_networkx_nodes(graph, ROLE_POSITIONS, node_size=node_sizes, node_color=node_colors, edgecolors="#2f6f9f", linewidths=1.3, ax=axis)
    nx.draw_networkx_labels(graph, ROLE_POSITIONS, font_size=9, ax=axis)
    axis.text(2, 3, f"Sequences: {sequence_count:,} | edge width: passes per sequence", fontsize=8, color="#344054")


def select_pressure_comparison_target(sequence_dataset: pd.DataFrame) -> tuple[int, str, str] | None:
    counts = sequence_dataset.groupby(["goalkeeper_id", "goalkeeper", "team", "pressure_level"]).size().unstack(fill_value=0)
    if "low" not in counts.columns or "high" not in counts.columns:
        return None

    candidates = counts.loc[(counts["low"] >= 10) & (counts["high"] >= 10)].copy()
    if candidates.empty:
        fallback_counts = counts.copy()
        fallback_counts["total"] = fallback_counts[["low", "high"]].sum(axis=1)
        fallback_counts = fallback_counts.sort_values(["total", "low", "high"], ascending=False)
        if fallback_counts.empty:
            return None
        selected = fallback_counts.index[0]
        return int(selected[0]), str(selected[1]), str(selected[2])

    candidates["balance"] = candidates[["low", "high"]].min(axis=1) / candidates[["low", "high"]].max(axis=1)
    candidates = candidates.loc[candidates["balance"].ge(0.30)]
    if candidates.empty:
        candidates = counts.copy()
        candidates["total"] = candidates[["low", "high"]].sum(axis=1)
        candidates = candidates.sort_values(["total", "low", "high"], ascending=False)
        if candidates.empty:
            return None
        selected = candidates.index[0]
        return int(selected[0]), str(selected[1]), str(selected[2])

    candidates["total"] = candidates[["low", "high"]].sum(axis=1)
    selected = candidates.sort_values(["balance", "total"], ascending=False).index[0]
    return int(selected[0]), str(selected[1]), str(selected[2])


def select_success_failure_target(sequence_dataset: pd.DataFrame) -> tuple[int, str, str] | None:
    eligible = sequence_dataset.loc[
        sequence_dataset["defensive_start"].eq(True)
        & sequence_dataset["defensive_third_exit"].isin([0, 1])
    ]
    counts = eligible.groupby(["goalkeeper_id", "goalkeeper", "team", "defensive_third_exit"]).size().unstack(fill_value=0)
    if 0 not in counts.columns or 1 not in counts.columns:
        return None
    candidates = counts.loc[(counts[0] >= 15) & (counts[1] >= 15)].copy()
    if candidates.empty:
        return None
    candidates["balance"] = candidates[[0, 1]].min(axis=1) / candidates[[0, 1]].max(axis=1)
    candidates = candidates.loc[candidates["balance"].ge(0.30)]
    if candidates.empty:
        return None
    candidates["total"] = candidates[[0, 1]].sum(axis=1)
    selected = candidates.sort_values(["balance", "total"], ascending=False).index[0]
    return int(selected[0]), str(selected[1]), str(selected[2])


def save_pressure_level_player_networks(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None

    subset = sequence_dataset.loc[sequence_dataset["pressure_level"].isin(["low", "high"])].copy()
    passes = completed_sequence_passes(subset)
    if passes.empty:
        return None

    role_passes = passes.copy()
    role_passes["source_role"] = role_passes["player_position"].map(position_role)
    role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
    role_edges = (
        role_passes.groupby(["source_role", "target_role"], as_index=False)
        .agg(pass_count=("id", "size"))
    )
    role_edges["rate"] = role_edges["pass_count"] / max(passes["buildup_sequence_id"].nunique(), 1)
    global_edge_rate = float(role_edges["rate"].max()) if not role_edges.empty else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for axis, pressure in zip(axes, ["low", "high"], strict=False):
        pressure_passes = role_passes.loc[role_passes["pressure_level"].eq(pressure)]
        draw_role_network(axis, pressure_passes, pressure_level_display_label(pressure), global_edge_rate)

    fig.suptitle("Pooled-role build-up networks by observed anchor pressure", fontsize=14, weight="bold")
    fig.tight_layout()

    output = FIGURE_DIR / "10_low_high_pressure_player_networks.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_progression_success_failure_player_networks(sequence_dataset: pd.DataFrame) -> Path | None:
    selected = select_success_failure_target(sequence_dataset)
    if selected is None:
        subset = sequence_dataset.loc[
            sequence_dataset["defensive_start"].eq(True)
            & sequence_dataset["defensive_third_exit"].isin([0, 1])
        ].copy()
        goalkeeper_id = int(subset["goalkeeper_id"].mode().iloc[0])
        goalkeeper = "All eligible goalkeepers"
        team = "Pooled sample"
    else:
        goalkeeper_id, goalkeeper, team = selected
        subset = sequence_dataset.loc[
            sequence_dataset["goalkeeper_id"].eq(goalkeeper_id)
            & sequence_dataset["team"].eq(team)
            & sequence_dataset["defensive_start"].eq(True)
            & sequence_dataset["defensive_third_exit"].isin([0, 1])
        ].copy()
    passes = completed_sequence_passes(subset)
    if passes.empty:
        return None

    if selected is None:
        role_passes = passes.copy()
        role_passes["source_role"] = role_passes["player_position"].map(position_role)
        role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
        role_edges = (
            role_passes.groupby(["source_role", "target_role"], as_index=False)
            .agg(pass_count=("id", "size"))
        )
        role_rates = role_edges["pass_count"] / max(
            role_passes["buildup_sequence_id"].nunique(),
            1,
        )
        global_edge_rate = float(role_rates.max()) if not role_rates.empty else 1.0
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
        draw_role_network(
            axes[0],
            passes.loc[passes["defensive_third_exit"].eq(1)],
            "Pooled roles | Defensive-third exit success",
            global_edge_rate,
        )
        draw_role_network(
            axes[1],
            passes.loc[passes["defensive_third_exit"].eq(0)],
            "Pooled roles | Defensive-third exit failure",
            global_edge_rate,
        )
        fig.suptitle("Pooled role-network structure in progression success and failure", fontsize=14, weight="bold")
        fig.tight_layout()
        output = FIGURE_DIR / "Figure_4_success_versus_failure_role_network.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output

    fixed_positions, fixed_labels = pooled_player_layout(passes)
    pooled_edges = edge_rates_with_stability(passes)
    global_edge_rate = float(pooled_edges["edge_rate_per_sequence"].max()) if not pooled_edges.empty else 1.0
    pooled_node_rate = pd.concat([passes.groupby("player_id").size(), passes.groupby("pass_recipient_id").size()]).groupby(level=0).sum()
    global_node_rate = float((pooled_node_rate / passes["buildup_sequence_id"].nunique()).max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    draw_standardized_player_network(
        axes[0],
        passes.loc[passes["defensive_third_exit"].eq(1)],
        goalkeeper_id if selected is not None else -1,
        f"{goalkeeper} ({team}) | Defensive-third exit success",
        fixed_positions,
        fixed_labels,
        global_edge_rate,
        global_node_rate,
    )
    draw_standardized_player_network(
        axes[1],
        passes.loc[passes["defensive_third_exit"].eq(0)],
        goalkeeper_id if selected is not None else -1,
        f"{goalkeeper} ({team}) | Defensive-third exit failure",
        fixed_positions,
        fixed_labels,
        global_edge_rate,
        global_node_rate,
    )
    fig.suptitle("Build-up network structure in progression success and failure", fontsize=14, weight="bold")
    fig.tight_layout()

    output = FIGURE_DIR / "Figure_4_success_versus_failure_role_network.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_pressure_multilayer_player_network(sequence_dataset: pd.DataFrame) -> Path | None:
    selected = select_pressure_comparison_target(sequence_dataset)
    if selected is None:
        return None
    goalkeeper_id, goalkeeper, team = selected
    layers = ["none", "low", "high"]
    subset = sequence_dataset.loc[
        sequence_dataset["goalkeeper_id"].eq(goalkeeper_id)
        & sequence_dataset["team"].eq(team)
        & sequence_dataset["pressure_level"].isin(layers)
    ].copy()
    passes = completed_sequence_passes(subset)
    if passes.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for axis, pressure in zip(axes, layers, strict=False):
        pressure_passes = passes.loc[passes["pressure_level"].eq(pressure)]
        draw_player_network(axis, pressure_passes, goalkeeper_id, f"Layer: {pressure_level_display_label(pressure)}", top_edges=20)
    fig.suptitle(f"Pressure-stratified build-up networks: {goalkeeper} ({team})", fontsize=14, weight="bold")
    fig.tight_layout()

    output = FIGURE_DIR / "12_pressure_multilayer_player_network.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_pressure_level_role_networks_for_subset(
    sequence_dataset: pd.DataFrame,
    title: str,
    output_name: str,
) -> Path | None:
    if sequence_dataset.empty:
        return None

    filters = [
        pressure_level
        for pressure_level in ["none", "low", "high"]
        if pressure_level in set(sequence_dataset["pressure_level"].astype(str).str.lower())
    ]
    if not filters:
        return None

    subset = sequence_dataset.loc[sequence_dataset["pressure_level"].isin(filters)].copy()
    passes = completed_sequence_passes(subset)
    if passes.empty:
        return None

    role_passes = passes.copy()
    role_passes["source_role"] = role_passes["player_position"].map(position_role)
    role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
    role_edges = (
        role_passes.groupby(["source_role", "target_role"], as_index=False)
        .agg(pass_count=("id", "size"))
    )
    role_edges["rate"] = role_edges["pass_count"] / max(passes["buildup_sequence_id"].nunique(), 1)
    global_edge_rate = float(role_edges["rate"].max()) if not role_edges.empty else 1.0

    fig, axes = plt.subplots(1, len(filters), figsize=(6.0 * len(filters), 6.5))
    if len(filters) == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    for axis, pressure in zip(axes_list, filters, strict=False):
        pressure_passes = passes.loc[passes["pressure_level"].eq(pressure)]
        draw_role_network(axis, pressure_passes, pressure_level_display_label(pressure), global_edge_rate)

    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    output = FIGURE_DIR / output_name
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _filter_sequence_dataset_by_stage_context(sequence_dataset: pd.DataFrame, stage_context: str) -> pd.DataFrame:
    if sequence_dataset.empty:
        return sequence_dataset.iloc[0:0].copy()

    if stage_context == "overall":
        return sequence_dataset.copy()

    if "competition_stage" not in sequence_dataset.columns:
        return sequence_dataset.iloc[0:0].copy()

    stage_values = sequence_dataset["competition_stage"].fillna("").astype(str).str.strip()
    if stage_context == "group":
        return sequence_dataset.loc[stage_values.eq("Group Stage")].copy()
    if stage_context == "tournament":
        return sequence_dataset.loc[~stage_values.eq("Group Stage")].copy()
    return sequence_dataset.copy()


def save_pressure_level_role_networks(sequence_dataset: pd.DataFrame) -> Path | None:
    subset = _filter_sequence_dataset_by_stage_context(sequence_dataset, "overall")
    return _save_pressure_level_role_networks_for_subset(
        subset,
        "Pressure-layer build-up networks (pooled roles)",
        "12_pressure_level_role_networks.png",
    )


def save_group_stage_pressure_level_role_networks(sequence_dataset: pd.DataFrame) -> Path | None:
    subset = _filter_sequence_dataset_by_stage_context(sequence_dataset, "group")
    return _save_pressure_level_role_networks_for_subset(
        subset,
        "Group-stage pressure-layer build-up networks (pooled roles)",
        "12_group_stage_pressure_level_role_networks.png",
    )


def save_tournament_pressure_level_role_networks(sequence_dataset: pd.DataFrame) -> Path | None:
    subset = _filter_sequence_dataset_by_stage_context(sequence_dataset, "tournament")
    return _save_pressure_level_role_networks_for_subset(
        subset,
        "Tournament pressure-layer build-up networks (pooled roles)",
        "12_tournament_pressure_level_role_networks.png",
    )


def save_total_pressure_player_network(sequence_dataset: pd.DataFrame) -> Path | None:
    selected = select_pressure_comparison_target(sequence_dataset)
    if selected is None:
        return None
    goalkeeper_id, goalkeeper, team = selected

    subset = sequence_dataset.loc[
        sequence_dataset["goalkeeper_id"].eq(goalkeeper_id)
        & sequence_dataset["team"].eq(team)
        & sequence_dataset["pressure_level"].isin(["none", "low", "high"])
    ].copy()
    passes = completed_sequence_passes(subset)
    if passes.empty:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 6.5))
    draw_player_network(
        axis,
        passes,
        goalkeeper_id,
        f"Overall pressure network: {goalkeeper} ({team})",
        top_edges=30,
    )
    fig.tight_layout()

    output = FIGURE_DIR / "12_total_pressure_player_network.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_tournament_pressure_panel_network(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None

    layers = [
        ("overall", ["none", "low", "high"]),
        ("none", ["none"]),
        ("low", ["low"]),
        ("high", ["high"]),
    ]

    panel_data: list[tuple[str, pd.DataFrame]] = []
    for label, pressure_levels in layers:
        subset = sequence_dataset.loc[sequence_dataset["pressure_level"].isin(pressure_levels)].copy()
        passes = completed_sequence_passes(subset)
        if not passes.empty:
            panel_data.append((label, passes))

    if not panel_data:
        return None

    all_passes = completed_sequence_passes(sequence_dataset.loc[sequence_dataset["pressure_level"].isin(["none", "low", "high"])]).copy()
    role_passes = all_passes.copy()
    role_passes["source_role"] = role_passes["player_position"].map(position_role)
    role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
    all_edges = (
        role_passes.groupby(["source_role", "target_role"], as_index=False)
        .agg(pass_count=("id", "size"))
    )
    all_edges["rate"] = all_edges["pass_count"] / max(all_passes["buildup_sequence_id"].nunique(), 1)
    global_edge_rate = float(all_edges["rate"].max()) if not all_edges.empty else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    titles = {
        "overall": "Overall tournament pooled-role network",
        "none": "No pressure",
        "low": "Low pressure",
        "high": "Pressure on all observed anchor actions",
    }
    for axis, (label, passes) in zip(axes.ravel(), panel_data, strict=False):
        role_passes = passes.copy()
        role_passes["source_role"] = role_passes["player_position"].map(position_role)
        role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
        draw_role_network(axis, role_passes, titles[label], global_edge_rate)

    if len(panel_data) < 4:
        for axis in axes.ravel()[len(panel_data):]:
            axis.axis("off")
            axis.text(60, 40, "No completed passes", ha="center", va="center", fontsize=11)

    fig.suptitle("Tournament-wide pooled-role build-up networks by pressure level", fontsize=14, weight="bold")
    fig.tight_layout()

    output = FIGURE_DIR / "12_tournament_pressure_panel_network.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


r'''Temporal multilayer implementation removed from the executable analysis.

The archived source below is intentionally inert and can be moved to a separate
longitudinal follow-up project without affecting this paper's analysis.

def build_temporal_multilayer_metrics(sequence_dataset: pd.DataFrame) -> pd.DataFrame:
    if sequence_dataset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = ["goalkeeper_id", "goalkeeper", "team"]
    for group_key, group in sequence_dataset.groupby(group_columns, dropna=False):
        goalkeeper_id, goalkeeper, team = group_key
        passes = completed_sequence_passes(group)
        if passes.empty:
            continue
        previous_edges: set[tuple[int, int]] | None = None
        previous_centrality: dict[int, float] | None = None
        previous_gk_receivers: set[int] | None = None
        for layer in TIME_LAYER_LABELS:
            layer_passes = passes.loc[passes["time_layer"].eq(layer)]
            graph, _, _ = aggregate_player_network(layer_passes)
            edge_set = set((int(source), int(target)) for source, target in graph.edges)
            centrality = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_edges() else {}
            gk_receivers = set(
                layer_passes.loc[layer_passes["player_id"].eq(goalkeeper_id), "pass_recipient_id"].astype(int)
            )
            if previous_edges is None:
                edge_persistence = float("nan")
                centrality_volatility = float("nan")
                receiver_switching_rate = float("nan")
            else:
                union = previous_edges | edge_set
                edge_persistence = safe_divide(len(previous_edges & edge_set), len(union))
                shared_nodes = set(previous_centrality or {}) | set(centrality)
                centrality_volatility = (
                    float(np.mean([
                        abs((previous_centrality or {}).get(node, 0.0) - centrality.get(node, 0.0))
                        for node in shared_nodes
                    ]))
                    if shared_nodes
                    else float("nan")
                )
                receiver_union = (previous_gk_receivers or set()) | gk_receivers
                receiver_switching_rate = 1.0 - safe_divide(
                    len((previous_gk_receivers or set()) & gk_receivers),
                    len(receiver_union),
                ) if receiver_union else float("nan")
            rows.append(
                {
                    "goalkeeper_id": int(goalkeeper_id),
                    "goalkeeper": goalkeeper,
                    "team": team,
                    "time_layer": layer,
                    "sequences": int(layer_passes["buildup_sequence_id"].nunique()),
                    "network_nodes": int(graph.number_of_nodes()),
                    "network_edges": int(graph.number_of_edges()),
                    "edge_persistence_vs_previous_layer": edge_persistence,
                    "centrality_volatility_vs_previous_layer": centrality_volatility,
                    "receiver_switching_rate_vs_previous_layer": receiver_switching_rate,
                }
            )
            previous_edges = edge_set
            previous_centrality = centrality
            previous_gk_receivers = gk_receivers
    return pd.DataFrame(rows)


def draw_temporal_multilayer_network(
    axis: plt.Axes,
    passes: pd.DataFrame,
    goalkeeper_id: int,
    goalkeeper: str,
    team: str,
    compact: bool = False,
) -> None:
    axis.axis("off")
    pooled_positions, pooled_labels = pooled_player_layout(passes)
    layer_width = 125.0 if not compact else 95.0
    node_size_gk = 210 if not compact else 80
    node_size_player = 105 if not compact else 36
    label_size = 5.5 if not compact else 0
    layer_graphs: dict[str, nx.DiGraph] = {}
    layer_positions: dict[tuple[int, str], tuple[float, float]] = {}
    for layer_index, layer in enumerate(TIME_LAYER_LABELS):
        layer_passes = passes.loc[passes["time_layer"].eq(layer)]
        graph, _, _ = aggregate_player_network(layer_passes)
        layer_graphs[layer] = graph
        x_offset = layer_index * layer_width
        axis.add_patch(plt.Rectangle((x_offset, 0), 120, 80, facecolor="#edf3ee", edgecolor="#7a8b7d", linewidth=0.7 if compact else 1.0))
        axis.text(x_offset + 60, 84, layer, ha="center", va="bottom", fontsize=7 if compact else 11, weight="bold")
        for node in graph.nodes:
            base_x, base_y = pooled_positions.get(int(node), (60.0, 40.0))
            layer_positions[(int(node), layer)] = (x_offset + base_x, base_y)
        if graph.number_of_edges():
            edge_rates = edge_rates_with_stability(layer_passes).head(8 if compact else 12)
            for edge in edge_rates.itertuples(index=False):
                source_key = (int(edge.player_id), layer)
                target_key = (int(edge.pass_recipient_id), layer)
                if source_key not in layer_positions or target_key not in layer_positions:
                    continue
                axis.annotate(
                    "",
                    xy=layer_positions[target_key],
                    xytext=layer_positions[source_key],
                    arrowprops={
                        "arrowstyle": "->",
                        "lw": 0.5 + 4.2 * float(edge.edge_rate_per_sequence),
                        "color": "#315f72",
                        "alpha": 0.68,
                    },
                )
        for node in graph.nodes:
            x_position, y_position = layer_positions[(int(node), layer)]
            is_goalkeeper = int(node) == int(goalkeeper_id)
            axis.scatter(
                x_position,
                y_position,
                s=node_size_gk if is_goalkeeper else node_size_player,
                marker="s" if is_goalkeeper else "o",
                c="#b33f62" if is_goalkeeper else "#f7fbff",
                edgecolors="#7a1235" if is_goalkeeper else "#2f6f9f",
                linewidths=0.8,
                zorder=3,
            )
            if label_size > 0:
                axis.text(x_position, y_position - 4, short_player_label(pooled_labels.get(int(node), str(node))), ha="center", va="top", fontsize=label_size)

    for current_index in range(len(TIME_LAYER_LABELS) - 1):
        current_layer = TIME_LAYER_LABELS[current_index]
        next_layer = TIME_LAYER_LABELS[current_index + 1]
        shared_nodes = set(layer_graphs[current_layer].nodes) & set(layer_graphs[next_layer].nodes)
        for node in shared_nodes:
            source_position = layer_positions[(int(node), current_layer)]
            target_position = layer_positions[(int(node), next_layer)]
            axis.plot(
                [source_position[0], target_position[0]],
                [source_position[1], target_position[1]],
                color="#8f99a3",
                linewidth=0.35 if compact else 0.55,
                linestyle="--",
                alpha=0.5,
                zorder=1,
            )
    axis.set_xlim(-5, layer_width * len(TIME_LAYER_LABELS) - 5)
    axis.set_ylim(-8, 91)
    axis.set_title(
        f"{goalkeeper} ({team})",
        fontsize=10 if compact else 15,
        weight="bold",
    )


def save_true_temporal_multilayer_network(sequence_dataset: pd.DataFrame) -> Path | None:
    temporal_metrics = build_temporal_multilayer_metrics(sequence_dataset)
    if temporal_metrics.empty:
        return None

    candidate_metrics = (
        temporal_metrics.groupby(["goalkeeper_id", "goalkeeper", "team"], as_index=False)
        .agg(
            sequences=("sequences", "sum"),
            active_layers=("sequences", lambda values: int((values > 0).sum())),
            mean_edge_persistence=("edge_persistence_vs_previous_layer", "mean"),
            mean_centrality_volatility=("centrality_volatility_vs_previous_layer", "mean"),
            mean_receiver_switching=("receiver_switching_rate_vs_previous_layer", "mean"),
        )
    )
    eligible = candidate_metrics.loc[
        candidate_metrics["sequences"].ge(TEMPORAL_CASE_MIN_SEQUENCES)
        & candidate_metrics["active_layers"].ge(TEMPORAL_CASE_MIN_ACTIVE_LAYERS)
    ].copy()
    if eligible.empty:
        return None

    eligible["temporal_change_score"] = (
        eligible["mean_receiver_switching"].fillna(0.0)
        + eligible["mean_centrality_volatility"].fillna(0.0)
        - eligible["mean_edge_persistence"].fillna(0.0)
    )

    stable_cases = (
        eligible.sort_values(
            ["mean_edge_persistence", "mean_receiver_switching", "sequences"],
            ascending=[False, True, False],
        )
        .head(TEMPORAL_CASES_PER_CONTRAST)
        .copy()
    )
    stable_cases["selection_group"] = "stable temporal structure"
    stable_cases["selection_score"] = stable_cases["mean_edge_persistence"]

    selected_ids = set(stable_cases["goalkeeper_id"].astype(int))
    adaptive_cases = (
        eligible.loc[~eligible["goalkeeper_id"].astype(int).isin(selected_ids)]
        .sort_values(
            ["temporal_change_score", "sequences"],
            ascending=[False, False],
        )
        .head(TEMPORAL_CASES_PER_CONTRAST)
        .copy()
    )
    adaptive_cases["selection_group"] = "adaptive receiver switching"
    adaptive_cases["selection_score"] = adaptive_cases["temporal_change_score"]

    candidates = pd.concat([stable_cases, adaptive_cases], ignore_index=True)
    if len(candidates) < TEMPORAL_CASES_PER_CONTRAST * 2:
        fill_cases = (
            eligible.loc[~eligible["goalkeeper_id"].astype(int).isin(candidates["goalkeeper_id"].astype(int))]
            .sort_values("sequences", ascending=False)
            .head(TEMPORAL_CASES_PER_CONTRAST * 2 - len(candidates))
            .copy()
        )
        fill_cases["selection_group"] = "largest remaining eligible sample"
        fill_cases["selection_score"] = fill_cases["sequences"]
        candidates = pd.concat([candidates, fill_cases], ignore_index=True)
    if candidates.empty:
        return None

    candidates["selection_rule"] = (
        f"eligible if >= {TEMPORAL_CASE_MIN_SEQUENCES} sequences and >= "
        f"{TEMPORAL_CASE_MIN_ACTIVE_LAYERS} active time layers; select "
        f"{TEMPORAL_CASES_PER_CONTRAST} highest edge-persistence stable cases and "
        f"{TEMPORAL_CASES_PER_CONTRAST} highest temporal-change adaptive cases"
    )

    output_dir = FIGURE_DIR / "temporal_multilayer_goalkeepers"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    first_output: Path | None = None

    for rank, row in enumerate(candidates.itertuples(index=False), start=1):
        goalkeeper_id = int(row.goalkeeper_id)
        goalkeeper = str(row.goalkeeper)
        team = str(row.team)
        subset = sequence_dataset.loc[
            sequence_dataset["goalkeeper_id"].eq(goalkeeper_id)
            & sequence_dataset["team"].eq(team)
        ]
        passes = completed_sequence_passes(subset)
        if passes.empty:
            continue

        pooled_positions, pooled_labels = pooled_player_layout(passes)
        fig, axis = plt.subplots(figsize=(19, 8))
        axis.axis("off")
        layer_width = 125.0
        layer_graphs: dict[str, nx.DiGraph] = {}
        layer_positions: dict[tuple[int, str], tuple[float, float]] = {}
        for layer_index, layer in enumerate(TIME_LAYER_LABELS):
            layer_passes = passes.loc[passes["time_layer"].eq(layer)]
            graph, _, _ = aggregate_player_network(layer_passes)
            layer_graphs[layer] = graph
            x_offset = layer_index * layer_width
            axis.add_patch(plt.Rectangle((x_offset, 0), 120, 80, facecolor="#edf3ee", edgecolor="#7a8b7d", linewidth=1.0))
            axis.text(x_offset + 60, 84, layer, ha="center", va="bottom", fontsize=11, weight="bold")
            for node in graph.nodes:
                base_x, base_y = pooled_positions.get(int(node), (60.0, 40.0))
                layer_positions[(int(node), layer)] = (x_offset + base_x, base_y)
            if graph.number_of_edges():
                edge_rates = edge_rates_with_stability(layer_passes).head(12)
                for edge in edge_rates.itertuples(index=False):
                    source_key = (int(edge.player_id), layer)
                    target_key = (int(edge.pass_recipient_id), layer)
                    if source_key not in layer_positions or target_key not in layer_positions:
                        continue
                    source_position = layer_positions[source_key]
                    target_position = layer_positions[target_key]
                    axis.annotate(
                        "",
                        xy=target_position,
                        xytext=source_position,
                        arrowprops={
                            "arrowstyle": "->",
                            "lw": 0.8 + 5 * float(edge.edge_rate_per_sequence),
                            "color": "#315f72",
                            "alpha": 0.75,
                        },
                    )
            for node in graph.nodes:
                x_position, y_position = layer_positions[(int(node), layer)]
                is_goalkeeper = int(node) == int(goalkeeper_id)
                axis.scatter(
                    x_position,
                    y_position,
                    s=210 if is_goalkeeper else 105,
                    marker="s" if is_goalkeeper else "o",
                    c="#b33f62" if is_goalkeeper else "#f7fbff",
                    edgecolors="#7a1235" if is_goalkeeper else "#2f6f9f",
                    linewidths=1.0,
                    zorder=3,
                )
                axis.text(x_position, y_position - 4, short_player_label(pooled_labels.get(int(node), str(node))), ha="center", va="top", fontsize=5.5)

        for current_index in range(len(TIME_LAYER_LABELS) - 1):
            current_layer = TIME_LAYER_LABELS[current_index]
            next_layer = TIME_LAYER_LABELS[current_index + 1]
            shared_nodes = set(layer_graphs[current_layer].nodes) & set(layer_graphs[next_layer].nodes)
            for node in shared_nodes:
                source_position = layer_positions[(int(node), current_layer)]
                target_position = layer_positions[(int(node), next_layer)]
                axis.plot(
                    [source_position[0], target_position[0]],
                    [source_position[1], target_position[1]],
                    color="#8f99a3",
                    linewidth=0.55,
                    linestyle="--",
                    alpha=0.55,
                    zorder=1,
                )
        axis.set_xlim(-5, layer_width * len(TIME_LAYER_LABELS) - 5)
        axis.set_ylim(-8, 91)
        axis.set_title(
            f"Tournament-aggregated temporal multilayer network: {goalkeeper} ({team})",
            fontsize=15,
            weight="bold",
        )
        axis.text(0, -6, "Solid arrows: within-layer completed pre-outcome passes. Dashed lines: same player across adjacent time layers.", fontsize=9, color="#4d5966")
        fig.tight_layout()
        output = output_dir / f"13_{rank:02d}_{safe_filename(str(team))}_{safe_filename(str(goalkeeper))}_{int(goalkeeper_id)}_temporal_multilayer.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        first_output = first_output or output
        manifest_rows.append(
            {
                "rank": rank,
                "goalkeeper_id": int(goalkeeper_id),
                "goalkeeper": str(goalkeeper),
                "team": str(team),
                "selection_group": str(row.selection_group),
                "selection_score": float(row.selection_score),
                "selection_rule": str(row.selection_rule),
                "sequences": int(row.sequences),
                "active_layers": int(row.active_layers),
                "mean_edge_persistence": float(row.mean_edge_persistence),
                "mean_receiver_switching": float(row.mean_receiver_switching),
                "mean_centrality_volatility": float(row.mean_centrality_volatility),
                "temporal_change_score": float(row.temporal_change_score),
                "figure_path": str(output.relative_to(OUTPUT_ROOT)),
            }
        )

    if not manifest_rows:
        return None

    pd.DataFrame(manifest_rows).to_csv(
        TABLE_DIR / "figure_13_temporal_multilayer_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for axis, row in zip(axes.ravel(), manifest_rows, strict=False):
        subset = sequence_dataset.loc[
            sequence_dataset["goalkeeper_id"].eq(row["goalkeeper_id"])
            & sequence_dataset["team"].eq(row["team"])
        ]
        passes = completed_sequence_passes(subset)
        draw_temporal_multilayer_network(
            axis,
            passes,
            row["goalkeeper_id"],
            row["goalkeeper"],
            row["team"],
            compact=True,
        )
    fig.suptitle("Tournament-aggregated temporal multilayer networks: stable vs adaptive goalkeeper cases", fontsize=14, weight="bold")
    fig.tight_layout()
    summary_output = FIGURE_DIR / "13_true_temporal_multilayer_network.png"
    fig.savefig(summary_output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary_output


'''


def save_goalkeeper_pressure_role_networks(
    goalkeeper_sequences: pd.DataFrame,
    goalkeeper_id: int,
    goalkeeper: str,
    team: str,
    title_prefix: str,
    output_path: Path,
) -> Path | None:
    if goalkeeper_sequences.empty:
        return None

    subset = goalkeeper_sequences.loc[goalkeeper_sequences["pressure_level"].isin(["low", "high"])].copy()
    passes = completed_sequence_passes(subset)
    if passes.empty:
        return None

    role_passes = passes.copy()
    role_passes["source_role"] = role_passes["player_position"].map(position_role)
    role_passes["target_role"] = role_passes["pass_recipient_position"].map(position_role)
    role_edges = (
        role_passes.groupby(["source_role", "target_role"], as_index=False)
        .agg(pass_count=("id", "size"))
    )
    role_edges["rate"] = role_edges["pass_count"] / max(passes["buildup_sequence_id"].nunique(), 1)
    global_edge_rate = float(role_edges["rate"].max()) if not role_edges.empty else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for axis, pressure in zip(axes, ["low", "high"], strict=False):
        pressure_passes = role_passes.loc[role_passes["pressure_level"].eq(pressure)]
        draw_role_network(axis, pressure_passes, pressure_level_display_label(pressure), global_edge_rate)

    fig.suptitle(
        f"{goalkeeper} ({team}) | {title_prefix} pooled-role build-up networks by pressure level",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_all_goalkeeper_network_figures(
    sequence_dataset: pd.DataFrame,
    min_sequences: int = 5,
) -> list[Path]:
    if sequence_dataset.empty:
        return []

    root = FIGURE_DIR / "all_goalkeepers"
    overall_dir = root / "overall_networks"
    pressure_dir = root / "pressure_layers"
    progression_dir = root / "progression_success_failure"
    pooled_role_dir = root / "pooled_role_pressure_layers"
    for directory in [root, overall_dir, pressure_dir, progression_dir, pooled_role_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    groups = (
        sequence_dataset.groupby(["goalkeeper_id", "goalkeeper", "team"], dropna=False)
        .size()
        .reset_index(name="sequences")
        .sort_values("sequences", ascending=False)
    )
    groups = groups.loc[groups["sequences"].ge(min_sequences)]

    for group in groups.itertuples(index=False):
        goalkeeper_id = int(group.goalkeeper_id)
        goalkeeper = str(group.goalkeeper)
        team = str(group.team)
        goalkeeper_sequences = sequence_dataset.loc[
            sequence_dataset["goalkeeper_id"].eq(goalkeeper_id)
            & sequence_dataset["team"].eq(team)
        ].copy()
        passes = completed_sequence_passes(goalkeeper_sequences)
        if passes.empty:
            continue

        file_stem = safe_filename(f"{team}_{goalkeeper}_{goalkeeper_id}")

        fig, axis = plt.subplots(figsize=(8.8, 6.4))
        draw_player_network(
            axis,
            passes,
            goalkeeper_id,
            f"{goalkeeper} ({team}) | All goalkeeper-centred sequences",
            top_edges=28,
        )
        fig.tight_layout()
        output = overall_dir / f"{file_stem}_overall_network.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
        for axis, pressure in zip(axes, ["none", "low", "high"], strict=False):
            draw_player_network(
                axis,
                passes.loc[passes["pressure_level"].eq(pressure)],
                goalkeeper_id,
                pressure_level_display_label(pressure),
                top_edges=20,
            )
        fig.suptitle(f"{goalkeeper} ({team}) | Pressure-layer build-up networks", fontsize=14, weight="bold")
        fig.tight_layout()
        output = pressure_dir / f"{file_stem}_pressure_layers.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
        draw_player_network(
            axes[0],
            passes.loc[passes["defensive_third_exit"].eq(1)],
            goalkeeper_id,
            "Defensive-third exit success",
            top_edges=24,
        )
        draw_player_network(
            axes[1],
            passes.loc[passes["defensive_third_exit"].eq(0)],
            goalkeeper_id,
            "Defensive-third exit failure",
            top_edges=24,
        )
        fig.suptitle(f"{goalkeeper} ({team}) | Progression success vs failure", fontsize=14, weight="bold")
        fig.tight_layout()
        output = progression_dir / f"{file_stem}_success_failure.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

        stage_groups = [
            ("overall", goalkeeper_sequences.copy()),
            (
                "group_stage",
                goalkeeper_sequences.loc[
                    goalkeeper_sequences["competition_stage"].fillna("").astype(str).str.strip().eq("Group Stage")
                ].copy()
                if "competition_stage" in goalkeeper_sequences.columns
                else goalkeeper_sequences.iloc[0:0].copy(),
            ),
            (
                "tournament",
                goalkeeper_sequences.loc[
                    ~goalkeeper_sequences["competition_stage"].fillna("").astype(str).str.strip().eq("Group Stage")
                ].copy()
                if "competition_stage" in goalkeeper_sequences.columns
                else goalkeeper_sequences.iloc[0:0].copy(),
            ),
        ]
        for stage_label, stage_subset in stage_groups:
            if stage_subset.empty:
                continue
            title_prefix = {
                "overall": "Overall",
                "group_stage": "Group-stage",
                "tournament": "Tournament",
            }[stage_label]
            output = pooled_role_dir / f"{file_stem}_{stage_label}_pooled_role_pressure.png"
            saved_output = save_goalkeeper_pressure_role_networks(
                stage_subset,
                goalkeeper_id,
                goalkeeper,
                team,
                title_prefix,
                output,
            )
            if saved_output is not None:
                outputs.append(saved_output)

    manifest = pd.DataFrame(
        [{"figure_path": str(path.relative_to(OUTPUT_ROOT))} for path in outputs]
    )
    manifest.to_csv(root / "all_goalkeeper_network_figures_manifest.csv", index=False, encoding="utf-8-sig")
    return outputs


def save_representative_network_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None
    candidates = (
        sequence_dataset.groupby(["goalkeeper", "team"])
        .agg(
            sequences=("buildup_sequence_id", "size"),
            final_third_entry=("final_third_entry", "mean"),
            receiver_entropy=("receiver_entropy", "mean"),
            long_distribution_share=("long_distribution_share", "mean"),
            cb_fb_distribution_share=("cb_fb_distribution_share", "mean"),
            top_receiver_share=("top_receiver_share", "mean"),
        )
        .query("sequences >= 20")
        .sort_values(["final_third_entry", "sequences"], ascending=False)
        .head(1)
    )
    if candidates.empty:
        return None
    row = candidates.iloc[0]

    graph = nx.DiGraph()
    goalkeeper_label = f"{candidates.index[0][0]}\nGK"
    nodes = [goalkeeper_label, "Centre backs", "Full backs", "Midfield", "Long option"]
    graph.add_nodes_from(nodes)
    edge_specs = [
        (goalkeeper_label, "Centre backs", max(float(row["cb_fb_distribution_share"]) * 0.65, 0.05)),
        (goalkeeper_label, "Full backs", max(float(row["cb_fb_distribution_share"]) * 0.35, 0.05)),
        (goalkeeper_label, "Midfield", max(1.0 - float(row["cb_fb_distribution_share"]) - float(row["long_distribution_share"]), 0.05)),
        (goalkeeper_label, "Long option", max(float(row["long_distribution_share"]), 0.05)),
        ("Centre backs", "Midfield", max(float(row["receiver_entropy"]), 0.05)),
        ("Full backs", "Midfield", max(float(row["receiver_entropy"]) * 0.8, 0.05)),
    ]
    for source, target, weight in edge_specs:
        graph.add_edge(source, target, weight=weight)

    positions = {
        goalkeeper_label: (0.05, 0.5),
        "Centre backs": (0.35, 0.72),
        "Full backs": (0.35, 0.28),
        "Midfield": (0.67, 0.5),
        "Long option": (0.9, 0.78),
    }
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.set_title(f"Representative goalkeeper build-up network: {candidates.index[0][0]} ({candidates.index[0][1]})")
    axis.axis("off")
    widths = [1.0 + 7.0 * graph.edges[edge]["weight"] for edge in graph.edges]
    nx.draw_networkx_nodes(graph, positions, node_color="#f3f6f8", edgecolors="#2f6f9f", node_size=2600, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_size=9, ax=axis)
    nx.draw_networkx_edges(
        graph,
        positions,
        width=widths,
        edge_color="#426a5a",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        ax=axis,
    )
    axis.text(
        0.02,
        0.04,
        "Edge widths are schematic, scaled from observed distribution shares.",
        transform=axis.transAxes,
        fontsize=9,
        color="#4d5966",
    )
    fig.tight_layout()

    output = FIGURE_DIR / "07_representative_goalkeeper_network.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_sequence_origin_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None

    outcomes = ["defensive_third_exit", "controlled_final_third_entry", "shot_created"]
    grouped = (
        sequence_dataset.groupby("sequence_origin")
        .agg(
            sequences=("buildup_sequence_id", "size"),
            defensive_third_exit=("defensive_third_exit", "mean"),
            controlled_final_third_entry=("controlled_final_third_entry", "mean"),
            shot_created=("shot_created", "mean"),
        )
        .sort_values("sequences", ascending=False)
    )
    grouped = grouped.loc[grouped["sequences"].ge(10)]
    if grouped.empty:
        return None

    fig, axis = plt.subplots(figsize=(11, 6))
    x = np.arange(len(grouped.index))
    width = 0.22
    labels = {
        "defensive_third_exit": "Defensive-third exit",
        "controlled_final_third_entry": "Controlled final-third entry",
        "shot_created": "Shot created",
    }
    colors = ["#426a5a", "#2f6f9f", "#b33f62"]

    def wilson_interval(proportion: float, sample_size: int) -> tuple[float, float]:
        if sample_size <= 0:
            return float("nan"), float("nan")
        z_value = 1.96
        denominator = 1 + z_value**2 / sample_size
        center = (proportion + z_value**2 / (2 * sample_size)) / denominator
        radius = z_value * math.sqrt(
            proportion * (1 - proportion) / sample_size
            + z_value**2 / (4 * sample_size**2)
        ) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)

    for offset, column in enumerate(outcomes):
        positions = x + (offset - 1) * width
        values = grouped[column]
        outcome_counts = (
            sequence_dataset.groupby("sequence_origin")[column]
            .count()
            .reindex(grouped.index)
        )
        intervals = [
            wilson_interval(float(value), int(outcome_counts.iloc[index]))
            for index, value in enumerate(values)
        ]
        errors = np.array([
            [float(value) - interval[0] for value, interval in zip(values, intervals)],
            [interval[1] - float(value) for value, interval in zip(values, intervals)],
        ])
        axis.bar(
            positions,
            grouped[column],
            width=width,
            label=labels[column],
            color=colors[offset],
        )
        axis.errorbar(positions, values, yerr=errors, fmt="none", ecolor="#344054", capsize=2, linewidth=0.8)

    axis.set_title("Progression outcomes by sequence origin")
    axis.set_xlabel("Sequence origin")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{str(value).replace('_', ' ').title()}\n(n={int(grouped.loc[value, 'sequences'])})" for value in grouped.index], rotation=20, ha="right")
    clean_rate_axis(axis, ylabel="Sequence outcome rate")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()

    output = FIGURE_DIR / "02_sequence_origin_progression.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_network_efficiency_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    required = {"global_efficiency", "controlled_final_third_entry", "lost_within_10s"}
    if sequence_dataset.empty or not required.issubset(sequence_dataset.columns):
        return None

    data = sequence_dataset[list(required)].dropna().copy()
    if data.empty:
        return None

    data["efficiency_bin"] = pd.qcut(
        data["global_efficiency"].rank(method="first"),
        q=min(8, len(data)),
        duplicates="drop",
    )
    grouped = (
        data.groupby("efficiency_bin", observed=False)
        .agg(
            global_efficiency=("global_efficiency", "mean"),
            controlled_final_third_entry=("controlled_final_third_entry", "mean"),
            lost_within_10s=("lost_within_10s", "mean"),
            sequences=("controlled_final_third_entry", "size"),
        )
        .reset_index(drop=True)
    )
    if grouped.empty:
        return None

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        grouped["global_efficiency"],
        grouped["controlled_final_third_entry"],
        marker="o",
        linewidth=2.4,
        color="#2f6f9f",
        label="Controlled final-third entry",
    )
    axis.plot(
        grouped["global_efficiency"],
        grouped["lost_within_10s"],
        marker="s",
        linewidth=2.4,
        color="#d9822b",
        label="Lost within 10s",
    )
    axis.set_title("Network efficiency and build-up outcomes")
    axis.set_xlabel("Mean global efficiency by sequence bin")
    clean_rate_axis(axis)
    axis.legend(frameon=False)
    fig.tight_layout()

    output = SUPPLEMENT_FIGURE_DIR / "s03_network_efficiency_outcomes.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_goalkeeper_profile_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None

    grouped = (
        sequence_dataset.groupby(["goalkeeper", "team"])
        .agg(
            sequences=("buildup_sequence_id", "size"),
            controlled_final_third_entry=("controlled_final_third_entry", "mean"),
            shot_created=("shot_created", "mean"),
            lost_within_10s=("lost_within_10s", "mean"),
        )
        .reset_index()
        .sort_values("sequences", ascending=False)
        .head(15)
    )
    if grouped.empty:
        return None

    grouped = grouped.sort_values("controlled_final_third_entry", ascending=True)
    labels = grouped["goalkeeper"] + "\n" + grouped["team"]
    y = np.arange(len(grouped))

    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(y - 0.22, grouped["controlled_final_third_entry"], height=0.2, color="#2f6f9f", label="Controlled final third")
    axis.barh(y, grouped["shot_created"], height=0.2, color="#2f9f6f", label="Shot")
    axis.barh(y + 0.22, grouped["lost_within_10s"], height=0.2, color="#d9822b", label="Loss 10s")
    axis.set_title("Goalkeeper build-up outcome profile")
    axis.set_xlabel("Sequence rate")
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlim(0, 1)
    axis.grid(axis="x", color="#d8dee9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()

    output = SUPPLEMENT_FIGURE_DIR / "s01_goalkeeper_outcome_profile.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_metric_correlation_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    variables = [
        "weighted_out_strength",
        "betweenness",
        "receiver_entropy",
        "receiver_hhi",
        "network_density",
        "global_efficiency",
        "forward_pass_share",
        "long_distribution_share",
        "controlled_final_third_entry",
        "shot_created",
        "lost_within_10s",
    ]
    available = [column for column in variables if column in sequence_dataset.columns]
    if sequence_dataset.empty or len(available) < 3:
        return None

    corr = sequence_dataset[available].apply(pd.to_numeric, errors="coerce").corr()
    labels = [column.replace("_", " ").title() for column in corr.columns]

    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_title("Network variables and outcome correlations")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    for row in range(corr.shape[0]):
        for column in range(corr.shape[1]):
            value = corr.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output = SUPPLEMENT_FIGURE_DIR / "s02_metric_outcome_correlation.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_model_forest_plot(model_results: pd.DataFrame) -> Path | None:
    if model_results.empty:
        return None
    focus_terms = [
        PRESSURE_MODEL_TERM,
        "z_global_efficiency",
        "z_long_distribution_share",
        "z_cb_fb_distribution_share",
    ]
    data = model_results.loc[
        model_results["term"].isin(focus_terms)
        & model_results["outcome"].isin(CORE_MODEL_OUTCOMES)
    ].copy()
    if data.empty:
        return None
    data["label"] = data["outcome"].str.replace("_", " ").str.title() + " | " + data["term"].str.replace("_", " ").str.title()
    data = data.sort_values(["outcome", "term"])
    y = np.arange(len(data))

    fig, axis = plt.subplots(figsize=(10, max(7, len(data) * 0.22)))
    axis.errorbar(
        data["odds_ratio"],
        y,
        xerr=[data["odds_ratio"] - data["odds_ratio_ci_low"], data["odds_ratio_ci_high"] - data["odds_ratio"]],
        fmt="o",
        color="#2f6f9f",
        ecolor="#9aa6b2",
        elinewidth=1.2,
        capsize=2,
    )
    axis.axvline(1.0, color="#2f3a45", linewidth=1.0, linestyle="--")
    axis.set_xscale("log")
    axis.set_yticks(y)
    axis.set_yticklabels(data["label"], fontsize=8)
    axis.set_xlabel("Odds ratio, log scale")
    axis.set_title("Mixed logistic model fixed-effect odds ratios")
    axis.grid(axis="x", color="#d8dee9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()

    output = FIGURE_DIR / "08_mixed_model_forest_plot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_primary_outcome_forest_plots(model_results: pd.DataFrame) -> list[Path]:
    if model_results.empty:
        return []
    focus_terms = [
        PRESSURE_MODEL_TERM,
        "z_global_efficiency",
        "z_long_distribution_share",
        "z_cb_fb_distribution_share",
    ]
    outputs: list[Path] = []
    for index, outcome in enumerate(CORE_MODEL_OUTCOMES, start=1):
        data = model_results.loc[
            model_results["outcome"].eq(outcome)
            & model_results["term"].isin(focus_terms)
        ].copy()
        if data.empty:
            continue
        data = data.sort_values("term")
        y = np.arange(len(data))
        fig, axis = plt.subplots(figsize=(8.5, 4.6))
        axis.errorbar(
            data["odds_ratio"],
            y,
            xerr=[data["odds_ratio"] - data["odds_ratio_ci_low"], data["odds_ratio_ci_high"] - data["odds_ratio"]],
            fmt="o",
            color="#2f6f9f",
            ecolor="#9aa6b2",
            capsize=2,
        )
        axis.axvline(1.0, color="#2f3a45", linewidth=1.0, linestyle="--")
        axis.set_xscale("log")
        axis.set_yticks(y)
        axis.set_yticklabels(data["term"].str.replace("z_", "").str.replace("_", " ").str.title())
        axis.set_xlabel("Odds ratio (pressure per 0.5; network metrics per 1 SD, log scale)")
        axis.set_title(f"Figure 08{chr(64 + index)}. {outcome.replace('_', ' ').title()}")
        axis.grid(axis="x", color="#d8dee9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        fig.tight_layout()
        output = FIGURE_DIR / f"08{chr(64 + index)}_{outcome}_forest_plot.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def save_empirical_prediction_figure(sequence_dataset: pd.DataFrame) -> Path | None:
    if sequence_dataset.empty:
        return None
    grouped = (
        sequence_dataset.groupby(["pressure_level", "score_state"])
        .agg(
            sequences=("buildup_sequence_id", "size"),
            final_third_entry=("final_third_entry", "mean"),
            shot_created=("shot_created", "mean"),
            lost_within_10s=("lost_within_10s", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["sequences"].ge(10)].copy()
    if grouped.empty:
        return None

    pressure_order = ["none", "low", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    outcome_labels = [
        ("final_third_entry", "Final third entry"),
        ("shot_created", "Shot created"),
        ("lost_within_10s", "Lost within 10s"),
    ]
    colors = {"drawing": "#2f6f9f", "winning": "#2f9f6f", "losing": "#d9822b"}
    for axis, (outcome, title) in zip(axes, outcome_labels, strict=False):
        for score_state, score_group in grouped.groupby("score_state"):
            score_group = score_group.set_index("pressure_level").reindex(pressure_order).dropna(subset=[outcome])
            if score_group.empty:
                continue
            axis.plot(
                score_group.index,
                score_group[outcome],
                marker="o",
                linewidth=2.2,
                color=colors.get(str(score_state), "#4d5966"),
                label=str(score_state).title(),
            )
        axis.set_title(title)
        axis.set_xlabel("Pressure level")
        axis.set_xticks(pressure_order)
        axis.set_xticklabels(
            [pressure_level_display_label(level) for level in pressure_order],
            rotation=12,
            ha="right",
        )
        clean_rate_axis(axis, ylabel="Predicted/observed probability")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Outcome probability by pressure and score state", y=1.02, fontsize=14, weight="bold")
    fig.tight_layout()

    output = FIGURE_DIR / "09_pressure_score_predicted_probabilities.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_adjusted_interaction_prediction_figure() -> Path | None:
    prediction_file = TABLE_DIR / "table_4_adjusted_predictions_and_ame.csv"
    if not prediction_file.exists():
        return None
    predictions = pd.read_csv(prediction_file)
    predictions["network_z_numeric"] = pd.to_numeric(
        predictions["network_z"],
        errors="coerce",
    )
    data = predictions.loc[
        predictions["outcome"].eq("controlled_final_third_entry")
        & predictions["network_z_numeric"].isin([-1, 1])
    ].copy()
    if data.empty:
        return None

    preferred_terms = [
        "long_distribution_share",
        "global_efficiency",
        "cb_fb_distribution_share",
    ]
    available_terms = set(data["network_term"].dropna().astype(str))
    terms = [term for term in preferred_terms if term in available_terms]
    if not terms:
        return None
    labels = {
        "long_distribution_share": "Long distribution share",
        "global_efficiency": "Global efficiency",
        "cb_fb_distribution_share": "CB/FB distribution share",
    }
    pressure_order = [
        "No pressure-anchor exposure (0.00)",
        "One-anchor exposure (0.50)",
    ]
    fig_width = max(7.0, 5.2 * len(terms))
    fig, axes = plt.subplots(1, len(terms), figsize=(fig_width, 5.2), sharey=True, squeeze=False)
    colors = {-1: "#d9822b", 1: "#2f6f9f"}
    for axis, term in zip(axes.ravel(), terms, strict=False):
        term_data = data.loc[data["network_term"].eq(term)]
        for z_value in [-1, 1]:
            values = term_data.loc[
                term_data["network_z_numeric"].eq(z_value)
            ].set_index("pressure_level").reindex(pressure_order)
            if values.empty:
                continue
            x = np.arange(len(pressure_order))
            axis.plot(x, values["adjusted_probability"], marker="o", linewidth=2.2, color=colors[z_value], label=f"{z_value:+d} SD")
            axis.fill_between(x, values["credible_interval_low"], values["credible_interval_high"], color=colors[z_value], alpha=0.18)
        axis.set_title(labels[term])
        axis.set_xticks(np.arange(len(pressure_order)))
        axis.set_xticklabels(["No pressure-anchor exposure\n(0.00)", "One-anchor exposure\n(0.50)"])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#d8dee9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Adjusted probability")
    axes[0, 0].legend(frameon=False, title="Long-distribution share")
    fig.suptitle("Adjusted probability of controlled final-third entry by pressure and long-distribution share", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Posterior fixed-effect predictions with 95% credible intervals; continuous covariates held at means, categorical covariates at reference levels, and random effects at zero.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output = FIGURE_DIR / "09_adjusted_interaction_predicted_probabilities.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _wilson_interval(proportion: float, sample_size: int) -> tuple[float, float]:
    if sample_size <= 0 or pd.isna(proportion):
        return float("nan"), float("nan")
    z_value = 1.96
    denominator = 1 + z_value**2 / sample_size
    center = (proportion + z_value**2 / (2 * sample_size)) / denominator
    radius = z_value * math.sqrt(
        proportion * (1 - proportion) / sample_size
        + z_value**2 / (4 * sample_size**2)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def save_figure_1_standalone_panels(sequence_dataset: pd.DataFrame) -> list[Path]:
    """Render Figure 1 panels on independent canvases (never cropped from a neighbour)."""
    outputs: list[Path] = []

    pressure_order = ["none", "low", "high"]
    outcomes = [("controlled_final_third_entry", "Controlled final-third entry", "#1f77b4"),
                ("defensive_third_exit", "Defensive-third exit", "#ff7f0e"),
                ("lost_within_10s", "Loss within 10 s", "#2ca02c"),
                ("shot_created", "Shot created", "#d62728")]
    grouped = sequence_dataset.groupby("pressure_level")[[x[0] for x in outcomes]].mean().reindex(pressure_order)
    totals = sequence_dataset.groupby("pressure_level").size().reindex(pressure_order).fillna(0).astype(int)
    fig, axis = plt.subplots(figsize=(8.2, 5.8)); x = np.arange(3); width=.19
    for offset, (column, label, color) in enumerate(outcomes):
        values=grouped[column].to_numpy(float)
        ns=sequence_dataset.groupby("pressure_level")[column].count().reindex(pressure_order).fillna(0).astype(int)
        ci=[_wilson_interval(v, int(n)) for v,n in zip(values,ns)]
        err=np.array([[v-lo for v,(lo,hi) in zip(values,ci)],[hi-v for v,(lo,hi) in zip(values,ci)]])
        pos=x+(offset-1.5)*width; axis.bar(pos,values,width,label=label,color=color)
        axis.errorbar(pos,values,yerr=err,fmt="none",ecolor="black",capsize=2,lw=.8)
    labels=["No pressure-anchor\nexposure","One-anchor\nexposure","All-observed-anchor\nexposure"]
    axis.set_xticks(x,[f"{v}\n(n={totals.loc[k]:,})" for v,k in zip(labels,pressure_order)],fontsize=8)
    axis.set_ylim(0,1.02); axis.set_ylabel("Sequence rate"); axis.grid(axis="y",alpha=.25)
    axis.legend(frameon=False,ncol=2,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.14))
    axis.set_title("A. Unadjusted outcomes by early pressure exposure",loc="left",weight="bold")
    path=FIGURE_DIR/"Figure_1A_unadjusted_outcomes_by_early_pressure.png"
    fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); outputs.append(path)

    origin_order=["goal_kick","open_play_recovery","free_kick","other_goalkeeper_action","throw_in","distribution_after_opponent_shot"]
    origin_names={"goal_kick":"Goal kick","open_play_recovery":"Open-play recovery","free_kick":"Free kick",
                  "other_goalkeeper_action":"Other goalkeeper action","throw_in":"Throw-in","distribution_after_opponent_shot":"After opponent shot"}
    cols=[("defensive_third_exit","Defensive-third exit","o","#1f77b4"),
          ("controlled_final_third_entry","Controlled final-third entry","s","#ff7f0e"),("shot_created","Shot created","^","#2ca02c")]
    origin=sequence_dataset.groupby("sequence_origin")[[x[0] for x in cols]].mean().reindex(origin_order).dropna(how="all")
    fig,axis=plt.subplots(figsize=(8.2,5.4)); y=np.arange(len(origin))
    for column,label,marker,color in cols:
        values=origin[column].to_numpy(float)
        ns=sequence_dataset.groupby("sequence_origin")[column].count().reindex(origin.index).fillna(0).astype(int)
        ci=[_wilson_interval(v,int(n)) for v,n in zip(values,ns)]
        err=np.array([[v-lo for v,(lo,hi) in zip(values,ci)],[hi-v for v,(lo,hi) in zip(values,ci)]])
        axis.errorbar(values,y,xerr=err,fmt=marker,color=color,capsize=2,label=label)
    axis.set_yticks(y,[origin_names[x] for x in origin.index]); axis.invert_yaxis(); axis.set_xlim(0,1.02)
    axis.set_xlabel("Sequence rate"); axis.grid(axis="x",alpha=.25); axis.legend(frameon=False,ncol=3,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.14))
    axis.set_title("B. Progression outcomes by sequence origin",loc="left",weight="bold")
    path=FIGURE_DIR/"Figure_1B_progression_outcomes_by_sequence_origin.png"
    fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); outputs.append(path)

    criteria=[("Goalkeeper first action",int(sequence_dataset["gk_starts_sequence"].sum())),
              (f"Goalkeeper pass within first {EARLY_PASS_LIMIT}",int(sequence_dataset["gk_early_pass"].sum())),
              ("Goal-kick sequence",int(sequence_dataset["from_goal_kick"].sum())),
              ("Defensive-third start",int(sequence_dataset["defensive_start"].sum()))]
    names,values=zip(*criteria); fig,axis=plt.subplots(figsize=(7.4,4.6)); y=np.arange(4)
    axis.barh(y,values,color=["#2f6f9f","#2f9f6f","#d9822b","#7b5ea7"]); axis.set_yticks(y,names); axis.invert_yaxis()
    axis.set_xlabel("Sequences"); axis.grid(axis="x",alpha=.25)
    for i,v in enumerate(values): axis.text(v+max(values)*.01,i,f"{v:,}",va="center")
    axis.set_xlim(0,max(values)*1.12); axis.set_title("C. Non-mutually exclusive inclusion criteria",loc="left",weight="bold")
    path=FIGURE_DIR/"Figure_1C_inclusion_criteria.png"
    fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); outputs.append(path)
    return outputs


def save_main_figure_1(sequence_dataset: pd.DataFrame) -> Path | None:
    """Create the three-panel descriptive Figure 1 (panels A-C)."""
    if sequence_dataset.empty:
        return None
    fig = plt.figure(figsize=(13.6, 10.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.48, wspace=0.48)
    ax_b = fig.add_subplot(grid[0, :])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    pressure_order = ["none", "low", "high"]
    outcomes = [
        ("controlled_final_third_entry", "Controlled final-third entry", "#1f77b4"),
        ("defensive_third_exit", "Defensive-third exit", "#ff7f0e"),
        ("lost_within_10s", "Loss within 10 s", "#2ca02c"),
        ("shot_created", "Shot created", "#d62728"),
    ]
    grouped = sequence_dataset.groupby("pressure_level")[[item[0] for item in outcomes]].mean().reindex(pressure_order)
    counts = sequence_dataset.groupby("pressure_level").size().reindex(pressure_order).fillna(0).astype(int)
    x_positions = np.arange(len(pressure_order)); width = 0.19
    for offset, (column, label, color) in enumerate(outcomes):
        values = grouped[column].to_numpy(dtype=float)
        outcome_ns = sequence_dataset.groupby("pressure_level")[column].count().reindex(pressure_order).fillna(0).astype(int)
        intervals = [_wilson_interval(value, int(outcome_ns.iloc[i])) for i, value in enumerate(values)]
        errors = np.array([[value - low for value, (low, high) in zip(values, intervals)],
                           [high - value for value, (low, high) in zip(values, intervals)]])
        positions = x_positions + (offset - 1.5) * width
        ax_b.bar(positions, values, width, label=label, color=color)
        ax_b.errorbar(positions, values, yerr=errors, fmt="none", ecolor="black", capsize=2, lw=0.8)
    pressure_tick_labels = ["No pressure-anchor\nexposure", "One-anchor\nexposure", "All-observed-anchor\nexposure"]
    ax_b.set_xticks(x_positions, [f"{label}\n(n={counts.loc[level]:,})" for label, level in zip(pressure_tick_labels, pressure_order)], fontsize=8)
    ax_b.set_ylim(0, 1.02); ax_b.set_ylabel("Sequence rate"); ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax_b.set_title("A. Unadjusted outcomes by early pressure exposure", loc="left", weight="bold")

    origin_order = ["goal_kick", "open_play_recovery", "free_kick", "other_goalkeeper_action", "throw_in", "distribution_after_opponent_shot"]
    origin_labels = ["Goal kick", "Open-play recovery", "Free kick", "Other goalkeeper action", "Throw-in", "After opponent shot"]
    origin_data = sequence_dataset.groupby("sequence_origin").agg(
        sequences=("buildup_sequence_id", "size"), defensive_third_exit=("defensive_third_exit", "mean"),
        controlled_final_third_entry=("controlled_final_third_entry", "mean"), shot_created=("shot_created", "mean"),
    ).reindex(origin_order).dropna(subset=["sequences"])
    origin_labels = [origin_labels[origin_order.index(value)] for value in origin_data.index]
    y_positions = np.arange(len(origin_data))
    origin_outcomes = [("defensive_third_exit", "Defensive-third exit", "o", "#1f77b4"),
                       ("controlled_final_third_entry", "Controlled final-third entry", "s", "#ff7f0e"),
                       ("shot_created", "Shot created", "^", "#2ca02c")]
    for column, label, marker, color in origin_outcomes:
        values = origin_data[column].to_numpy(dtype=float)
        ns = sequence_dataset.groupby("sequence_origin")[column].count().reindex(origin_data.index).fillna(0).to_numpy(dtype=int)
        intervals = [_wilson_interval(value, int(n)) for value, n in zip(values, ns)]
        errors = np.array([[value - low for value, (low, high) in zip(values, intervals)],
                           [high - value for value, (low, high) in zip(values, intervals)]])
        ax_c.errorbar(values, y_positions, xerr=errors, fmt=marker, color=color, capsize=2, label=label)
    ax_c.set_yticks(y_positions, origin_labels, fontsize=8); ax_c.invert_yaxis(); ax_c.set_xlim(0, 1.02)
    ax_c.set_xlabel("Sequence rate"); ax_c.grid(axis="x", alpha=0.25)
    ax_c.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax_c.set_title("B. Progression outcomes by sequence origin", loc="left", weight="bold")

    criteria = [("Goalkeeper first action", int(sequence_dataset["gk_starts_sequence"].sum())),
                (f"Goalkeeper pass within first {EARLY_PASS_LIMIT}", int(sequence_dataset["gk_early_pass"].sum())),
                ("Goal-kick sequence", int(sequence_dataset["from_goal_kick"].sum())),
                ("Defensive-third start", int(sequence_dataset["defensive_start"].sum()))]
    labels, values = zip(*criteria); y = np.arange(len(labels))
    inclusion_colors = ["#2f6f9f", "#2f9f6f", "#d9822b", "#7b5ea7"]
    ax_d.barh(y, values, color=inclusion_colors); ax_d.set_yticks(y, labels, fontsize=8); ax_d.invert_yaxis()
    ax_d.set_xlabel("Sequences"); ax_d.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values): ax_d.text(value + max(values) * 0.01, index, f"{value:,}", va="center", fontsize=8)
    ax_d.set_xlim(0, max(values) * 1.12)
    ax_d.set_title("C. Non-mutually exclusive inclusion criteria", loc="left", weight="bold")
    output = FIGURE_DIR / "Figure_1_study_design_and_descriptive_profiles.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")

    plt.close(fig)
    save_figure_1_standalone_panels(sequence_dataset)
    return output


def save_main_figure_2(model_results: pd.DataFrame, pressure_only_results: pd.DataFrame) -> Path | None:
    term_labels = {"pressure_full_sample": "Pressure — full-sample model (per 0.5)",
                   "pressure_network_augmented": "Pressure — network-augmented model (per 0.5)",
                   "z_global_efficiency": "Global efficiency (per 1 SD)",
                   "z_long_distribution_share": "Long-distribution share (per 1 SD)",
                   "z_cb_fb_distribution_share": "CB/FB distribution share (per 1 SD)"}
    outcomes = [("controlled_final_third_entry", "Controlled final-third entry"), ("lost_within_10s", "Loss within 10 s")]
    network = model_results.copy()
    network = network.loc[network["term"].isin([PRESSURE_MODEL_TERM, "z_global_efficiency", "z_long_distribution_share", "z_cb_fb_distribution_share"])]
    network["display_term"] = network["term"].replace({PRESSURE_MODEL_TERM: "pressure_network_augmented"})
    full = pressure_only_results.loc[pressure_only_results["term"].eq(PRESSURE_MODEL_TERM)].copy()
    full["display_term"] = "pressure_full_sample"
    data = pd.concat([full, network], ignore_index=True, sort=False)
    data = data.loc[data["outcome"].isin(dict(outcomes))].copy()
    if data.empty: return None
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    for axis, (outcome, title) in zip(axes, outcomes, strict=False):
        panel = data.loc[data["outcome"].eq(outcome)].set_index("display_term").reindex(term_labels)
        y = np.arange(len(term_labels)); values = panel["odds_ratio"].to_numpy(dtype=float)
        axis.errorbar(values, y, xerr=[values - panel["odds_ratio_ci_low"], panel["odds_ratio_ci_high"] - values], fmt="o", capsize=3)
        axis.axvline(1, ls="--", lw=1, color="#1f77b4"); axis.set_xscale("log"); axis.grid(axis="x", alpha=0.25)
        ticks = [0.35, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.7]
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_locator(mticker.NullLocator())
        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.set_xlim(0.35, 1.7)
        axis.set_title(title, weight="bold"); axis.set_xlabel("Odds ratio (pressure per 0.5; network metrics per 1 SD, log scale)")
    axes[0].set_yticks(np.arange(len(term_labels)), list(term_labels.values())); axes[0].invert_yaxis()
    fig.suptitle("Fixed-effect associations from the primary Bayesian multilevel models", weight="bold", fontsize=15)
    fig.tight_layout(); output = FIGURE_DIR / "Figure_2_primary_model_forest_plot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def save_main_figure_3() -> Path | None:
    prediction_file = TABLE_DIR / "table_4_adjusted_predictions_and_ame.csv"
    if not prediction_file.exists(): return None
    data = pd.read_csv(prediction_file)
    data["network_z_numeric"] = pd.to_numeric(data["network_z"], errors="coerce")
    data = data.loc[data["network_term"].eq("long_distribution_share") & data["network_z_numeric"].isin([-1, 1])].copy()
    outcomes = [("controlled_final_third_entry", "Controlled final-third entry"), ("lost_within_10s", "Loss within 10 s")]
    if data.empty: return None
    pressure_order = ["No pressure-anchor exposure (0.00)", "One-anchor exposure (0.50)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for axis, (outcome, title) in zip(axes, outcomes, strict=False):
        panel = data.loc[data["outcome"].eq(outcome)]
        for z_value, color in [(-1, "#1f77b4"), (1, "#ff7f0e")]:
            values = panel.loc[panel["network_z_numeric"].eq(z_value)].set_index("pressure_level").reindex(pressure_order)
            x = np.arange(2); probability = values["adjusted_probability"].to_numpy(dtype=float)
            errors = np.array([probability - values["credible_interval_low"].to_numpy(dtype=float), values["credible_interval_high"].to_numpy(dtype=float) - probability])
            axis.errorbar(x, probability, yerr=errors, marker="o", capsize=3, label=f"Long-distribution share: {z_value:+d} SD", color=color)
        axis.set_xticks([0, 1], ["No pressure-anchor exposure\n(0.00)", "One-anchor exposure\n(0.50)"])
        axis.set_ylabel("Conditional predicted probability"); axis.grid(axis="y", alpha=0.25); axis.set_title(title, weight="bold")
        axis.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        axis.set_ylim(bottom=0)
    fig.suptitle("Adjusted probabilities by early pressure exposure and long-distribution share", weight="bold", fontsize=15)
    fig.tight_layout(); output = FIGURE_DIR / "Figure_3_pressure_long_distribution_interaction.png"
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def save_main_figure_set(
    sequence_dataset: pd.DataFrame,
    model_results: pd.DataFrame,
    pressure_only_results: pd.DataFrame,
    figure_4_output: Path | None,
) -> list[Path]:
    """Generate main figures directly from current-run analysis outputs."""
    outputs = [save_main_figure_1(sequence_dataset), save_main_figure_2(model_results, pressure_only_results), save_main_figure_3()]
    if figure_4_output is not None: outputs.append(figure_4_output)
    outputs.extend(sorted(FIGURE_DIR.glob("Figure_1[A-D]_*.png")))
    return [output for output in outputs if output is not None]


def save_research_figures(sequence_dataset: pd.DataFrame) -> list[Path]:
    figure_functions = [
        save_study_flow_figure,
        save_pressure_outcome_figure,
        save_sequence_origin_figure,
        save_sequence_definition_figure,
        save_pressure_level_role_networks,
        save_group_stage_pressure_level_role_networks,
        save_tournament_pressure_level_role_networks,
        save_progression_success_failure_player_networks,
    ]
    outputs: list[Path] = []
    for function in figure_functions:
        output = function(sequence_dataset)
        if output is not None:
            outputs.append(output)
    for function in [
        save_goalkeeper_profile_figure,
        save_metric_correlation_figure,
        save_network_efficiency_figure,
    ]:
        output = function(sequence_dataset)
        if output is not None:
            outputs.append(output)
    outputs.extend(save_all_goalkeeper_network_figures(sequence_dataset))
    return outputs


# ---------------------------------------------------------------------
# 7. Optional multilevel logistic models
# ---------------------------------------------------------------------

def prepare_model_data(model_data: pd.DataFrame) -> pd.DataFrame:
    result = model_data.copy()
    for column in MODEL_NUMERIC_PREDICTORS:
        if column in result.columns:
            values = pd.to_numeric(
                result[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            if column not in LAYER_NETWORK_PREDICTORS:
                values = values.fillna(0.0)
            result[column] = values
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=0))
            if column == "pressure_anchor_share":
                result[PRESSURE_MODEL_TERM] = values / 0.5
            elif standard_deviation > 0:
                result[f"z_{column}"] = (values - mean) / standard_deviation
            else:
                result[f"z_{column}"] = np.nan if column in LAYER_NETWORK_PREDICTORS else 0.0
    for column in MODEL_CATEGORICAL_PREDICTORS:
        if column in result.columns:
            result[column] = result[column].fillna("unknown").astype(str)
    return result


def fit_optional_multilevel_models(model_data: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        message = (
            "statsmodels is not installed in the current Python environment.\n"
            "The sequence-level model-ready CSV was saved.\n"
            "Install statsmodels to fit Bayesian binomial mixed models with random intercepts for team and match.\n"
        )
        (MODEL_DIR / "multilevel_model_not_run.txt").write_text(message, encoding="utf-8")
        print(message.strip())
        return pd.DataFrame()

    not_run_file = MODEL_DIR / "multilevel_model_not_run.txt"
    if not_run_file.exists():
        not_run_file.unlink()

    data = prepare_model_data(model_data)
    predictors = [PRESSURE_MODEL_TERM] + [
        f"z_{column}" for column in MODEL_NUMERIC_PREDICTORS
        if column != "pressure_anchor_share"
    ] + [
        f"C({column})" for column in MODEL_CATEGORICAL_PREDICTORS
    ]
    variance_components = {
        "team": "0 + C(team)",
        "match": "0 + C(match_id)",
        "goalkeeper": "0 + C(goalkeeper_id)",
        "pressure_layer": "0 + C(layer_id)",
    }
    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for outcome in CORE_MODEL_OUTCOMES:
        outcome_data = data.dropna(
            subset=[outcome, *[f"z_{column}" for column in LAYER_NETWORK_PREDICTORS]],
        ).copy()
        if "layer_model_eligible" in outcome_data.columns:
            outcome_data = outcome_data.loc[
                outcome_data["layer_model_eligible"].fillna(False)
            ].copy()
        if outcome == "defensive_third_exit" and "defensive_third_exit_eligible" in outcome_data.columns:
            outcome_data = outcome_data.loc[outcome_data["defensive_third_exit_eligible"].eq(1)].copy()
        if outcome_data.empty or outcome_data[outcome].nunique() < 2:
            continue
        formula = f"{outcome} ~ " + " + ".join(predictors)
        try:
            model = BinomialBayesMixedGLM.from_formula(formula, variance_components, outcome_data)
            result = fit_vb_deterministic(model)
            (MODEL_DIR / f"{outcome}_mixed_logit_summary.txt").write_text(
                str(result.summary()),
                encoding="utf-8",
            )
            optim_retvals = getattr(result, "optim_retvals", {}) or {}
            diagnostic_rows.append(
                {
                    "outcome": outcome,
                    "converged": bool(optim_retvals.get("success", False)),
                    "optimizer_message": str(optim_retvals.get("message", "")),
                    "function_value": optim_retvals.get("fun", float("nan")),
                    "iterations": optim_retvals.get("nit", float("nan")),
                    "fixed_effect_terms": len(model.exog_names),
                    "random_effect_components": len(model.vcp_names),
                }
            )
            for term, mean, sd in zip(model.exog_names, result.fe_mean, result.fe_sd, strict=False):
                ci_low = float(mean - 1.96 * sd)
                ci_high = float(mean + 1.96 * sd)
                result_rows.append(
                    {
                        "outcome": outcome,
                        "term": term,
                        "log_odds": float(mean),
                        "posterior_sd": float(sd),
                        "log_odds_ci_low": ci_low,
                        "log_odds_ci_high": ci_high,
                        "odds_ratio": float(math.exp(mean)),
                        "odds_ratio_ci_low": float(math.exp(ci_low)),
                        "odds_ratio_ci_high": float(math.exp(ci_high)),
                    }
                )
            for component, mean, sd in zip(model.vcp_names, result.vcp_mean, result.vcp_sd, strict=False):
                log_sd_low = float(mean - 1.96 * sd)
                log_sd_high = float(mean + 1.96 * sd)
                result_rows.append(
                    {
                        "outcome": outcome,
                        "term": f"random_effect_sd_{component}",
                        "log_odds": float("nan"),
                        "posterior_sd": float(sd),
                        "log_odds_ci_low": float("nan"),
                        "log_odds_ci_high": float("nan"),
                        "odds_ratio": float(math.exp(mean)),
                        "odds_ratio_ci_low": float(math.exp(log_sd_low)),
                        "odds_ratio_ci_high": float(math.exp(log_sd_high)),
                        "parameter_scale": "random_effect_standard_deviation",
                    }
                )
        except Exception as error:
            (MODEL_DIR / f"{outcome}_mixed_logit_error.txt").write_text(
                repr(error),
                encoding="utf-8",
            )
            diagnostic_rows.append(
                {
                    "outcome": outcome,
                    "converged": False,
                    "optimizer_message": repr(error),
                    "function_value": float("nan"),
                    "iterations": float("nan"),
                    "fixed_effect_terms": 0,
                    "random_effect_components": 0,
                }
            )

    results = pd.DataFrame(result_rows)
    if not results.empty:
        results.to_csv(MODEL_DIR / "mixed_logit_fixed_effect_odds_ratios.csv", index=False, encoding="utf-8-sig")
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(
            MODEL_DIR / "mixed_logit_model_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return results


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------

def write_reproducibility_manifest(
    competition_id: int,
    season_id: int,
    sequence_dataset: pd.DataFrame,
    layer_dataset: pd.DataFrame,
    figure_outputs: list[Path],
) -> None:
    source_path = Path(__file__).resolve()
    source_copy = OUTPUT_ROOT / source_path.name
    if source_path != source_copy.resolve():
        shutil.copy2(source_path, source_copy)
    requirements_source = Path.cwd() / "requirements.txt"
    requirements_target = OUTPUT_ROOT / "requirements.txt"
    if (
        requirements_source.exists()
        and requirements_source.resolve() != requirements_target.resolve()
    ):
        shutil.copy2(requirements_source, requirements_target)
    package_names = [
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "scikit-learn",
        "statsbombpy",
        "statsmodels",
        "tqdm",
    ]
    package_versions = {
        package: importlib.metadata.version(package)
        for package in package_names
        if importlib.metadata.packages_distributions().get(package)
    }
    hash_targets = [
        source_copy,
        *sorted(TABLE_DIR.glob("*.csv")),
        *sorted(MODEL_DIR.glob("*.csv")),
    ]
    file_hashes = {
        path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix(): hashlib.sha256(
            path.read_bytes(),
        ).hexdigest()
        for path in hash_targets
        if path.exists() and path.is_file()
    }
    manifest = {
        "analysis_script": Path(__file__).name,
        "analysis_source_copy": str(source_copy.relative_to(OUTPUT_ROOT)),
        "analysis_version": ANALYSIS_VERSION,
        "competition": TARGET_COMPETITION_NAME,
        "season": TARGET_SEASON_NAME,
        "competition_id": competition_id,
        "season_id": season_id,
        "unit_of_analysis": "goalkeeper-centred possession sequence",
        "outcome_window": "entire post-distribution team-possession sequence",
        "network_window": f"first {INITIAL_BUILDUP_ACTIONS} post-distribution team actions, censored at the first observed outcome",
        "pressure_window": "first two post-distribution team actions (pressure anchors)",
        "primary_outcomes": CORE_MODEL_OUTCOMES,
        "random_intercepts": ["team", "match", "goalkeeper", "pressure_layer"],
        "primary_pressure_measure": "unstandardized early pressure-anchor share; odds ratios per 0.5-unit increase",
        "figure_notes": {
            "Figure_1": "Defensive-third-exit rates and Wilson 95% confidence intervals in panels A and B were calculated using outcome-eligible sequences only.",
            "Figure_3": "The models additionally adjusted for global efficiency, centre-back/full-back distribution share and match-context covariates.",
            "Figure_4": "Self-loops on the Other node represent completed passes between different players assigned to the same aggregated role category.",
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "file_sha256": file_hashes,
        "sequence_rows": int(len(sequence_dataset)),
        "layer_rows": int(len(layer_dataset)),
        "portable_json_event_caches": len(list((JSON_CACHE_DIR / "events").glob("*.json.gz"))),
        "portable_json_lineup_caches": len(list((JSON_CACHE_DIR / "lineups").glob("*.json.gz"))),
        "generated_figures": [
            path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()
            for path in sorted(FIGURE_DIR.rglob("*.png"))
        ],
        "generated_tables": [
            path.relative_to(OUTPUT_ROOT).as_posix()
            for path in sorted(TABLE_DIR.glob("*.csv"))
        ],
        "generated_models": [
            path.relative_to(OUTPUT_ROOT).as_posix()
            for path in sorted(MODEL_DIR.glob("*"))
        ],
    }
    (OUTPUT_ROOT / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = (
        "Reproduction commands:\n"
        f"  Windows: .\\.venv\\Scripts\\python.exe \".\\{Path(__file__).name}\"\n"
        f"  macOS/Linux: ./.venv/bin/python \"{Path(__file__).name}\"\n\n"
        "Install dependencies first:\n"
        "  python -m pip install -r requirements.txt\n\n"
        "The packaged euro2024_matches.csv and event/lineup caches are used by default, "
        "so reproduction does not require internet access. Set FORCE_REFRESH_CACHE=True "
        "in the script only when an online refresh is intended.\n\n"
        f"This output set is generated by {Path(__file__).name}. "
        "The legacy clustering scripts do not generate these tables or figures.\n"
    )
    (OUTPUT_ROOT / "README_reproduce.txt").write_text(readme, encoding="utf-8")


def main() -> dict[str, Any]:
    matches_file = TABLE_DIR / "euro2024_matches.csv"
    legacy_matches_file = LEGACY_OUTPUT_ROOT / "tables" / "euro2024_matches.csv"
    if not matches_file.exists() and legacy_matches_file.exists() and not FORCE_REFRESH_CACHE:
        matches_file = legacy_matches_file
    if matches_file.exists() and not FORCE_REFRESH_CACHE:
        competition_id = TARGET_COMPETITION_ID
        season_id = TARGET_SEASON_ID
        competitions = pd.DataFrame()
        matches = pd.read_csv(matches_file)
        print(f"Using local match list: {matches_file}")
    else:
        competition_id, season_id, competitions = resolve_target_competition()
        matches = retry_call(
            sb.matches,
            competition_id=competition_id,
            season_id=season_id,
        )
    matches = matches.sort_values("match_date").reset_index(drop=True)

    matches.to_csv(TABLE_DIR / "euro2024_matches.csv", index=False, encoding="utf-8-sig")
    sequence_dataset, errors = build_sequence_dataset(matches)
    export_json_reproducibility_cache()
    context_summary = summarize_context_layers(sequence_dataset)
    layer_dataset, leave_one_out_dataset = build_goalkeeper_match_pressure_layers(
        sequence_dataset,
    )
    past_only_dataset = build_past_only_network_metrics(sequence_dataset)
    model_data = save_model_ready_dataset(
        sequence_dataset,
        layer_dataset,
        leave_one_out_dataset,
    )
    if errors.empty:
        errors = pd.DataFrame(columns=["match_id", "team", "error"])
    descriptive_tables = save_descriptive_result_tables(sequence_dataset)

    sequence_dataset.to_csv(TABLE_DIR / "goalkeeper_buildup_sequence_dataset.csv", index=False, encoding="utf-8-sig")
    context_summary.to_csv(TABLE_DIR / "goalkeeper_buildup_context_summary.csv", index=False, encoding="utf-8-sig")
    layer_dataset.to_csv(TABLE_DIR / "goalkeeper_match_pressure_layer_metrics.csv", index=False, encoding="utf-8-sig")
    leave_one_out_dataset.to_csv(TABLE_DIR / "goalkeeper_match_pressure_layer_loo_metrics.csv", index=False, encoding="utf-8-sig")
    past_only_dataset.to_csv(TABLE_DIR / "goalkeeper_match_pressure_layer_past_only_metrics.csv", index=False, encoding="utf-8-sig")
    model_data.to_csv(TABLE_DIR / "goalkeeper_buildup_model_ready.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(TABLE_DIR / "processing_errors.csv", index=False, encoding="utf-8-sig")

    pressure_only_results = fit_pressure_only_multilevel_models(sequence_dataset)
    model_results = fit_optional_multilevel_models(model_data)
    interaction_results = fit_pressure_network_interaction_models(model_data)
    final_tables = {
        "table_1": save_table_1_sample_characteristics(sequence_dataset),
        "table_2": save_table_2_pressure_network_characteristics(layer_dataset),
        "table_3": save_table_3_main_multilevel_models(model_results, pressure_only_results),
        "table_4": save_table_4_key_interaction_and_probabilities(interaction_results),
        "table_5": save_table_5_sensitivity_robustness(model_data),
        "global_efficiency_size_sensitivity": save_global_efficiency_size_sensitivity(model_data),
        "alternative_estimation_sensitivity": save_alternative_estimation_sensitivity(model_data),
        "frequentist_gee_sensitivity": save_gee_estimation_sensitivity(model_data),
        "past_only_network_sensitivity": save_past_only_network_sensitivity(model_data, past_only_dataset),
        "top_receiver_share_audit": save_top_receiver_share_audit(sequence_dataset, layer_dataset),
        "all_goalkeeper_pressure_differences": save_all_goalkeeper_pressure_differences(layer_dataset),
    }
    # Figures are generated only after all model and table outputs are complete.
    # Figure 4 is descriptive and is rebuilt from current-run sequence passes.
    figure_4_output = save_progression_success_failure_player_networks(sequence_dataset)
    figure_outputs = save_main_figure_set(
        sequence_dataset,
        model_results,
        pressure_only_results,
        figure_4_output,
    )
    write_reproducibility_manifest(
        competition_id=competition_id,
        season_id=season_id,
        sequence_dataset=sequence_dataset,
        layer_dataset=layer_dataset,
        figure_outputs=figure_outputs,
    )

    print("Analysis complete")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Matches: {len(matches)}")
    print(f"Sequences: {len(sequence_dataset)}")
    print(f"Context rows: {len(context_summary)}")
    print(f"Goalkeeper-match-pressure layers: {len(layer_dataset)}")
    print(f"Model-ready rows: {len(model_data)}")
    print(f"Descriptive result tables: {len(descriptive_tables)}")
    print(f"Pressure-only model terms: {len(pressure_only_results)}")
    print(f"Mixed-model terms: {len(model_results)}")
    print(f"Interaction-model terms: {len(interaction_results)}")
    print(f"Figures: {len(figure_outputs)}")

    return {
        "competitions": competitions,
        "matches": matches,
        "sequence_dataset": sequence_dataset,
        "context_summary": context_summary,
        "layer_dataset": layer_dataset,
        "leave_one_out_dataset": leave_one_out_dataset,
        "past_only_dataset": past_only_dataset,
        "model_data": model_data,
        "descriptive_tables": descriptive_tables,
        "final_tables": final_tables,
        "model_results": model_results,
        "pressure_only_results": pressure_only_results,
        "interaction_results": interaction_results,
        "figure_outputs": figure_outputs,
        "errors": errors,
        "output_root": OUTPUT_ROOT,
    }


if __name__ == "__main__":
    RESULTS = main()
