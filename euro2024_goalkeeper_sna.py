from __future__ import annotations

import math
import os
import pickle
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.patches
from mplsoccer import Pitch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import mannwhitneyu
from statsbombpy import sb
from tqdm.auto import tqdm

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

TARGET_COMPETITION_NAME = "UEFA Euro"
TARGET_SEASON_NAME = "2024"
TARGET_GENDER = "male"

# Central analysis sample:
# - "Regular Play": open-play circulation
# - "From Goal Kick": restart build-up initiated by the goalkeeper
# Set to None to include every play pattern.
BUILDUP_PLAY_PATTERNS: set[str] | None = {"Regular Play", "From Goal Kick"}

# Operational definition. StatsBomb uses a 120 x 80 pitch coordinate model.
LONG_PASS_THRESHOLD = 30.0

MIN_TOTAL_MINUTES = 270.0
MIN_STINT_MINUTES = 1.0

# None = choose k by the largest silhouette coefficient.
# Set to 4 for a fixed four-type solution.
FORCE_K: int | None = None
K_CANDIDATES = range(2, 6)
RANDOM_STATE = 42
ROBUSTNESS_MINUTES = (180.0, 270.0, 360.0)
N_BOOTSTRAP = 1000
OUTLIER_GOALKEEPERS = ("Angus Gunn",)
STRICT_PROCESSING_VALIDATION = True
# Display both summary figures and all eligible-goalkeeper network figures.
SHOW_SUMMARY_FIGURES = True
SHOW_NETWORK_FIGURES = True

MAX_RETRIES = 4
RETRY_BASE_SECONDS = 2.0
FORCE_REFRESH_CACHE = False

DEFAULT_ROOT = (
    Path("/content/euro2024_gk_sna")
    if Path("/content").exists()
    else Path.cwd() / "euro2024_gk_sna"
)
OUTPUT_ROOT = Path(os.environ.get("EURO_GK_OUTPUT_DIR", DEFAULT_ROOT))
CACHE_DIR = OUTPUT_ROOT / "cache"
EVENT_CACHE_DIR = CACHE_DIR / "events"
LINEUP_CACHE_DIR = CACHE_DIR / "lineups"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"

for directory in [
    OUTPUT_ROOT,
    CACHE_DIR,
    EVENT_CACHE_DIR,
    LINEUP_CACHE_DIR,
    TABLE_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Generic helpers
# ---------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when the denominator is zero."""
    if denominator is None or pd.isna(denominator) or float(denominator) == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def console_safe(text: Any) -> str:
    """Replace characters unsupported by the active console encoding."""
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return value.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        return value


def weighted_average(
    values: pd.Series,
    weights: pd.Series,
    fallback: float = float("nan"),
) -> float:
    """NaN-safe weighted average."""
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)

    if not valid.any():
        valid_values = values.dropna()
        return float(valid_values.mean()) if not valid_values.empty else fallback

    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def retry_call(function, *args, **kwargs):
    """Retry transient network calls with exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return function(*args, **kwargs)
        except Exception as error:  # network/library exceptions vary
            last_error = error
            if attempt == MAX_RETRIES:
                break
            sleep_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"요청 실패 ({attempt}/{MAX_RETRIES}): {error}\n"
                f"{sleep_seconds:.0f}초 후 재시도합니다."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"{function.__name__} 호출이 {MAX_RETRIES}회 모두 실패했습니다."
    ) from last_error


def load_match_events(match_id: int) -> pd.DataFrame:
    """Load one match's events from cache or StatsBomb Open Data."""
    cache_file = EVENT_CACHE_DIR / f"{match_id}.pkl"

    if cache_file.exists() and not FORCE_REFRESH_CACHE:
        return pd.read_pickle(cache_file)

    events = retry_call(sb.events, match_id=int(match_id))
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError(f"경기 {match_id}: 이벤트 데이터가 비어 있습니다.")

    events.to_pickle(cache_file)
    return events


def load_match_lineups(match_id: int) -> dict[str, pd.DataFrame]:
    """Load one match's lineups from cache or StatsBomb Open Data."""
    cache_file = LINEUP_CACHE_DIR / f"{match_id}.pkl"

    if cache_file.exists() and not FORCE_REFRESH_CACHE:
        with cache_file.open("rb") as file:
            return pickle.load(file)

    lineups = retry_call(sb.lineups, match_id=int(match_id))
    if not isinstance(lineups, dict) or not lineups:
        raise ValueError(f"경기 {match_id}: 라인업 데이터가 비어 있습니다.")

    with cache_file.open("wb") as file:
        pickle.dump(lineups, file)

    return lineups


def clock_to_minutes(value: Any) -> float:
    """Convert StatsBomb lineup clock strings such as '92:45' to minutes."""
    if value is None:
        return float("nan")
    if isinstance(value, float) and np.isnan(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    parts = text.split(":")

    try:
        if len(parts) == 2:
            minute, second = parts
            return float(minute) + float(second) / 60.0
        if len(parts) == 3:
            hour, minute, second = parts
            return float(hour) * 60.0 + float(minute) + float(second) / 60.0
    except ValueError:
        return float("nan")

    return float("nan")


def prepare_event_time(events: pd.DataFrame) -> pd.DataFrame:
    """Add a cumulative decimal-minute column."""
    result = events.copy()
    minutes = pd.to_numeric(result.get("minute", 0), errors="coerce").fillna(0)
    seconds = pd.to_numeric(result.get("second", 0), errors="coerce").fillna(0)
    result["_event_minute"] = minutes + seconds / 60.0
    return result


def get_match_end_minute(events: pd.DataFrame) -> float:
    """Estimate match end, excluding penalty-shootout period 5."""
    if "_event_minute" not in events.columns:
        events = prepare_event_time(events)

    if "period" in events.columns:
        regulation = events[events["period"].isin([1, 2, 3, 4])]
    else:
        regulation = events

    match_end = pd.to_numeric(
        regulation["_event_minute"], errors="coerce"
    ).max()

    if pd.isna(match_end):
        return 90.0

    return max(float(match_end), 90.0)


def display_player_name(row: pd.Series | Any) -> str:
    """Prefer StatsBomb nickname; otherwise use the full player name."""
    nickname = getattr(row, "player_nickname", None)
    full_name = getattr(row, "player_name", None)

    if nickname is not None and not pd.isna(nickname) and str(nickname).strip():
        return str(nickname).strip()

    if full_name is not None and not pd.isna(full_name):
        return str(full_name).strip()

    return "Unknown goalkeeper"


def merge_intervals(stints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping/adjacent goalkeeper intervals for the same player."""
    if not stints:
        return []

    sorted_stints = sorted(
        stints,
        key=lambda row: (
            int(row["goalkeeper_id"]),
            float(row["start_minute"]),
            float(row["end_minute"]),
        ),
    )

    merged: list[dict[str, Any]] = []

    for stint in sorted_stints:
        if not merged:
            merged.append(stint.copy())
            continue

        previous = merged[-1]
        same_player = previous["goalkeeper_id"] == stint["goalkeeper_id"]
        touching = float(stint["start_minute"]) <= float(previous["end_minute"]) + 0.02

        if same_player and touching:
            previous["end_minute"] = max(
                float(previous["end_minute"]),
                float(stint["end_minute"]),
            )
        else:
            merged.append(stint.copy())

    return merged


def extract_goalkeeper_stints(
    lineup_df: pd.DataFrame,
    team_name: str,
    match_end_minute: float,
) -> list[dict[str, Any]]:
    """Extract goalkeeper on-pitch intervals from a StatsBomb lineup DataFrame."""
    stints: list[dict[str, Any]] = []

    if lineup_df is None or lineup_df.empty:
        return stints

    required = {"player_id", "player_name", "positions"}
    missing = required - set(lineup_df.columns)
    if missing:
        raise ValueError(
            f"{team_name} 라인업에 필요한 열이 없습니다: {sorted(missing)}"
        )

    for row in lineup_df.itertuples(index=False):
        positions = getattr(row, "positions", [])
        if not isinstance(positions, list):
            continue

        for position in positions:
            if not isinstance(position, dict):
                continue

            is_goalkeeper = (
                position.get("position_id") == 1
                or position.get("position") == "Goalkeeper"
            )
            if not is_goalkeeper:
                continue

            start = clock_to_minutes(position.get("from"))
            end = clock_to_minutes(position.get("to"))

            if pd.isna(start):
                start = 0.0
            if pd.isna(end):
                end = match_end_minute

            start = max(float(start), 0.0)
            end = min(max(float(end), start), match_end_minute)

            player_id = int(getattr(row, "player_id"))
            stints.append(
                {
                    "team": team_name,
                    "goalkeeper_id": player_id,
                    "goalkeeper": display_player_name(row),
                    "goalkeeper_full_name": str(getattr(row, "player_name")),
                    "start_minute": start,
                    "end_minute": end,
                    "stint_minutes": end - start,
                    "started_match": start <= 0.02,
                }
            )

    return merge_intervals(stints)


def fallback_goalkeeper_stints_from_events(
    events: pd.DataFrame,
    team_name: str,
    match_end_minute: float,
) -> list[dict[str, Any]]:
    """Fallback when lineup positions are unavailable."""
    candidates = events.loc[
        events.get("team").eq(team_name)
        & events.get("position").eq("Goalkeeper")
        & events.get("player_id").notna(),
        ["player_id", "player"],
    ].drop_duplicates()

    stints: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        stints.append(
            {
                "team": team_name,
                "goalkeeper_id": int(row.player_id),
                "goalkeeper": str(row.player),
                "goalkeeper_full_name": str(row.player),
                "start_minute": 0.0,
                "end_minute": match_end_minute,
                "stint_minutes": match_end_minute,
                "started_match": True,
            }
        )
    return stints


def filter_buildup_passes(events: pd.DataFrame) -> pd.DataFrame:
    """Select team-network passes; this is not a possession-sequence definition of build-up."""
    passes = events.loc[events["type"].eq("Pass")].copy()

    if BUILDUP_PLAY_PATTERNS is not None:
        passes = passes.loc[
            passes["play_pattern"].isin(BUILDUP_PLAY_PATTERNS)
        ].copy()

    return passes


def validate_event_columns(events: pd.DataFrame) -> None:
    required = {
        "id",
        "type",
        "team",
        "player",
        "player_id",
        "pass_recipient",
        "pass_recipient_id",
        "pass_outcome",
        "pass_length",
        "play_pattern",
        "minute",
        "second",
        "period",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            "현재 statsbombpy 출력에 필요한 열이 없습니다: "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------
# 3. One goalkeeper-stint network and distribution metrics
# ---------------------------------------------------------------------

def calculate_goalkeeper_stint_metrics(
    events: pd.DataFrame,
    team_name: str,
    goalkeeper_id: int,
    goalkeeper_name: str,
    start_minute: float,
    end_minute: float,
) -> dict[str, Any]:
    """
    Build a directed weighted team pass network for one goalkeeper's stint
    and return goalkeeper-level network/distribution metrics.
    """
    validate_event_columns(events)

    if "_event_minute" not in events.columns:
        events = prepare_event_time(events)

    # Exclude shootout events and restrict analysis to the goalkeeper's stint.
    stint_events = events.loc[
        events["period"].isin([1, 2, 3, 4])
        & events["team"].eq(team_name)
        & events["_event_minute"].ge(start_minute)
        & events["_event_minute"].lt(end_minute)
    ].copy()

    passes = filter_buildup_passes(stint_events)

    # Team network: only completed passes with an identifiable recipient.
    completed_team = passes.loc[
        passes["pass_outcome"].isna()
        & passes["player_id"].notna()
        & passes["pass_recipient_id"].notna()
    ].copy()

    if not completed_team.empty:
        completed_team["player_id"] = completed_team["player_id"].astype(int)
        completed_team["pass_recipient_id"] = (
            completed_team["pass_recipient_id"].astype(int)
        )

    edge_table = (
        completed_team.groupby(
            ["player_id", "pass_recipient_id"],
            as_index=False,
        )
        .agg(weight=("id", "size"))
        .rename(
            columns={
                "player_id": "source_id",
                "pass_recipient_id": "target_id",
            }
        )
    )

    graph = nx.DiGraph()
    if not edge_table.empty:
        for edge in edge_table.itertuples(index=False):
            graph.add_edge(
                int(edge.source_id),
                int(edge.target_id),
                weight=int(edge.weight),
                distance=1.0 / float(edge.weight),
            )

    graph.add_node(int(goalkeeper_id))

    if graph.number_of_nodes() >= 2 and graph.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(
            graph,
            weight="distance",
            normalized=True,
        )
        try:
            pagerank = nx.pagerank(
                graph,
                weight="weight",
                max_iter=1000,
            )
        except nx.PowerIterationFailedConvergence:
            pagerank = {node: float("nan") for node in graph.nodes}
    else:
        betweenness = {node: 0.0 for node in graph.nodes}
        pagerank = {node: 0.0 for node in graph.nodes}

    goalkeeper_attempts = passes.loc[
        passes["player_id"].eq(goalkeeper_id)
    ].copy()

    goalkeeper_completed = goalkeeper_attempts.loc[
        goalkeeper_attempts["pass_outcome"].isna()
        & goalkeeper_attempts["pass_recipient_id"].notna()
    ].copy()

    if not goalkeeper_completed.empty:
        goalkeeper_completed["pass_recipient_id"] = (
            goalkeeper_completed["pass_recipient_id"].astype(int)
        )

    attempted_passes = int(len(goalkeeper_attempts))
    completed_passes = int(len(goalkeeper_completed))
    team_completed_passes = int(len(completed_team))

    pass_lengths = pd.to_numeric(
        goalkeeper_attempts["pass_length"],
        errors="coerce",
    )
    long_mask = pass_lengths.ge(LONG_PASS_THRESHOLD)
    long_attempts = int(long_mask.sum())

    long_completed = int(
        (
            long_mask
            & goalkeeper_attempts["pass_outcome"].isna()
            & goalkeeper_attempts["pass_recipient_id"].notna()
        ).sum()
    )

    receiver_counts = (
        goalkeeper_completed.groupby("pass_recipient_id")
        .size()
        .sort_values(ascending=False)
    )

    receiver_diversity = int(receiver_counts.size)

    if completed_passes > 0:
        receiver_shares = receiver_counts / completed_passes
        hhi = float(np.square(receiver_shares).sum())

        if receiver_diversity > 1:
            entropy = float(
                -np.sum(receiver_shares * np.log(receiver_shares))
                / np.log(receiver_diversity)
            )
        else:
            entropy = 0.0

        top_receiver_id = int(receiver_counts.index[0])
        top_receiver_passes = int(receiver_counts.iloc[0])
        top_receiver_share = top_receiver_passes / completed_passes

        receiver_name_lookup = (
            goalkeeper_completed[
                ["pass_recipient_id", "pass_recipient"]
            ]
            .dropna()
            .drop_duplicates("pass_recipient_id")
            .set_index("pass_recipient_id")["pass_recipient"]
            .to_dict()
        )
        top_receiver = receiver_name_lookup.get(
            top_receiver_id,
            str(top_receiver_id),
        )
    else:
        hhi = float("nan")
        entropy = float("nan")
        top_receiver_id = float("nan")
        top_receiver_passes = 0
        top_receiver_share = float("nan")
        top_receiver = None

    gk_id = int(goalkeeper_id)
    out_strength = int(graph.out_degree(gk_id, weight="weight"))
    in_strength = int(graph.in_degree(gk_id, weight="weight"))
    out_degree = int(graph.out_degree(gk_id))
    in_degree = int(graph.in_degree(gk_id))

    return {
        "goalkeeper_id": gk_id,
        "goalkeeper": goalkeeper_name,
        "start_minute": float(start_minute),
        "end_minute": float(end_minute),
        "stint_minutes": float(end_minute - start_minute),
        "gk_pass_attempts": attempted_passes,
        "gk_completed_passes": completed_passes,
        "gk_pass_length_sum": float(pass_lengths.sum(skipna=True)),
        "pass_completion_pct": safe_divide(completed_passes, attempted_passes),
        "long_pass_attempts": long_attempts,
        "long_pass_completed": long_completed,
        "long_pass_share": safe_divide(long_attempts, attempted_passes),
        "long_pass_completion_pct": safe_divide(
            long_completed,
            long_attempts,
        ),
        "average_pass_length": (
            float(pass_lengths.mean()) if pass_lengths.notna().any() else float("nan")
        ),
        "team_completed_passes": team_completed_passes,
        "team_pass_share": safe_divide(
            completed_passes,
            team_completed_passes,
        ),
        "out_strength": out_strength,
        "in_strength": in_strength,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "betweenness": float(betweenness.get(gk_id, 0.0)),
        "pagerank": float(pagerank.get(gk_id, 0.0)),
        "receiver_diversity": receiver_diversity,
        "receiver_hhi": hhi,
        "normalized_entropy": entropy,
        "top_receiver_id": top_receiver_id,
        "top_receiver": top_receiver,
        "top_receiver_passes": top_receiver_passes,
        "top_receiver_share": top_receiver_share,
        "network_nodes": int(graph.number_of_nodes()),
        "network_edges": int(graph.number_of_edges()),
    }


# ---------------------------------------------------------------------
# 4. Tournament download and match-level table
# ---------------------------------------------------------------------

def resolve_target_competition() -> tuple[int, int, pd.DataFrame]:
    """Resolve IDs dynamically and verify that Euro 2024 is available."""
    competitions = retry_call(sb.competitions)

    target = competitions.loc[
        competitions["competition_name"].eq(TARGET_COMPETITION_NAME)
        & competitions["season_name"].astype(str).eq(TARGET_SEASON_NAME)
        & competitions["competition_gender"].eq(TARGET_GENDER)
    ].copy()

    if target.empty:
        raise ValueError(
            f"{TARGET_COMPETITION_NAME} {TARGET_SEASON_NAME} "
            "Open Data 항목을 찾지 못했습니다."
        )

    row = target.iloc[0]
    competition_id = int(row["competition_id"])
    season_id = int(row["season_id"])

    print(
        "대회 확인:",
        row["competition_name"],
        row["season_name"],
        f"(competition_id={competition_id}, season_id={season_id})",
    )

    return competition_id, season_id, competitions


def add_match_context(
    record: dict[str, Any],
    match_row: Any,
    team_name: str,
) -> dict[str, Any]:
    """Attach opponent, result, venue, stage and date."""
    is_home = str(match_row.home_team) == team_name

    if is_home:
        opponent = str(match_row.away_team)
        goals_for = int(match_row.home_score)
        goals_against = int(match_row.away_score)
        venue = "Home"
    else:
        opponent = str(match_row.home_team)
        goals_for = int(match_row.away_score)
        goals_against = int(match_row.home_score)
        venue = "Away"

    if goals_for > goals_against:
        result = "Win"
    elif goals_for < goals_against:
        result = "Loss"
    else:
        result = "Draw"

    record.update(
        {
            "match_id": int(match_row.match_id),
            "match_date": str(match_row.match_date),
            "competition_stage": str(match_row.competition_stage),
            "team": team_name,
            "opponent": opponent,
            "venue": venue,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": result,
        }
    )

    return record


def build_goalkeeper_match_table(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all tournament matches and return metrics plus an error log."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for match_row in tqdm(
        matches.sort_values("match_date").itertuples(index=False),
        total=len(matches),
        desc="Euro 2024 경기 처리",
    ):
        match_id = int(match_row.match_id)

        try:
            events = prepare_event_time(load_match_events(match_id))
            lineups = load_match_lineups(match_id)
            match_end = get_match_end_minute(events)

            for team_name in [
                str(match_row.home_team),
                str(match_row.away_team),
            ]:
                lineup_df = lineups.get(team_name)

                stints = extract_goalkeeper_stints(
                    lineup_df=lineup_df,
                    team_name=team_name,
                    match_end_minute=match_end,
                )

                fallback_used = False
                if not stints:
                    fallback_used = True
                    stints = fallback_goalkeeper_stints_from_events(
                        events=events,
                        team_name=team_name,
                        match_end_minute=match_end,
                    )

                if not stints:
                    errors.append(
                        {
                            "match_id": match_id,
                            "team": team_name,
                            "error": "골키퍼 출전 구간을 찾지 못함",
                        }
                    )
                    continue

                for stint in stints:
                    if stint["stint_minutes"] < MIN_STINT_MINUTES:
                        continue

                    metrics = calculate_goalkeeper_stint_metrics(
                        events=events,
                        team_name=team_name,
                        goalkeeper_id=int(stint["goalkeeper_id"]),
                        goalkeeper_name=str(stint["goalkeeper"]),
                        start_minute=float(stint["start_minute"]),
                        end_minute=float(stint["end_minute"]),
                    )

                    metrics["goalkeeper_full_name"] = stint[
                        "goalkeeper_full_name"
                    ]
                    metrics["started_match"] = bool(stint["started_match"])
                    metrics["lineup_fallback_used"] = fallback_used

                    records.append(
                        add_match_context(
                            record=metrics,
                            match_row=match_row,
                            team_name=team_name,
                        )
                    )

        except Exception as error:
            errors.append(
                {
                    "match_id": match_id,
                    "team": None,
                    "error": repr(error),
                }
            )
            print(f"경기 {match_id} 처리 실패: {error}")

    metrics_df = pd.DataFrame(records)
    errors_df = pd.DataFrame(errors, columns=["match_id", "team", "error"])

    if metrics_df.empty:
        raise RuntimeError(
            "골키퍼 경기별 지표가 생성되지 않았습니다. 오류 로그를 확인하십시오."
        )

    return metrics_df, errors_df


# ---------------------------------------------------------------------
# 5. Goalkeeper tournament aggregation
# ---------------------------------------------------------------------

def aggregate_goalkeepers(
    goalkeeper_match: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate stint-level records to one row per goalkeeper/team."""
    rows: list[dict[str, Any]] = []

    group_columns = [
        "goalkeeper_id",
        "goalkeeper",
        "goalkeeper_full_name",
        "team",
    ]

    for group_key, group in goalkeeper_match.groupby(
        group_columns,
        dropna=False,
    ):
        goalkeeper_id, goalkeeper, full_name, team = group_key
        minutes = float(group["stint_minutes"].sum())
        attempts = int(group["gk_pass_attempts"].sum())
        completed = int(group["gk_completed_passes"].sum())
        long_attempts = int(group["long_pass_attempts"].sum())
        long_completed = int(group["long_pass_completed"].sum())
        team_completed = int(group["team_completed_passes"].sum())

        row = {
            "goalkeeper_id": int(goalkeeper_id),
            "goalkeeper": goalkeeper,
            "goalkeeper_full_name": full_name,
            "team": team,
            "matches": int(group["match_id"].nunique()),
            "starts": int(group["started_match"].sum()),
            "minutes": minutes,
            "pass_attempts": attempts,
            "completed_passes": completed,
            "pass_attempts_per90": safe_divide(attempts * 90.0, minutes),
            "completed_passes_per90": safe_divide(completed * 90.0, minutes),
            "pass_completion_pct": safe_divide(completed, attempts),
            "long_pass_attempts": long_attempts,
            "long_pass_completed": long_completed,
            "long_pass_share": safe_divide(long_attempts, attempts),
            "long_pass_completion_pct": safe_divide(
                long_completed,
                long_attempts,
            ),
            "average_pass_length": safe_divide(
                group["gk_pass_length_sum"].sum(),
                attempts,
            ),
            "team_completed_passes_per90": safe_divide(
                team_completed * 90.0,
                minutes,
            ),
            "team_pass_share": safe_divide(completed, team_completed),
            "out_strength_per90": safe_divide(
                group["out_strength"].sum() * 90.0,
                minutes,
            ),
            "in_strength_per90": safe_divide(
                group["in_strength"].sum() * 90.0,
                minutes,
            ),
            "betweenness_mean": weighted_average(
                group["betweenness"],
                group["stint_minutes"],
            ),
            "pagerank_mean": weighted_average(
                group["pagerank"],
                group["stint_minutes"],
            ),
            "receiver_diversity_mean": weighted_average(
                group["receiver_diversity"],
                group["stint_minutes"],
            ),
            "receiver_hhi": weighted_average(
                group["receiver_hhi"],
                group["gk_completed_passes"],
            ),
            "normalized_entropy": weighted_average(
                group["normalized_entropy"],
                group["gk_completed_passes"],
            ),
            "top_receiver_share_mean": weighted_average(
                group["top_receiver_share"],
                group["gk_completed_passes"],
            ),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["minutes", "team"],
        ascending=[False, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# 6. K-means typology
# ---------------------------------------------------------------------

CLUSTER_FEATURES = [
    "team_pass_share",
    "betweenness_mean",
    "receiver_diversity_mean",
    "normalized_entropy",
    "long_pass_share",
]

FEATURE_LABELS_EN = {
    "team_pass_share": "GK share of team completed passes",
    "betweenness_mean": "Betweenness centrality",
    "receiver_diversity_mean": "Mean receiver diversity",
    "normalized_entropy": "Normalized entropy",
    "long_pass_share": "GK long-pass attempt share",
}


def choose_cluster_count(
    scaled_features: np.ndarray,
) -> tuple[int, pd.DataFrame]:
    """Choose k using silhouette, unless FORCE_K is set."""
    n_samples = scaled_features.shape[0]

    if n_samples < 3:
        raise ValueError("군집분석에는 최소 3명의 골키퍼가 필요합니다.")

    valid_k = [
        k
        for k in K_CANDIDATES
        if 2 <= k < n_samples
    ]

    if not valid_k:
        raise ValueError("표본 수에 맞는 군집 수 후보가 없습니다.")

    scores: list[dict[str, float]] = []

    for k in valid_k:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50,
        )
        labels = model.fit_predict(scaled_features)
        scores.append({
            "k": int(k),
            "silhouette": float(silhouette_score(scaled_features, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(scaled_features, labels)),
            "davies_bouldin": float(davies_bouldin_score(scaled_features, labels)),
        })

    score_df = pd.DataFrame(scores)

    if FORCE_K is not None:
        if FORCE_K not in valid_k:
            raise ValueError(
                f"FORCE_K={FORCE_K}는 현재 표본에서 사용할 수 없습니다."
            )
        selected_k = int(FORCE_K)
    else:
        selected_k = int(
            score_df.sort_values(
                ["silhouette", "k"],
                ascending=[False, True],
            ).iloc[0]["k"]
        )

    return selected_k, score_df


def make_cluster_names(
    profile_z: pd.DataFrame,
) -> dict[int, str]:
    """
    Assign transparent descriptive labels from standardized centroids.
    Labels are interpretive summaries, not externally validated categories.
    """
    remaining = set(int(index) for index in profile_z.index)
    names: dict[int, str] = {}

    if len(remaining) == 2:
        active_scores = profile_z[CLUSTER_FEATURES].mean(axis=1)
        active_cluster = int(active_scores.loc[list(remaining)].idxmax())
        limited_cluster = next(
            cluster for cluster in remaining if cluster != active_cluster
        )
        return {
            active_cluster: "Relatively higher-involvement configuration",
            limited_cluster: "Relatively lower-involvement configuration",
        }

    if remaining:
        direct_scores = (
            profile_z["long_pass_share"]
            - 0.25 * profile_z["team_pass_share"]
        )
        cluster = int(direct_scores.loc[list(remaining)].idxmax())
        names[cluster] = "Direct long-distribution profile"
        remaining.remove(cluster)

    if remaining:
        hub_scores = (
            profile_z["team_pass_share"]
            + profile_z["betweenness_mean"]
        ) / 2.0
        cluster = int(hub_scores.loc[list(remaining)].idxmax())
        names[cluster] = "Build-up hub profile"
        remaining.remove(cluster)

    if remaining:
        diversity_scores = (
            profile_z["receiver_diversity_mean"]
            + profile_z["normalized_entropy"]
        ) / 2.0
        cluster = int(diversity_scores.loc[list(remaining)].idxmax())
        names[cluster] = "Diversified distribution profile"
        remaining.remove(cluster)

    for sequence, cluster in enumerate(sorted(remaining), start=1):
        suffix = "" if len(remaining) == 1 else f" {sequence}"
        names[int(cluster)] = f"Balanced mixed profile{suffix}"

    return names


def cluster_goalkeepers(
    goalkeeper_summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
    KMeans,
    StandardScaler,
]:
    """Filter eligible goalkeepers, cluster them, and build profiles."""
    eligible = goalkeeper_summary.loc[
        goalkeeper_summary["minutes"].ge(MIN_TOTAL_MINUTES)
    ].copy()

    eligible = eligible.dropna(subset=CLUSTER_FEATURES).reset_index(drop=True)

    if len(eligible) < 3:
        raise ValueError(
            f"{MIN_TOTAL_MINUTES:.0f}분 기준을 통과한 골키퍼가 "
            "군집분석에 충분하지 않습니다."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(eligible[CLUSTER_FEATURES])

    selected_k, silhouette_df = choose_cluster_count(scaled)

    model = KMeans(
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
        n_init=50,
    )
    labels = model.fit_predict(scaled)

    eligible["cluster"] = labels.astype(int)
    eligible["distance_to_centroid"] = np.linalg.norm(
        scaled - model.cluster_centers_[labels],
        axis=1,
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coordinates = pca.fit_transform(scaled)
    eligible["pca1"] = pca_coordinates[:, 0]
    eligible["pca2"] = pca_coordinates[:, 1]

    profile_z = pd.DataFrame(
        model.cluster_centers_,
        columns=CLUSTER_FEATURES,
    )
    profile_z.index.name = "cluster"

    profile_raw = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=CLUSTER_FEATURES,
    )
    profile_raw.index.name = "cluster"

    cluster_names = make_cluster_names(profile_z)
    eligible["cluster_name"] = eligible["cluster"].map(cluster_names)

    profile_z.insert(
        0,
        "cluster_name",
        [cluster_names[int(index)] for index in profile_z.index],
    )
    profile_raw.insert(
        0,
        "cluster_name",
        [cluster_names[int(index)] for index in profile_raw.index],
    )

    representatives = (
        eligible.sort_values("distance_to_centroid")
        .groupby("cluster", as_index=False)
        .first()[
            [
                "cluster",
                "cluster_name",
                "goalkeeper_id",
                "goalkeeper",
                "team",
                "distance_to_centroid",
            ]
        ]
    )

    return (
        eligible,
        silhouette_df,
        profile_raw,
        profile_z,
        selected_k,
        model,
        scaler,
    )


# ---------------------------------------------------------------------
# 6A. Cluster robustness, context documentation, and supplementary validation
# ---------------------------------------------------------------------

EXTERNAL_VALIDATION_VARIABLES = [
    "pass_completion_pct",
    "completed_passes_per90",
    "average_pass_length",
    "long_pass_completion_pct",
]


def cluster_quality_indices(scaled: np.ndarray) -> pd.DataFrame:
    """Evaluate candidate k with three complementary internal indices."""
    rows = []
    for k in K_CANDIDATES:
        if not 2 <= k < len(scaled):
            continue
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(scaled)
        rows.append({
            "k": int(k),
            "silhouette": silhouette_score(scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(scaled, labels),
            "davies_bouldin": davies_bouldin_score(scaled, labels),
        })
    return pd.DataFrame(rows)


def fit_labels(frame: pd.DataFrame, k: int, method: str = "kmeans") -> tuple[np.ndarray, np.ndarray]:
    """Standardize features and return labels plus standardized observations."""
    scaled = StandardScaler().fit_transform(frame[CLUSTER_FEATURES])
    if method == "ward":
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
    elif method == "gmm":
        labels = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-5,
                                 n_init=50, random_state=RANDOM_STATE).fit_predict(scaled)
    else:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(scaled)
    return labels.astype(int), scaled


def method_sensitivity(eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compare Ward and Gaussian mixtures with the primary K-means partition."""
    base = eligible["cluster"].to_numpy()
    rows = [{"method": "K-means", "adjusted_rand_vs_kmeans": 1.0,
             "bic": np.nan, "converged": True}]
    labels, scaled = fit_labels(eligible, k, "ward")
    rows.append({"method": "Ward hierarchical",
                 "adjusted_rand_vs_kmeans": adjusted_rand_score(base, labels),
                 "bic": np.nan, "converged": True})
    gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-5,
                          n_init=50, random_state=RANDOM_STATE).fit(scaled)
    rows.append({"method": "Gaussian mixture (diagonal covariance)",
                 "adjusted_rand_vs_kmeans": adjusted_rand_score(base, gmm.predict(scaled)),
                 "bic": gmm.bic(scaled), "converged": bool(gmm.converged_)})
    robust = RobustScaler().fit_transform(eligible[CLUSTER_FEATURES])
    robust_labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(robust)
    rows.append({"method": "K-means with RobustScaler",
                 "adjusted_rand_vs_kmeans": adjusted_rand_score(base, robust_labels),
                 "bic": np.nan, "converged": True})
    return pd.DataFrame(rows)


def select_scenario_cluster_count(frame: pd.DataFrame) -> tuple[int, float]:
    """Select k afresh for a sensitivity sample, independently of FORCE_K."""
    scaled = StandardScaler().fit_transform(frame[CLUSTER_FEATURES])
    indices = cluster_quality_indices(scaled)
    best = indices.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best["k"]), float(best["silhouette"])


def pca_diagnostics(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return explained variance and variable loadings for the plotted PCA."""
    scaled = StandardScaler().fit_transform(eligible[CLUSTER_FEATURES])
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaled)
    variance = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"])
    loadings.insert(0, "feature", CLUSTER_FEATURES)
    return variance, loadings


def bootstrap_cluster_jaccard(eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Clusterboot-style stability: refit resamples, classify all cases, match clusters by Jaccard."""
    x = eligible[CLUSTER_FEATURES].to_numpy(float)
    base = eligible["cluster"].to_numpy(int)
    rng = np.random.default_rng(RANDOM_STATE)
    scores = {cluster: [] for cluster in sorted(np.unique(base))}
    for _ in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(x), len(x))
        scaler = StandardScaler().fit(x[sample])
        xb = scaler.transform(x[sample])
        model = KMeans(n_clusters=k, random_state=int(rng.integers(0, 2**31 - 1)), n_init=20).fit(xb)
        predicted = model.predict(scaler.transform(x))
        for cluster in scores:
            original_set = base == cluster
            best = 0.0
            for candidate in range(k):
                new_set = predicted == candidate
                union = np.logical_or(original_set, new_set).sum()
                if union:
                    best = max(best, np.logical_and(original_set, new_set).sum() / union)
            scores[cluster].append(best)
    return pd.DataFrame([
        {"cluster": c, "mean_jaccard": np.mean(v), "ci_low": np.quantile(v, .025),
         "ci_high": np.quantile(v, .975), "replications": N_BOOTSTRAP}
        for c, v in scores.items()
    ])


def subset_sensitivity(eligible: pd.DataFrame, goalkeeper_summary: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out, minute-threshold and prespecified outlier checks."""
    loo_rows = []
    for idx, omitted in eligible.iterrows():
        reduced = eligible.drop(index=idx).reset_index(drop=True)
        scenario_k, best_silhouette = select_scenario_cluster_count(reduced)
        labels, _ = fit_labels(reduced, scenario_k)
        loo_rows.append({
            "omitted_goalkeeper": omitted["goalkeeper"], "n_remaining": len(reduced),
            "selected_k": scenario_k, "best_silhouette": best_silhouette,
            "adjusted_rand_index": adjusted_rand_score(reduced["cluster"], labels),
        })
    sensitivity = []
    scenarios = [(f"minimum_{int(m)}_minutes",
                  goalkeeper_summary.loc[goalkeeper_summary["minutes"].ge(m)].dropna(subset=CLUSTER_FEATURES))
                 for m in ROBUSTNESS_MINUTES]
    scenarios += [(f"exclude_{name.replace(' ', '_')}", eligible.loc[~eligible["goalkeeper"].eq(name)])
                  for name in OUTLIER_GOALKEEPERS]
    for scenario, frame in scenarios:
        frame = frame.reset_index(drop=True)
        if len(frame) < 3:
            sensitivity.append({"scenario": scenario, "n": len(frame), "selected_k": np.nan,
                                "best_silhouette": np.nan, "adjusted_rand_on_overlap": np.nan})
            continue
        scenario_k, best_silhouette = select_scenario_cluster_count(frame)
        labels, _ = fit_labels(frame, scenario_k)
        comparison = frame[["goalkeeper_id"]].copy()
        comparison["new_cluster"] = labels
        comparison = comparison.merge(eligible[["goalkeeper_id", "cluster"]], on="goalkeeper_id")
        ari = adjusted_rand_score(comparison["cluster"], comparison["new_cluster"]) if len(comparison) > 1 else np.nan
        sensitivity.append({"scenario": scenario, "n": len(frame), "selected_k": scenario_k,
                            "best_silhouette": best_silhouette, "overlap_n": len(comparison),
                            "adjusted_rand_on_overlap": ari})
    return pd.DataFrame(loo_rows), pd.DataFrame(sensitivity)


def leave_one_feature_out_sensitivity(eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Test whether any single clustering feature drives the primary partition."""
    base = eligible["cluster"].to_numpy()
    rows = []
    for omitted in CLUSTER_FEATURES:
        features = [feature for feature in CLUSTER_FEATURES if feature != omitted]
        scaled = StandardScaler().fit_transform(eligible[features])
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(scaled)
        rows.append({"omitted_feature": omitted,
                     "adjusted_rand_index": adjusted_rand_score(base, labels)})
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Nonparametric effect size: P(a>b) - P(a<b)."""
    return float(np.sign(a[:, None] - b[None, :]).mean())


def supplementary_cluster_validation(eligible: pd.DataFrame) -> pd.DataFrame:
    """Compare clusters on non-input passing metrics; this is not external validation."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    clusters = sorted(eligible["cluster"].unique())
    for variable in EXTERNAL_VALIDATION_VARIABLES:
        for i, first in enumerate(clusters):
            for second in clusters[i + 1:]:
                a = eligible.loc[eligible["cluster"].eq(first), variable].dropna().to_numpy(float)
                b = eligible.loc[eligible["cluster"].eq(second), variable].dropna().to_numpy(float)
                if not len(a) or not len(b):
                    continue
                delta = cliffs_delta(a, b)
                boot = [cliffs_delta(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                        for _ in range(N_BOOTSTRAP)]
                test = mannwhitneyu(a, b, alternative="two-sided")
                rows.append({"variable": variable, "cluster_1": first, "cluster_2": second,
                             "n_1": len(a), "n_2": len(b), "mean_1": a.mean(), "mean_2": b.mean(),
                             "median_1": np.median(a), "median_2": np.median(b),
                             "cliffs_delta": delta, "delta_ci_low": np.quantile(boot, .025),
                             "delta_ci_high": np.quantile(boot, .975), "mann_whitney_u": test.statistic,
                             "p_value": test.pvalue})
    result = pd.DataFrame(rows)
    if not result.empty:
        order = np.argsort(result["p_value"].to_numpy())
        ordered_p = result["p_value"].to_numpy()[order]
        adjusted = np.maximum.accumulate(ordered_p * (len(result) - np.arange(len(result))))
        holm = np.empty(len(result))
        holm[order] = np.minimum(1.0, adjusted)
        result["p_holm"] = holm
    return result


def match_resampling_stability(goalkeeper_match: pd.DataFrame, eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Resample matches within goalkeeper, reaggregate, and compare partitions on retained cases."""
    rng = np.random.default_rng(RANDOM_STATE)
    aris = []
    groups = list(goalkeeper_match.groupby(["goalkeeper_id", "goalkeeper", "goalkeeper_full_name", "team"], dropna=False))
    for _ in range(N_BOOTSTRAP):
        sampled = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for _, g in groups], ignore_index=True)
        summary = aggregate_goalkeepers(sampled)
        frame = summary.merge(eligible[["goalkeeper_id", "cluster"]], on="goalkeeper_id").dropna(subset=CLUSTER_FEATURES)
        if len(frame) > k:
            labels, _ = fit_labels(frame, k)
            aris.append(adjusted_rand_score(frame["cluster"], labels))
    return pd.DataFrame({"measure": ["adjusted_rand_index"], "mean": [np.mean(aris)],
                         "ci_low": [np.quantile(aris, .025)], "ci_high": [np.quantile(aris, .975)],
                         "replications": [len(aris)]})


def context_availability_table() -> pd.DataFrame:
    """Prevent unavailable team/context controls from being silently implied."""
    return pd.DataFrame([
        ("team pass volume", "available but not explicitly adjusted",
         "team_completed_passes_per90 is available; team_pass_share uses team completed passes as its denominator"),
        ("competition stage", "available descriptively", "competition_stage in goalkeeper-match table"),
        ("match result", "available proxy", "result and goals_for/goals_against; not time-varying game state"),
        ("opponent pressure", "not derived", "requires event-level pressure operationalisation"),
        ("leading/drawing/trailing", "not derived", "requires score timeline at each action"),
        ("goal kick/open play share", "not separately aggregated", "play patterns are pooled in current sample"),
        ("opponent strength", "not available", "requires an external or tournament-strength measure"),
        ("goalkeeper re-involvements", "not derived", "requires sequence-level event linkage"),
        ("passes retained after goalkeeper pass", "not derived", "requires sequence-level event linkage"),
    ], columns=["factor", "status", "implementation_note"])


def processing_quality_control(
    matches: pd.DataFrame,
    goalkeeper_match: pd.DataFrame,
    error_log: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise and, optionally, enforce complete tournament processing."""
    collected = int(len(matches))
    processed = int(goalkeeper_match["match_id"].nunique())
    failed_ids = set(pd.to_numeric(error_log.get("match_id", pd.Series(dtype=float)),
                                   errors="coerce").dropna().astype(int))
    team_match_pairs = int(goalkeeper_match[["match_id", "team"]].drop_duplicates().shape[0])
    fallback_count = int(goalkeeper_match.get(
        "lineup_fallback_used", pd.Series(False, index=goalkeeper_match.index)
    ).fillna(False).astype(bool).sum())
    quality = pd.DataFrame([{
        "matches_collected": collected,
        "matches_processed": processed,
        "matches_with_errors": len(failed_ids),
        "processed_team_match_pairs": team_match_pairs,
        "expected_team_match_pairs": 2 * collected,
        "goalkeeper_stints": len(goalkeeper_match),
        "lineup_fallback_stints": fallback_count,
        "complete": bool(not len(error_log) and processed == collected
                         and team_match_pairs == 2 * collected and fallback_count == 0),
    }])
    return quality


# ---------------------------------------------------------------------
# 7. Visualizations
# ---------------------------------------------------------------------

def save_silhouette_plot(
    silhouette_df: pd.DataFrame,
    selected_k: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        silhouette_df["k"],
        silhouette_df["silhouette"],
        marker="o",
    )
    selected_score = silhouette_df.loc[
        silhouette_df["k"].eq(selected_k),
        "silhouette",
    ].iloc[0]
    ax.scatter([selected_k], [selected_score], s=100, zorder=3)
    ax.set_title("Selecting the Number of Clusters: Silhouette Coefficient")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette coefficient")
    ax.set_xticks(silhouette_df["k"])
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output = FIGURE_DIR / "01_silhouette_scores.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if SHOW_SUMMARY_FIGURES:
        plt.show()
    plt.close(fig)
    return output


def save_pca_plot(
    eligible: pd.DataFrame,
    pca_variance: pd.DataFrame | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8))

    for cluster_name, group in eligible.groupby("cluster_name"):
        ax.scatter(
            group["pca1"],
            group["pca2"],
            s=85,
            alpha=0.85,
            label=cluster_name,
        )

        for row in group.itertuples(index=False):
            ax.annotate(
                f"{row.goalkeeper}\n({row.team})",
                (row.pca1, row.pca2),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axhline(0, linewidth=0.7, alpha=0.4)
    ax.axvline(0, linewidth=0.7, alpha=0.4)
    ax.set_title("UEFA Euro 2024 Goalkeeper Involvement in Team Passing Networks")
    if pca_variance is not None and len(pca_variance) >= 2:
        pc1 = 100 * float(pca_variance.iloc[0]["explained_variance_ratio"])
        pc2 = 100 * float(pca_variance.iloc[1]["explained_variance_ratio"])
        ax.set_xlabel(f"PC1 ({pc1:.1f}% explained variance)")
        ax.set_ylabel(f"PC2 ({pc2:.1f}% explained variance)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax.legend(title="Exploratory configuration", loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output = FIGURE_DIR / "02_goalkeeper_clusters_pca.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if SHOW_SUMMARY_FIGURES:
        plt.show()
    plt.close(fig)
    return output


def save_cluster_profile_heatmap(profile_z: pd.DataFrame) -> Path:
    numeric = profile_z[CLUSTER_FEATURES].copy()
    labels = profile_z["cluster_name"].tolist()
    max_abs = max(float(np.abs(numeric.to_numpy()).max()), 1.0)

    fig, ax = plt.subplots(
        figsize=(10, max(4, 1.15 * len(numeric))),
    )
    image = ax.imshow(
        numeric.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(range(len(CLUSTER_FEATURES)))
    ax.set_xticklabels(
        [FEATURE_LABELS_EN[column] for column in CLUSTER_FEATURES],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Cluster Centroid Profiles (z-scores)")

    for row_index in range(numeric.shape[0]):
        for column_index in range(numeric.shape[1]):
            value = numeric.iloc[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Standardized cluster centroid")
    fig.tight_layout()

    output = FIGURE_DIR / "03_cluster_profile_heatmap.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if SHOW_SUMMARY_FIGURES:
        plt.show()
    plt.close(fig)
    return output


def save_involvement_longpass_plot(eligible: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8))

    for cluster_name, group in eligible.groupby("cluster_name"):
        sizes = 5000 * group["betweenness_mean"].clip(lower=0) + 55
        ax.scatter(
            group["team_pass_share"] * 100,
            group["long_pass_share"] * 100,
            s=sizes,
            alpha=0.75,
            label=cluster_name,
        )

        for row in group.itertuples(index=False):
            ax.annotate(
                row.goalkeeper,
                (
                    row.team_pass_share * 100,
                    row.long_pass_share * 100,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(
        "Goalkeeper Share of Team Passes vs. Long-Pass Share\n"
        "Point size = mean betweenness centrality"
    )
    ax.set_xlabel("Goalkeeper share of completed team passes (%)")
    ax.set_ylabel("Goalkeeper long-pass attempt share (%)")
    ax.legend(title="Exploratory configuration", loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output = FIGURE_DIR / "04_involvement_vs_long_pass.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if SHOW_SUMMARY_FIGURES:
        plt.show()
    plt.close(fig)
    return output


def get_node_name_lookup(
    lineups: dict[str, pd.DataFrame],
    team_name: str,
) -> dict[int, str]:
    lineup_df = lineups.get(team_name)
    if lineup_df is None or lineup_df.empty:
        return {}

    lookup: dict[int, str] = {}
    for row in lineup_df.itertuples(index=False):
        lookup[int(row.player_id)] = display_player_name(row)
    return lookup


def calculate_node_locations(
    completed_passes: pd.DataFrame,
) -> dict[int, tuple[float, float]]:
    """Average sender starts and recipient ends for a pass-network pitch plot."""
    location_rows: list[dict[str, float]] = []

    for row in completed_passes.itertuples(index=False):
        start = getattr(row, "location", None)
        end = getattr(row, "pass_end_location", None)

        if (
            isinstance(start, (list, tuple, np.ndarray))
            and len(start) >= 2
        ):
            location_rows.append(
                {
                    "player_id": int(row.player_id),
                    "x": float(start[0]),
                    "y": float(start[1]),
                }
            )

        if (
            isinstance(end, (list, tuple, np.ndarray))
            and len(end) >= 2
        ):
            location_rows.append(
                {
                    "player_id": int(row.pass_recipient_id),
                    "x": float(end[0]),
                    "y": float(end[1]),
                }
            )

    if not location_rows:
        return {}

    location_df = pd.DataFrame(location_rows)
    average_locations = (
        location_df.groupby("player_id")[["x", "y"]]
        .mean()
        .to_dict("index")
    )

    return {
        int(player_id): (float(values["x"]), float(values["y"]))
        for player_id, values in average_locations.items()
    }


def short_label(name: str) -> str:
    words = str(name).strip().split()
    if len(words) <= 2:
        return str(name)
    return " ".join(words[-2:])


def plot_goalkeeper_match_network(
    goalkeeper_match_row: pd.Series,
    minimum_edge_weight: int = 1,
) -> Path:
    """Plot the team pass network for one goalkeeper match."""
    match_id = int(goalkeeper_match_row["match_id"])
    team_name = str(goalkeeper_match_row["team"])
    goalkeeper_id = int(goalkeeper_match_row["goalkeeper_id"])
    goalkeeper_name = str(goalkeeper_match_row["goalkeeper"])
    start = float(goalkeeper_match_row["start_minute"])
    end = float(goalkeeper_match_row["end_minute"])

    events = prepare_event_time(load_match_events(match_id))
    lineups = load_match_lineups(match_id)

    window = events.loc[
        events["period"].isin([1, 2, 3, 4])
        & events["team"].eq(team_name)
        & events["_event_minute"].ge(start)
        & events["_event_minute"].lt(end)
    ].copy()

    passes = filter_buildup_passes(window)
    completed = passes.loc[
        passes["pass_outcome"].isna()
        & passes["player_id"].notna()
        & passes["pass_recipient_id"].notna()
    ].copy()

    if completed.empty:
        raise ValueError("대표 경기의 성공 패스가 없습니다.")

    completed["player_id"] = completed["player_id"].astype(int)
    completed["pass_recipient_id"] = (
        completed["pass_recipient_id"].astype(int)
    )

    edges = (
        completed.groupby(
            ["player_id", "pass_recipient_id"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "player_id": "source_id",
                "pass_recipient_id": "target_id",
                "size": "weight",
            }
        )
    )
    edges = edges.loc[edges["weight"].ge(minimum_edge_weight)].copy()

    node_locations = calculate_node_locations(completed)
    node_names = get_node_name_lookup(lineups, team_name)

    graph = nx.DiGraph()
    for edge in edges.itertuples(index=False):
        if (
            int(edge.source_id) in node_locations
            and int(edge.target_id) in node_locations
        ):
            graph.add_edge(
                int(edge.source_id),
                int(edge.target_id),
                weight=int(edge.weight),
            )

    if graph.number_of_edges() == 0:
        raise ValueError(
            "최소 간선 가중치 조건을 충족하는 패스 관계가 없습니다."
        )

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="black",
    )
    fig, ax = pitch.draw(figsize=(13, 8))

    strengths = {
        node: (
            graph.in_degree(node, weight="weight")
            + graph.out_degree(node, weight="weight")
        )
        for node in graph.nodes
    }

    max_edge_weight = max(
        data["weight"]
        for _, _, data in graph.edges(data=True)
    )

    for source, target, data in graph.edges(data=True):
        start_xy = node_locations[source]
        end_xy = node_locations[target]
        width = 0.7 + 4.0 * data["weight"] / max_edge_weight

        arrow = matplotlib.patches.FancyArrowPatch(
            posA=start_xy,
            posB=end_xy,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            alpha=0.35,
            connectionstyle="arc3,rad=0.05",
            color="0.35",
            zorder=2,
        )
        ax.add_patch(arrow)

    for node in graph.nodes:
        x, y = node_locations[node]
        node_size = 130 + 38 * math.sqrt(max(strengths[node], 1))
        is_goalkeeper = node == goalkeeper_id

        ax.scatter(
            [x],
            [y],
            s=node_size * (1.45 if is_goalkeeper else 1.0),
            edgecolors="black",
            linewidths=1.3 if is_goalkeeper else 0.8,
            zorder=3,
        )

        ax.text(
            x,
            y,
            short_label(node_names.get(node, str(node))),
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=4,
        )

    opponent = str(goalkeeper_match_row["opponent"])
    match_date = str(goalkeeper_match_row["match_date"])
    ax.set_title(
        f"{goalkeeper_name} / {team_name} vs {opponent}\n"
        f"{match_date}, Completed-pass network "
        f"(edges with at least {minimum_edge_weight} passes)",
        fontsize=14,
    )

    safe_name = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", goalkeeper_name)
    output = FIGURE_DIR / (
        f"network_{safe_name}_{team_name}_{match_id}.png".replace(" ", "_")
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if SHOW_NETWORK_FIGURES:
        plt.show()
    plt.close(fig)
    return output


def select_representative_matches(
    eligible: pd.DataFrame,
    goalkeeper_match: pd.DataFrame,
) -> pd.DataFrame:
    """Select centroid-nearest goalkeeper, then their busiest passing match."""
    representatives = (
        eligible.sort_values("distance_to_centroid")
        .groupby(["cluster", "cluster_name"], as_index=False)
        .first()
    )

    selected_rows: list[pd.Series] = []

    for representative in representatives.itertuples(index=False):
        candidates = goalkeeper_match.loc[
            goalkeeper_match["goalkeeper_id"].eq(
                int(representative.goalkeeper_id)
            )
            & goalkeeper_match["team"].eq(str(representative.team))
        ].copy()

        if candidates.empty:
            continue

        selected = candidates.sort_values(
            ["gk_completed_passes", "stint_minutes"],
            ascending=[False, False],
        ).iloc[0].copy()
        selected["cluster"] = int(representative.cluster)
        selected["cluster_name"] = str(representative.cluster_name)
        selected_rows.append(selected)

    return pd.DataFrame(selected_rows)


def select_goalkeeper_network_matches(
    eligible: pd.DataFrame,
    goalkeeper_match: pd.DataFrame,
) -> pd.DataFrame:
    """Select one high-volume passing match for every eligible goalkeeper."""
    selected_rows: list[pd.Series] = []

    for goalkeeper in eligible.itertuples(index=False):
        candidates = goalkeeper_match.loc[
            goalkeeper_match["goalkeeper_id"].eq(int(goalkeeper.goalkeeper_id))
            & goalkeeper_match["team"].eq(str(goalkeeper.team))
        ].copy()

        if candidates.empty:
            continue

        selected = candidates.sort_values(
            ["gk_completed_passes", "stint_minutes"],
            ascending=[False, False],
        ).iloc[0].copy()
        selected["cluster"] = int(goalkeeper.cluster)
        selected["cluster_name"] = str(goalkeeper.cluster_name)
        selected["distance_to_centroid"] = float(
            goalkeeper.distance_to_centroid
        )
        selected_rows.append(selected)

    return pd.DataFrame(selected_rows)


# ---------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------

def main() -> dict[str, Any]:
    competition_id, season_id, competitions = resolve_target_competition()

    matches = retry_call(
        sb.matches,
        competition_id=competition_id,
        season_id=season_id,
    ).sort_values("match_date").reset_index(drop=True)

    print(f"공개 경기 수: {len(matches)}")
    if len(matches) != 51:
        print(
            "주의: 일반적인 Euro 2024 전체 경기 수(51)와 다릅니다. "
            "현재 Open Data 공개 범위를 확인하십시오."
        )

    matches.to_csv(
        TABLE_DIR / "euro2024_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    goalkeeper_match, error_log = build_goalkeeper_match_table(matches)

    goalkeeper_match.to_csv(
        TABLE_DIR / "goalkeeper_match_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_log.to_csv(
        TABLE_DIR / "processing_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    processing_qc = processing_quality_control(matches, goalkeeper_match, error_log)
    processing_qc.to_csv(
        TABLE_DIR / "processing_quality_control.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if STRICT_PROCESSING_VALIDATION and not bool(processing_qc.iloc[0]["complete"]):
        raise RuntimeError(
            "Incomplete match processing: inspect processing_quality_control.csv "
            "and processing_errors.csv before interpreting clusters."
        )

    goalkeeper_summary = aggregate_goalkeepers(goalkeeper_match)
    goalkeeper_summary.to_csv(
        TABLE_DIR / "goalkeeper_tournament_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        eligible,
        silhouette_df,
        profile_raw,
        profile_z,
        selected_k,
        cluster_model,
        scaler,
    ) = cluster_goalkeepers(goalkeeper_summary)

    eligible.to_csv(
        TABLE_DIR / "eligible_goalkeepers_clustered.csv",
        index=False,
        encoding="utf-8-sig",
    )
    silhouette_df.to_csv(
        TABLE_DIR / "silhouette_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profile_raw.to_csv(
        TABLE_DIR / "cluster_profiles_raw.csv",
        encoding="utf-8-sig",
    )
    profile_z.to_csv(
        TABLE_DIR / "cluster_profiles_z.csv",
        encoding="utf-8-sig",
    )

    # SCIE-oriented robustness and external-validation analyses.
    method_comparison = method_sensitivity(eligible, selected_k)
    pca_variance, pca_loadings = pca_diagnostics(eligible)
    # Render the principal figures before the longer bootstrap analyses.
    save_silhouette_plot(silhouette_df, selected_k)
    save_pca_plot(eligible, pca_variance)
    save_cluster_profile_heatmap(profile_z)
    save_involvement_longpass_plot(eligible)
    bootstrap_jaccard = bootstrap_cluster_jaccard(eligible, selected_k)
    leave_one_out, sample_sensitivity = subset_sensitivity(
        eligible, goalkeeper_summary, selected_k
    )
    feature_correlations = (
        eligible[CLUSTER_FEATURES].corr(method="spearman")
        .rename_axis("feature").reset_index()
    )
    feature_loo = leave_one_feature_out_sensitivity(eligible, selected_k)
    match_bootstrap = match_resampling_stability(
        goalkeeper_match, eligible, selected_k
    )
    supplementary_validation = supplementary_cluster_validation(eligible)
    context_availability = context_availability_table()
    context_descriptives = (
        goalkeeper_match.groupby(["competition_stage", "result"], dropna=False)
        .agg(
            goalkeeper_stints=("goalkeeper_id", "size"),
            goalkeeper_pass_completion=("pass_completion_pct", "mean"),
            team_completed_passes=("team_completed_passes", "mean"),
            goalkeeper_team_pass_share=("team_pass_share", "mean"),
        )
        .reset_index()
    )
    robustness_tables = {
        "cluster_internal_validity.csv": silhouette_df,
        "cluster_method_sensitivity.csv": method_comparison,
        "pca_explained_variance.csv": pca_variance,
        "pca_loadings.csv": pca_loadings,
        "cluster_bootstrap_jaccard.csv": bootstrap_jaccard,
        "cluster_leave_one_goalkeeper_out.csv": leave_one_out,
        "cluster_feature_correlations.csv": feature_correlations,
        "cluster_leave_one_feature_out.csv": feature_loo,
        "cluster_sample_sensitivity.csv": sample_sensitivity,
        "cluster_match_resampling.csv": match_bootstrap,
        "cluster_supplementary_validation.csv": supplementary_validation,
        "team_context_availability.csv": context_availability,
        "team_context_descriptives.csv": context_descriptives,
    }
    for filename, table in robustness_tables.items():
        table.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")

    cluster_summary = (
        eligible.groupby(["cluster", "cluster_name"])
        .agg(
            goalkeeper_count=("goalkeeper_id", "size"),
            total_minutes=("minutes", "sum"),
            team_pass_share=("team_pass_share", "mean"),
            betweenness=("betweenness_mean", "mean"),
            receiver_diversity=("receiver_diversity_mean", "mean"),
            entropy=("normalized_entropy", "mean"),
            long_pass_share=("long_pass_share", "mean"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(
        TABLE_DIR / "cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    representative_matches = select_representative_matches(
        eligible,
        goalkeeper_match,
    )
    representative_matches.to_csv(
        TABLE_DIR / "cluster_representative_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    goalkeeper_network_matches = select_goalkeeper_network_matches(
        eligible,
        goalkeeper_match,
    )
    goalkeeper_network_matches.to_csv(
        TABLE_DIR / "goalkeeper_network_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for row in goalkeeper_network_matches.to_dict("records"):
        try:
            plot_goalkeeper_match_network(pd.Series(row))
        except Exception as error:
            print(
                f"골키퍼 네트워크 그림 생성 실패 "
                f"({row.get('goalkeeper')}): {error}"
            )

    methodology = f"""Data source: StatsBomb Open Data
Competition: {TARGET_COMPETITION_NAME} {TARGET_SEASON_NAME}
Competition ID: {competition_id}
Season ID: {season_id}
Build-up play patterns: {BUILDUP_PLAY_PATTERNS}
Long-pass threshold: pass_length >= {LONG_PASS_THRESHOLD}
Minimum goalkeeper minutes: {MIN_TOTAL_MINUTES}
Sensitivity thresholds: {ROBUSTNESS_MINUTES}
Cluster features: {', '.join(CLUSTER_FEATURES)}
Selected cluster count: {selected_k}
Random state: {RANDOM_STATE}
Bootstrap replications: {N_BOOTSTRAP}

Important:
- The analytical network contains all Regular Play and From Goal Kick passes during
  each goalkeeper stint. It measures goalkeeper involvement in the team passing
  network and must not be described as a goalkeeper-centred possession sequence.
- Completed pass = pass_outcome is missing.
- Network edge direction = passer -> recipient.
- Network edge weight = completed-pass count.
- Betweenness uses inverse edge weight (1 / pass count) as distance.
- GK long-pass attempt share uses all goalkeeper pass attempts in the selected build-up sample.
- Clusters are interpreted as goalkeeper-team build-up system profiles, not isolated goalkeeper traits.
- Cluster count is assessed with silhouette, Calinski-Harabasz and Davies-Bouldin indices.
- Robustness checks include Ward, Gaussian mixture, cluster-bootstrap Jaccard,
  leave-one-goalkeeper-out, 180/270/360-minute samples, Angus Gunn exclusion,
  and match-within-goalkeeper resampling.
- Supplementary validation uses passing metrics excluded from clustering, Cliff's delta with
  percentile bootstrap confidence intervals, Mann-Whitney U tests and Holm adjustment.
- Unavailable contextual controls are explicitly documented in team_context_availability.csv.
- Cluster names are centroid-based interpretive labels, not validated universal categories.
- One high-volume passing match is saved for every eligible goalkeeper using a
  prespecified rule. Separately, the goalkeeper nearest each centroid and that
  goalkeeper's highest-completion match are saved as cluster representatives.
- Node size is weighted in-strength plus weighted out-strength; the goalkeeper node
  receives an additional 1.45 display multiplier solely for visual identification.
"""
    (OUTPUT_ROOT / "methodology.txt").write_text(
        methodology,
        encoding="utf-8",
    )

    print("\n분석 완료")
    print(f"결과 폴더: {OUTPUT_ROOT}")
    print(f"유형 수: {selected_k}")
    print("\n유형별 요약")
    print(cluster_summary.to_string(index=False))
    print("\n골키퍼별 유형")
    goalkeeper_summary_text = (
        eligible[
            [
                "goalkeeper",
                "team",
                "minutes",
                "cluster_name",
                "team_pass_share",
                "betweenness_mean",
                "receiver_diversity_mean",
                "normalized_entropy",
                "long_pass_share",
            ]
        ]
        .sort_values(["cluster_name", "team"])
        .to_string(index=False)
    )
    print(console_safe(goalkeeper_summary_text))

    return {
        "competitions": competitions,
        "matches": matches,
        "goalkeeper_match": goalkeeper_match,
        "goalkeeper_summary": goalkeeper_summary,
        "eligible": eligible,
        "silhouette_scores": silhouette_df,
        "cluster_profile_raw": profile_raw,
        "cluster_profile_z": profile_z,
        "cluster_summary": cluster_summary,
        "cluster_method_sensitivity": method_comparison,
        "pca_explained_variance": pca_variance,
        "pca_loadings": pca_loadings,
        "processing_quality_control": processing_qc,
        "cluster_bootstrap_jaccard": bootstrap_jaccard,
        "cluster_leave_one_out": leave_one_out,
        "cluster_feature_correlations": feature_correlations,
        "cluster_leave_one_feature_out": feature_loo,
        "cluster_sample_sensitivity": sample_sensitivity,
        "cluster_match_resampling": match_bootstrap,
        "cluster_supplementary_validation": supplementary_validation,
        "team_context_availability": context_availability,
        "representative_matches": representative_matches,
        "goalkeeper_network_matches": goalkeeper_network_matches,
        "selected_k": selected_k,
        "cluster_model": cluster_model,
        "scaler": scaler,
        "output_root": OUTPUT_ROOT,
    }


if __name__ == "__main__":
    RESULTS = main()
