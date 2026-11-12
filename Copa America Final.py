"""지정 원고의 표 1~7, 그림 1~2 및 관측량·상대진영 산점도를 생성하는 독립 실행 코드.
대상: IJPAS_Copa_2024_EN_Discussion_Integrated (1).docx
실행: python "Copa America 논문 결과 전용.py"
출력: 코드 옆의 '논문 결과 표 및 그림' 폴더 (CSV 7개, Excel 1개, PNG 3개).
기존 copa_america2024_match_reanalysis 원자료 캐시와 review_sources를 사용합니다.
다른 프로젝트 Python 파일이나 원본 DOCX 없이 실행하며, 중간 결과는 자동 정리합니다.
원고 수치는 분석 입력이 아닌 검증 기준입니다. 계산값이 원고와 다르면 저장 전에 중단합니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
import math
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.patches
from mplsoccer import Pitch
from statsbombpy import sb

from tqdm.auto import tqdm

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

# Latin names require accented glyphs even when koreanize_matplotlib is installed.
plt.rcParams["font.family"] = ["DejaVu Sans", "Malgun Gothic"]

import json
import platform
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

FEATURES = ['team_pass_share', 'betweenness', 'receiver_diversity', 'normalized_entropy', 'long_pass_share']
CONTEXT = ['leading_time_share', 'trailing_time_share', 'gk_pressure_share', 'goal_kick_share']
MIN_MINUTES, MIN_ATTEMPTS, MIN_COMPLETED, MIN_CLUSTERS = 45., 10, 10, 6
BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = 1999, 20261015


def analysis_role(model_id):
    return 'supplementary_exploratory' if model_id == 'context_entropy' else 'main_exploratory'


def column(frame, name):
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def event_window(events, start, end):
    """Half-open intervals; the final match timestamp belongs to the last stint."""
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError('Invalid stint boundaries')
    regular = events.loc[events.period.isin([1, 2, 3, 4])]
    final = regular._event_minute.max()
    right = regular._event_minute.le(end) if np.isclose(end, final, atol=1e-9, rtol=0) else regular._event_minute.lt(end)
    return regular.loc[regular._event_minute.ge(start) & right].copy()


def ordered_events(events):
    regular = events.loc[events.period.isin([1, 2, 3, 4])].copy()
    keys = ['period', '_event_minute'] + (['index'] if 'index' in regular else [])
    return regular.sort_values(keys, kind='stable').reset_index(drop=True)


def score_deltas(ordered, team):
    goal = ordered.type.eq('Shot') & column(ordered, 'shot_outcome').eq('Goal')
    own = ordered.type.eq('Own Goal Against')
    # Own Goal For mirrors Own Goal Against: never add it a second time.
    return np.where(goal, np.where(ordered.team.eq(team), 1, -1),
                    np.where(own, np.where(ordered.team.eq(team), -1, 1), 0))


def interval_exposure(times, changes, start, end):
    times, changes = np.asarray(times, float), np.asarray(changes, float)
    points = [start, *sorted(set(times[(times > start) & (times < end)])), end]
    result = {'positive': 0., 'zero': 0., 'negative': 0.}
    for left, right in zip(points[:-1], points[1:]):
        state = changes[times <= left].sum()
        result['positive' if state > 0 else 'negative' if state < 0 else 'zero'] += right-left
    return result


def valid_endpoint(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return False
    try:
        x, y = float(value[0]), float(value[1])
        return bool(np.isfinite([x, y]).all() and 0 <= x <= 120 and 0 <= y <= 80)
    except (TypeError, ValueError):
        return False


def add_event_context(events, team, goalkeeper_id, start, end, patterns):
    ordered = ordered_events(events)
    delta = score_deltas(ordered, team)
    ordered['_score_before'] = np.cumsum(delta)-delta
    exposure = interval_exposure(ordered._event_minute, delta, start, end)
    result = {}
    for state, key in [('leading', 'positive'), ('drawing', 'zero'), ('trailing', 'negative')]:
        result[state+'_minutes'] = exposure[key]
        result[state+'_time_share'] = exposure[key]/(end-start)
    window = event_window(ordered, start, end)
    passes = window.loc[window.type.eq('Pass') & window.team.eq(team)]
    if patterns is not None:
        passes = passes.loc[passes.play_pattern.isin(patterns)]
    gk = passes.loc[passes.player_id.eq(goalkeeper_id)]
    for state, mask in [('leading', gk._score_before.gt(0)), ('drawing', gk._score_before.eq(0)), ('trailing', gk._score_before.lt(0))]:
        result[f'gk_{state}_attempts'] = int(mask.sum())
    result['pressure_field_available'] = 'under_pressure' in events
    # Sparse True field: blank CELLS are unflagged; absent FIELD is unavailable.
    result['gk_pressured_attempts'] = int(gk.under_pressure.eq(True).sum()) if 'under_pressure' in gk else np.nan
    result['gk_pressure_share'] = result['gk_pressured_attempts']/len(gk) if len(gk) else np.nan
    result['gk_goal_kick_attempts'] = int(gk.play_pattern.eq('From Goal Kick').sum())
    result['goal_kick_share'] = result['gk_goal_kick_attempts']/len(gk) if len(gk) else np.nan
    red = column(ordered, 'foul_committed_card').isin(['Red Card', 'Second Yellow']) | column(ordered, 'bad_behaviour_card').isin(['Red Card', 'Second Yellow'])
    dismissed = ordered.loc[red & ordered.player_id.notna()].drop_duplicates(['team', 'player_id'])
    manpower = interval_exposure(dismissed._event_minute, np.where(dismissed.team.eq(team), -1, 1), start, end)
    result['man_advantage_time_share'] = manpower['positive']/(end-start)
    result['man_disadvantage_time_share'] = manpower['negative']/(end-start)
    result['dismissal_recorded_in_match'] = bool(len(dismissed))
    tactics = ordered.loc[ordered.team.eq(team) & ordered.type.isin(['Starting XI', 'Tactical Shift'])]
    formations = column(tactics, 'tactics').map(lambda x: x.get('formation') if isinstance(x, dict) else None)
    prior = formations.loc[tactics._event_minute.le(start)].dropna()
    result['formation_at_stint_start'] = str(prior.iloc[-1]) if len(prior) else None
    result['tactical_shifts_in_stint'] = int((window.team.eq(team) & window.type.eq('Tactical Shift')).sum())
    # Concurrent territory, not progression AFTER GK involvement or escape success.
    outfield = passes.loc[passes.player_id.notna() & passes.player_id.ne(goalkeeper_id)
                          & passes.pass_outcome.isna() & passes.pass_recipient_id.notna()]
    endpoints = column(outfield, 'pass_end_location')
    known = endpoints.map(valid_endpoint).astype(bool)
    result['outfield_completed_passes'] = len(outfield)
    result['outfield_known_endpoints'] = int(known.sum())
    result['outfield_opponent_half_passes'] = int(endpoints.loc[known].map(lambda x: x[0] >= 60).sum())
    result['outfield_endpoint_coverage'] = float(known.mean()) if len(outfield) else np.nan
    result['outfield_opponent_half_share'] = result['outfield_opponent_half_passes']/known.sum() if known.any() else np.nan
    shots = window.loc[window.team.eq(team) & window.type.eq('Shot')]
    xg = pd.to_numeric(column(shots, 'shot_statsbomb_xg'), errors='coerce')
    result['team_shots'] = len(shots)
    result['team_xg_coverage'] = float(xg.notna().mean()) if len(shots) else 1.
    result['team_xg'] = float(xg.sum()) if xg.notna().all() else np.nan
    result['team_shots_per90'] = len(shots)*90/(end-start)
    result['team_xg_per90'] = result['team_xg']*90/(end-start)
    return result


def validate_match_context(events, match_row, metrics):
    ordered = ordered_events(events)
    goals = ordered.loc[(ordered.type.eq('Shot') & column(ordered, 'shot_outcome').eq('Goal')) | ordered.type.eq('Own Goal Against')]
    rows = []
    for team, score in [(str(match_row.home_team), match_row.home_score), (str(match_row.away_team), match_row.away_score)]:
        goals_for = int(((goals.type.eq('Shot') & goals.team.eq(team)) | (goals.type.eq('Own Goal Against') & goals.team.ne(team))).sum())
        d = metrics.loc[metrics.team.eq(team)]
        exposure_error = float((d[['leading_minutes', 'drawing_minutes', 'trailing_minutes']].sum(axis=1)-d.stint_minutes).abs().max())
        pass_error = int((d[['gk_leading_attempts', 'gk_drawing_attempts', 'gk_trailing_attempts']].sum(axis=1)-d.gk_pass_attempts).abs().sum())
        shots = int((ordered.team.eq(team) & ordered.type.eq('Shot')).sum())
        rows.append(dict(match_id=match_row.match_id, team=team, recorded_goals=int(score), reconstructed_goals=goals_for,
                         exposure_error_minutes=exposure_error, pass_state_count_error=pass_error,
                         raw_shots=shots, stint_shots=int(d.team_shots.sum()),
                         passed=goals_for == int(score) and exposure_error < 1e-8 and pass_error == 0 and shots == int(d.team_shots.sum())))
    return rows


def attach_opponent_strength(frame, ratings_path=None):
    d = frame.copy()
    if ratings_path is None:
        return d, 'unavailable; no pre-tournament ratings supplied'
    ratings = pd.read_csv(ratings_path)
    required = {'team', 'rating', 'rating_date', 'source'}
    if not required.issubset(ratings):
        raise ValueError(f'Ratings CSV requires {sorted(required)}')
    if ratings.team.duplicated().any():
        raise ValueError('Ratings must contain one row per team')
    dates = pd.to_datetime(ratings.rating_date, errors='coerce', utc=True)
    values = pd.to_numeric(ratings.rating, errors='coerce')
    if dates.isna().any() or dates.ge(pd.Timestamp('2024-06-20', tz='UTC')).any():
        raise ValueError('Every rating date must precede 2024-06-20')
    if not np.isfinite(values).all() or ratings.source.fillna('').str.strip().eq('').any():
        raise ValueError('Ratings must be finite and have a source')
    missing = (set(d.team) | set(d.opponent))-set(ratings.team)
    if missing:
        raise ValueError(f'Missing team ratings: {sorted(missing)}')
    mapping = pd.Series(values.to_numpy(), index=ratings.team)
    d['team_pre_elo'] = d.team.map(mapping)
    d['opponent_pre_elo'] = d.opponent.map(mapping)
    d['strength_difference'] = d.team_pre_elo-d.opponent_pre_elo
    d['strength_difference_100'] = d.strength_difference/100
    d['rating_date'] = d.team.map(ratings.set_index('team').rating_date)
    d['rating_source'] = d.team.map(ratings.set_index('team').source)
    return d, 'available; frozen pre-tournament ratings dated '+', '.join(sorted(ratings.rating_date.unique()))


def fixed_effect_model(frame, outcome, predictor, controls=()):
    return fit_fixed_effects(frame, outcome, [predictor, *controls], [predictor])[0][0]


def fit_fixed_effects(frame, outcome, predictors, focal_terms=None):
    """Pair FE, team+match CR1 sandwich. Singular/unsupported fits are explicit."""
    focal_terms = predictors if focal_terms is None else focal_terms
    names = list(dict.fromkeys(predictors))
    d = frame.copy()
    if 'team' not in d:  # supports synthetic pair-only regression tests
        d['team'] = d.pair_id
    required = [outcome, *names, 'pair_id', 'team', 'match_id']
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    complete_n = len(d)
    d = d.loc[d.groupby('pair_id').pair_id.transform('size').gt(1)].copy()
    info = dict(outcome=outcome, complete_case_n=complete_n, n=len(d), singleton_rows_removed=complete_n-len(d),
                pairs=d.pair_id.nunique(), teams=d.team.nunique(), matches=d.match_id.nunique(),
                covariance='two-way CR1: national team + match', controls=';'.join(names), status='exploratory')
    def stop(status):
        return [{**info, 'predictor': name, 'status': status} for name in focal_terms], d
    if min(info['teams'], info['matches']) < MIN_CLUSTERS:
        return stop('insufficient clusters')
    within = d[names]-d.groupby('pair_id')[names].transform('mean')
    if within.std().le(1e-12).any():
        return stop('rank deficient: predictor has no within-pair variation')
    x = pd.concat([d[names].astype(float), pd.get_dummies(d.pair_id, prefix='pair', drop_first=True, dtype=float)], axis=1)
    x = sm.add_constant(x, has_constant='add')
    n, k = x.shape
    info['residual_df'], info['condition_number'] = n-k, float(np.linalg.cond(x.to_numpy()))
    if n <= k+5 or np.linalg.matrix_rank(x.to_numpy()) < k:
        return stop('insufficient residual df or rank deficient')
    if info['condition_number'] > 1e12:
        return stop('numerically ill-conditioned design; estimate withheld')
    fit = sm.OLS(d[outcome].astype(float), x).fit()
    groups = np.column_stack([pd.factorize(d.team)[0], pd.factorize(d.match_id)[0]])
    covariance, _, _ = cov_cluster_2groups(fit, groups, use_correction=True)
    info['covariance_min_eigenvalue'] = float(np.linalg.eigvalsh(covariance).min())
    focal_indices = [x.columns.get_loc(name) for name in focal_terms]
    slope_indices = [x.columns.get_loc(name) for name in names]
    focal_cov = covariance[np.ix_(focal_indices, focal_indices)]
    info['focal_covariance_min_eigenvalue'] = float(np.linalg.eigvalsh(focal_cov).min())
    info['slope_covariance_min_eigenvalue'] = float(np.linalg.eigvalsh(covariance[np.ix_(slope_indices, slope_indices)]).min())
    info['focal_covariance_positive_definite'] = bool(info['focal_covariance_min_eigenvalue'] > 0)
    info['focal_covariance_json'] = json.dumps({'terms': focal_terms, 'matrix': focal_cov.tolist()})
    info['strength_adjusted'] = 'strength_difference_100' in names
    info['strength_coefficient_per100'] = float(fit.params.get('strength_difference_100', np.nan))
    df = min(info['teams'], info['matches'])-1
    rows = []
    for name in focal_terms:
        variance = covariance[x.columns.get_loc(name), x.columns.get_loc(name)]
        row = {**info, 'predictor': name, 'estimate': float(fit.params[name]), 'reference_df': df}
        if variance <= 0 or not np.isfinite(variance):
            row['status'] = 'nonpositive two-way variance; interval withheld'
        else:
            se = float(np.sqrt(variance))
            row.update(se=se, ci_low=fit.params[name]-t.ppf(.975, df)*se,
                       ci_high=fit.params[name]+t.ppf(.975, df)*se, p_unadjusted=2*t.sf(abs(fit.params[name]/se), df))
        row['predictor_increment'] = .1 if name.endswith('_share') else 1.
        for key in ['estimate', 'ci_low', 'ci_high']:
            row[key+'_per_increment'] = row.get(key, np.nan)*row['predictor_increment']
        rows.append(row)
    return rows, d


def analysis_samples(d):
    common = d.stint_minutes.ge(MIN_MINUTES) & d.team_completed_passes.gt(0)
    samples = {'primary': common,
            'minutes60': d.stint_minutes.ge(60) & d.team_completed_passes.gt(0),
            'attempts10_completed5': common & d.gk_pass_attempts.ge(10) & d.gk_completed_passes.ge(5),
            'attempts20_completed10': common & d.gk_pass_attempts.ge(20) & d.gk_completed_passes.ge(10),
            'no_recorded_dismissals': common & ~d.dismissal_recorded_in_match.astype(bool)}
    if 'strength_difference_100' in d:
        samples['without_opponent_strength'] = common
    return samples


def model_specs(strength_available):
    context = CONTEXT + (['strength_difference_100'] if strength_available else [])
    return [('context_participation', 'team_pass_share', context, CONTEXT, 'participation'),
            ('context_long_distribution', 'long_pass_share', context, CONTEXT, 'long'),
            ('context_entropy', 'normalized_entropy', [*context, 'log_completed_passes'], CONTEXT, 'entropy'),
            ('concurrent_territory', 'outfield_opponent_half_share', ['team_pass_share', *context], ['team_pass_share'], 'territory'),
            ('concurrent_shots', 'team_shots_per90', ['team_pass_share', *context], ['team_pass_share'], 'participation'),
            ('concurrent_xg', 'team_xg_per90', ['team_pass_share', *context], ['team_pass_share'], 'xg')]


def eligible_for_metric(sample, metric):
    if metric == 'long':
        return sample.loc[sample.gk_pass_length_observed.ge(MIN_ATTEMPTS) & sample.gk_pass_length_coverage.ge(.95)]
    if metric == 'entropy':
        return sample.loc[sample.gk_completed_passes.ge(MIN_COMPLETED)]
    if metric == 'territory':
        return sample.loc[sample.outfield_endpoint_coverage.ge(.95) & sample.outfield_known_endpoints.ge(10)]
    if metric == 'xg':
        return sample.loc[sample.team_xg_coverage.eq(1)]
    return sample


def weighted_within_coefficients(frame, outcome, predictors, weights):
    """FWL weighted pair demeaning; multiplicities preserve the crossed draw."""
    weights = np.asarray(weights, float)
    if len(weights) != len(frame) or (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError('Invalid resampling weights')
    selected = weights > 0
    d = frame.loc[selected].copy()
    w = weights[selected]
    if not len(d):
        raise ValueError('empty resample')
    repeated = d.groupby('pair_id').pair_id.transform('size').gt(1).to_numpy()
    d, w = d.loc[repeated], w[repeated]
    if min(d.team.nunique(), d.match_id.nunique()) < MIN_CLUSTERS:
        raise ValueError('too few represented clusters')
    codes, groups = pd.factorize(d.pair_id)
    raw = d[[outcome, *predictors]].to_numpy(float)
    totals, counts = np.zeros((len(groups), raw.shape[1])), np.zeros(len(groups))
    np.add.at(totals, codes, raw*w[:, None])
    np.add.at(counts, codes, w)
    centered = raw-totals[codes]/counts[codes, None]
    y, x = centered[:, 0]*np.sqrt(w), centered[:, 1:]*np.sqrt(w[:, None])
    if len(d)-len(groups) <= len(predictors)+1 or np.linalg.matrix_rank(x) < len(predictors):
        raise ValueError('insufficient within-pair rank')
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    if not np.isfinite(coefficients).all() or np.linalg.cond(x) > 1e12:
        raise ValueError('ill-conditioned resample')
    return coefficients


def crossed_cluster_bootstrap(used, outcome, predictors, focal, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Pigeonhole sensitivity: independently resample teams and match IDs.

    Not an observation bootstrap, not a restricted wild bootstrap, and not an
    exact few-cluster correction. Percentile intervals are sensitivity outputs.
    """
    order = ['team', 'match_id', 'pair_id'] + (['start_minute'] if 'start_minute' in used else [])
    used = used.sort_values(order, kind='stable').reset_index(drop=True)
    rng = np.random.default_rng(seed)
    team_codes, teams = pd.factorize(used.team)
    match_codes, matches = pd.factorize(used.match_id)
    rows, draws = [], []
    if min(len(teams), len(matches)) < MIN_CLUSTERS:
        return [dict(predictor=name, attempted=replicates, valid=0, status='insufficient original clusters') for name in focal], draws
    values = []
    for draw in range(replicates):
        team_count = rng.multinomial(len(teams), np.full(len(teams), 1/len(teams)))
        match_count = rng.multinomial(len(matches), np.full(len(matches), 1/len(matches)))
        weights = team_count[team_codes]*match_count[match_codes]
        try:
            coef = weighted_within_coefficients(used, outcome, predictors, weights)
        except ValueError as error:
            draws.append(dict(draw=draw, predictor='', estimate=np.nan, status=str(error)))
            continue
        values.append(coef)
        for name in focal:
            draws.append(dict(draw=draw, predictor=name, estimate=coef[predictors.index(name)], status='valid'))
    values = np.asarray(values)
    for name in focal:
        row = dict(predictor=name, attempted=replicates, valid=len(values), seed=seed,
                   method='team x match pigeonhole percentile sensitivity',
                   status='sensitivity only' if len(values) >= .8*replicates else 'too many failed resamples; interval withheld')
        if len(values) >= .8*replicates:
            array = values[:, predictors.index(name)]
            row.update(percentile_low=np.quantile(array, .025), percentile_high=np.quantile(array, .975),
                       bootstrap_sd=np.std(array, ddof=1), fraction_positive=np.mean(array > 0))
        rows.append(row)
    return rows, draws


def summarize_leave_one_team_out(lto, primary_models):
    rows = []
    for (model, term), group in lto.groupby(['model_id', 'predictor']):
        original = primary_models.loc[primary_models.model_id.eq(model) & primary_models.predictor.eq(term), 'estimate']
        baseline = original.iloc[0] if len(original) else np.nan
        estimates = pd.to_numeric(group.get('estimate'), errors='coerce').dropna()
        changes = int((np.sign(estimates) != np.sign(baseline)).sum()) if np.isfinite(baseline) else np.nan
        rows.append(dict(model_id=model, predictor=term, original_estimate=baseline,
                         omissions=len(group), estimable=len(estimates), min_estimate=estimates.min(), max_estimate=estimates.max(),
                         sign_changes=changes, stable_direction=bool(len(estimates) == len(group) and changes == 0),
                         interpretation='direction stability only; not proof of robustness or significance'))
    return pd.DataFrame(rows)


def create_figures(d, primary, figures):
    figures.mkdir(parents=True, exist_ok=True)
    labels = ['GK share of completed passes', 'Weighted betweenness', 'Observed recipients', 'Recipient entropy', 'Long-pass share']
    groups = list(primary.groupby('pair_id', sort=True))
    fig, axes = plt.subplots(1, 5, figsize=(20, max(6, len(groups)*.35)), sharey=True)
    for ax, feature, label in zip(axes, FEATURES, labels):
        for j, (_, g) in enumerate(groups):
            values = g[feature].dropna().to_numpy()
            if len(values):
                ax.scatter(values, j+np.linspace(-.12, .12, len(values)), s=18, color='#3577a1', alpha=.75)
                ax.scatter(values.mean(), j, s=40, marker='D', color='#bd6a22', zorder=3)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(groups)), [f'{g.team.iloc[0]} / {g.goalkeeper.iloc[0]}' for _, g in groups], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Match/stint variation within goalkeeper-team combinations\nBlue: observations; orange: unweighted pair mean. Descriptive duration-eligible stints.', fontsize=12)
    fig.tight_layout()
    fig.savefig(figures/'01_within_pair_variation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(d.gk_completed_passes, d.normalized_entropy, c=d.primary_eligible, cmap='cividis', s=28)
    axes[0].axvline(MIN_COMPLETED, ls='--', color='grey')
    axes[0].set(xlabel='Completed goalkeeper passes', ylabel='Normalized recipient entropy')
    axes[1].scatter(primary.team_pass_share, primary.outfield_opponent_half_share, s=28, alpha=.75)
    axes[1].set(xlabel='GK share of team completed passes', ylabel='Outfield opponent-half endpoint share')
    fig.suptitle('Observation volume and concurrent territory (raw descriptive associations)')
    fig.tight_layout()
    fig.savefig(figures/'02_observation_volume_and_territory.png', dpi=200)
    plt.close(fig)


def run_match_analysis(metrics, table_dir, output_root, ratings_path=None):
    table_dir, output_root = Path(table_dir), Path(output_root)
    table_dir.mkdir(parents=True, exist_ok=True)
    d, strength_status = attach_opponent_strength(metrics, ratings_path)
    d['pair_id'] = d.team.astype(str)+':'+d.goalkeeper_id.astype(str)
    d['observation_id'] = d.match_id.astype(str)+':'+d.pair_id+':'+d.start_minute.astype(str)
    if d.observation_id.duplicated().any():
        raise ValueError('Duplicate goalkeeper-stint records')
    d['log_completed_passes'] = np.log(d.gk_completed_passes.replace(0, np.nan))
    masks = analysis_samples(d)
    d['primary_eligible'] = masks['primary']
    d['exclusion_reason'] = np.select([d.stint_minutes.lt(MIN_MINUTES), d.team_completed_passes.le(0)],
                                      ['stint below 45 minutes', 'no observed team completed passes'], default='included')
    d['entropy_model_eligible'] = masks['primary'] & d.gk_completed_passes.ge(MIN_COMPLETED)
    primary = d.loc[d.primary_eligible].copy()
    for feature in FEATURES:
        d[feature+'_pair_mean'] = d.pair_id.map(primary.groupby('pair_id')[feature].mean())
        d[feature+'_within_pair'] = d[feature]-d[feature+'_pair_mean']
    primary = d.loc[d.primary_eligible].copy()
    variability = primary.groupby(['pair_id', 'team', 'goalkeeper'])[FEATURES].agg(['count', 'mean', 'std', 'min', 'max'])
    variability.columns = ['_'.join(c) for c in variability.columns]
    tables = {'goalkeeper_match_analysis.csv': d, 'primary_match_sample.csv': primary,
              'match_feature_descriptives.csv': primary[FEATURES].describe().T.rename_axis('feature').reset_index(),
              'pair_variability.csv': variability.reset_index(),
              'between_pair_correlations_descriptive.csv': primary.groupby('pair_id')[FEATURES].mean().corr().rename_axis('feature').reset_index(),
              'within_pair_correlations_descriptive.csv': primary[[f+'_within_pair' for f in FEATURES]].corr().rename_axis('feature').reset_index()}
    rows, memberships, attrition, specifications, lto, bootstrap, bootstrap_draws = [], [], [], [], [], [], []
    for sample_name, mask in masks.items():
        sample = d.loc[mask]
        attrition.append(dict(sample=sample_name, n=len(sample), pairs=sample.pair_id.nunique(), teams=sample.team.nunique(), matches=sample.match_id.nunique()))
        for spec_index, (model_id, outcome, predictors, focal, metric) in enumerate(model_specs('strength_difference_100' in d and sample_name != 'without_opponent_strength')):
            chosen = eligible_for_metric(sample, metric)
            fitted, used = fit_fixed_effects(chosen, outcome, predictors, focal)
            rows.extend({**r, 'sample': sample_name, 'model_id': model_id} for r in fitted)
            statuses = ';'.join(sorted(set(r['status'] for r in fitted)))
            memberships.extend({**r, 'sample': sample_name, 'model_id': model_id, 'fit_status': statuses}
                               for r in used[['observation_id', 'pair_id', 'match_id', 'team']].to_dict('records'))
            if sample_name == 'primary':
                specifications.append(dict(model_id=model_id, outcome=outcome, predictors=';'.join(predictors), focal_terms=';'.join(focal), metric_sample=metric))
                for team in sorted(used.team.unique()):
                    reduced, _ = fit_fixed_effects(chosen.loc[chosen.team.ne(team)], outcome, predictors, focal)
                    lto.extend({**r, 'omitted_team': team, 'model_id': model_id} for r in reduced)
                if all('estimate' in row for row in fitted):
                    summaries, draws = crossed_cluster_bootstrap(used, outcome, predictors, focal, seed=BOOTSTRAP_SEED+spec_index)
                    bootstrap.extend({**r, 'model_id': model_id} for r in summaries)
                    bootstrap_draws.extend({**r, 'model_id': model_id} for r in draws)
    models = pd.DataFrame(rows)
    models['analysis_role'] = models.model_id.map(analysis_role)
    if 'p_unadjusted' not in models:
        models['p_unadjusted'] = np.nan
    models['p_holm_primary_family'] = np.nan
    for family in ['context_', 'concurrent_']:
        mask = models['sample'].eq('primary') & models.model_id.str.startswith(family)
        if mask.any():
            adjusted = pd.Series(multipletests(models.loc[mask, 'p_unadjusted'].fillna(1), method='holm')[1], index=models.index[mask])
            models.loc[mask, 'p_holm_primary_family'] = adjusted.where(models.loc[mask, 'p_unadjusted'].notna())
    tables.update({'exploratory_pair_fixed_effects.csv': models, 'model_sample_membership.csv': pd.DataFrame(memberships),
                   'sample_sensitivity_counts.csv': pd.DataFrame(attrition), 'model_specifications.csv': pd.DataFrame(specifications),
                   'leave_one_team_out.csv': pd.DataFrame(lto)})
    # Preserve ALL previous coefficients, samples, seeds, and the 12+3 Holm
    # families. Separating the presentation does not redefine the test family.
    tables['main_inference_results.csv'] = models.loc[models['sample'].eq('primary') & models.analysis_role.eq('main_exploratory')].copy()
    tables['supplementary_entropy_results.csv'] = models.loc[models.model_id.eq('context_entropy')].copy()
    diagnostic_columns = ['sample', 'model_id', 'analysis_role', 'outcome', 'n', 'complete_case_n', 'singleton_rows_removed', 'pairs', 'teams', 'matches',
                          'residual_df', 'condition_number', 'covariance_min_eigenvalue', 'slope_covariance_min_eigenvalue',
                          'focal_covariance_min_eigenvalue', 'focal_covariance_positive_definite', 'focal_covariance_json',
                          'strength_adjusted', 'strength_coefficient_per100', 'controls', 'status']
    tables['model_diagnostics.csv'] = models.reindex(columns=diagnostic_columns).drop_duplicates(['sample', 'model_id'])
    tables['crossed_cluster_bootstrap.csv'] = pd.DataFrame(bootstrap)
    tables['crossed_cluster_bootstrap_draws.csv'] = pd.DataFrame(bootstrap_draws)
    unique_draws = pd.DataFrame(bootstrap_draws).drop_duplicates(['model_id', 'draw'])
    tables['bootstrap_failure_reasons.csv'] = unique_draws.groupby(['model_id', 'status']).size().rename('count').reset_index()
    bootstrap_summary = pd.DataFrame(bootstrap).drop_duplicates('model_id')[['model_id', 'attempted', 'valid', 'status']].copy()
    bootstrap_summary['valid_fraction'] = bootstrap_summary.valid/bootstrap_summary.attempted
    bootstrap_summary['analysis_role'] = bootstrap_summary.model_id.map(analysis_role)
    tables['bootstrap_model_summary.csv'] = bootstrap_summary.merge(
        tables['model_diagnostics.csv'].loc[tables['model_diagnostics.csv']['sample'].eq('primary'), ['model_id', 'n', 'pairs', 'teams', 'matches']], on='model_id')
    tables['leave_one_team_out_summary.csv'] = summarize_leave_one_team_out(pd.DataFrame(lto), models.loc[models['sample'].eq('primary')])
    tables['opponent_strength_match_check.csv'] = d.reindex(columns=['observation_id', 'match_id', 'team', 'opponent',
                                                                 'team_pre_elo', 'opponent_pre_elo', 'strength_difference', 'strength_difference_100', 'rating_date', 'rating_source'])
    feasibility = []
    for name in ['stint_minutes', 'gk_pass_attempts', 'gk_completed_passes', *CONTEXT, 'man_disadvantage_time_share',
                 'formation_at_stint_start', 'outfield_endpoint_coverage', 'team_shots', 'team_xg', 'gk_pass_length_coverage']:
        values = d[name]
        feasibility.append(dict(variable=name, observed=values.notna().sum(), missing=values.isna().sum(), distinct=values.nunique(),
                                nonzero=pd.to_numeric(values, errors='coerce').fillna(0).ne(0).sum()))
    tables['reanalysis_feasibility.csv'] = pd.DataFrame(feasibility)
    tables['context_availability.csv'] = pd.DataFrame([
        ('score state', 'measured and reconciled', 'State exposure; final result separate'),
        ('pressure', 'sparse event annotation', 'Blank cells unflagged; absent field unavailable; not pressing tactics'),
        ('goal kick', 'measured', 'Composition within selected play patterns'),
        ('opponent strength', strength_status, 'Pre-tournament team-minus-opponent Elo /100; own team level absorbed by pair FE'),
        ('dismissals', 'measured', 'Player red/second-yellow exposure; sensitivity excludes affected matches'),
        ('formation', 'descriptive only', 'Stint-start formation and shifts do not fully measure tactics'),
        ('territory', 'concurrent proxy', 'Outfield endpoints; not progression after GK involvement'),
        ('shots and xG', 'supplementary concurrent outcomes', 'Same-stint rates; no causal/predictive interpretation')
    ], columns=['factor', 'status', 'note'])
    tables['variable_dictionary.csv'] = variable_dictionary()
    for name, table in tables.items():
        table.to_csv(table_dir/name, index=False, encoding='utf-8-sig')
    manifest = dict(analysis='match context and continuous network characteristics', inference='exploratory; after-data redesign',
                    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, statsmodels=sm.__version__,
                    primary_min_minutes=MIN_MINUTES, distribution_min_known_lengths=MIN_ATTEMPTS, entropy_min_completed=MIN_COMPLETED,
                    opponent_strength=strength_status, observations=len(d), primary_observations=len(primary),
                    buildup_play_patterns=['Regular Play', 'From Goal Kick'],
                    long_pass_threshold=30, long_pass_threshold_units='StatsBomb pitch units, not metres',
                    cluster_dimensions=['national team', 'match'], fixed_effect='goalkeeper-team pair',
                    bootstrap={'method': 'team x match pigeonhole percentile sensitivity', 'replicates': BOOTSTRAP_REPLICATES,
                               'base_seed': BOOTSTRAP_SEED, 'minimum_valid_fraction': .8, 'not_exact_few_cluster_correction': True},
                    research_scope='within-pair variation and exploratory associations; no incremental network value claim',
                    entropy_role='descriptive variation retained; regression supplementary exploratory only',
                    multiplicity='unchanged 12 context coefficients (including 4 supplementary entropy) + 3 concurrent coefficients',
                    output_tables=sorted(tables), clustering=False, pca=False)
    (output_root/'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'재분석 완료: {d.match_id.nunique()}경기, {len(d)}구간, 주 기술표본 {len(primary)}구간. 결과: {output_root}')
    return tables


def variable_dictionary():
    return pd.DataFrame([
        ('team_pass_share', 'GK completed / team completed passes', 'Identifiable recipients; 0-1'),
        ('betweenness', 'Normalized directed weighted betweenness; edge distance=1/count', 'Observed network nodes vary; descriptive'),
        ('receiver_diversity', 'Distinct recipients of GK successful passes', 'Depends on observation volume'),
        ('normalized_entropy', '-sum(p*log(p))/log(observed recipients)', 'One recipient=0; no completions=missing; model requires >=10 completions'),
        ('long_pass_share', 'Length>=30 attempts / attempts with known valid length', 'StatsBomb units, not metres; model requires >=10 known lengths, >=95% coverage'),
        ('leading_time_share', 'Time ahead / stint elapsed minutes', 'Includes stoppage time, excludes breaks/shootout'),
        ('trailing_time_share', 'Time behind / stint elapsed minutes', 'Drawing is reference in adjusted context models'),
        ('gk_pressure_share', 'Flagged GK attempts / selected GK attempts', 'Sparse event annotation, not team tactics'),
        ('goal_kick_share', 'From Goal Kick GK attempts / selected GK attempts', 'Composition in selected patterns'),
        ('strength_difference_100', '(Frozen team Elo - frozen opponent Elo)/100', '2024-06-19 provider snapshot; own-team baseline absorbed by pair FE'),
        ('outfield_opponent_half_share', 'Outfield successful endpoints x>=60 / valid known endpoints', 'Concurrent territory; >=10 endpoints and >=95% coverage for model'),
        ('team_shots_per90', 'All team shots *90 / stint minutes', 'All patterns, excludes shootout'),
        ('team_xg_per90', 'Sum all team shot xG *90 / stint minutes', 'Missing if any shot xG unavailable; no shots=0'),
        ('venue', 'Home/away metadata designation', 'Administrative at neutral tournament, not home advantage'),
        ('*_pair_mean', 'Unweighted duration-eligible pair mean', 'Descriptive between-pair level'),
        ('*_within_pair', 'Value minus duration-eligible pair mean', 'Each regression fits its own sample fixed effects')
    ], columns=['variable', 'definition', 'qualification'])



# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

TARGET_COMPETITION_NAME = "Copa America"
TARGET_SEASON_NAME = "2024"
TARGET_GENDER = "male"

# Central analysis sample:
# - "Regular Play": open-play circulation
# - "From Goal Kick": restart build-up initiated by the goalkeeper
# Set to None to include every play pattern.
BUILDUP_PLAY_PATTERNS: set[str] | None = {"Regular Play", "From Goal Kick"}

# Operational definition. StatsBomb uses a 120 x 80 pitch coordinate model.
LONG_PASS_THRESHOLD = 30.0

# Retain every positive-duration source stint for audit. Analysis uses >=45 min.
MIN_STINT_MINUTES = 0.0

# Match-level reanalysis: no PCA, clustering, or cluster stability resampling.
STRICT_PROCESSING_VALIDATION = True
SHOW_NETWORK_FIGURES = False

MAX_RETRIES = 4
RETRY_BASE_SECONDS = 2.0
FORCE_REFRESH_CACHE = False
OFFLINE = True

SOURCE_ROOT = Path(__file__).resolve().parent / "copa_america2024_match_reanalysis"
DEFAULT_ROOT = Path(__file__).resolve().parent / "논문 결과 표 및 그림"
OUTPUT_ROOT = DEFAULT_ROOT
CACHE_DIR = SOURCE_ROOT / "cache"
EVENT_CACHE_DIR = CACHE_DIR / "events"
LINEUP_CACHE_DIR = CACHE_DIR / "lineups"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"

def initialize_output(output_dir=None, cache_dir=None):
    """Avoid creating files on import; preserve source caches when shared."""
    global OUTPUT_ROOT, CACHE_DIR, EVENT_CACHE_DIR, LINEUP_CACHE_DIR, TABLE_DIR, FIGURE_DIR
    OUTPUT_ROOT = Path(output_dir or OUTPUT_ROOT).resolve()
    CACHE_DIR = Path(cache_dir).resolve() if cache_dir else SOURCE_ROOT / "cache"
    EVENT_CACHE_DIR, LINEUP_CACHE_DIR = CACHE_DIR / "events", CACHE_DIR / "lineups"
    TABLE_DIR, FIGURE_DIR = OUTPUT_ROOT / "tables", OUTPUT_ROOT / "figures"
    for directory in (OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not OFFLINE:
        for directory in (EVENT_CACHE_DIR, LINEUP_CACHE_DIR):
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
    if OFFLINE:
        raise FileNotFoundError(f"Offline mode: uncached request {function.__name__}")
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


def read_portable_cache(cache_file: Path) -> Any | None:
    """Prefer cache pickles without Arrow-backed pandas columns."""
    portable_file = cache_file.with_suffix(".portable.pkl.gz")
    if portable_file.exists() and not FORCE_REFRESH_CACHE:
        return pd.read_pickle(portable_file, compression="gzip")
    if cache_file.exists() and not FORCE_REFRESH_CACHE:
        try:
            value = pd.read_pickle(cache_file)
        except ModuleNotFoundError as error:
            if error.name == "pyarrow":
                raise RuntimeError(
                    f"{cache_file} requires pyarrow. Generate the adjacent "
                    ".portable.pkl.gz caches with a Python environment that has pyarrow."
                ) from error
            raise
        # Reading a supplied cache must not mutate its directory.
        return value
    return None


def write_portable_cache(cache_file: Path, value: Any) -> None:
    """Save string columns as ordinary objects so cache loading does not need pyarrow."""
    def convert(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in result.columns:
            if isinstance(result[column].dtype, (pd.StringDtype, pd.ArrowDtype)):
                result[column] = result[column].astype(object)
        # pandas 3 can Arrow-back the column-name Index as well as the values.
        result.columns = pd.Index(list(result.columns), dtype=object)
        return result

    portable = (
        convert(value) if isinstance(value, pd.DataFrame)
        else {key: convert(frame) for key, frame in value.items()}
    )
    portable_file = cache_file.with_suffix(".portable.pkl.gz")
    temporary_file = portable_file.with_suffix(".tmp")
    pd.to_pickle(portable, temporary_file, compression="gzip")
    temporary_file.replace(portable_file)


def load_match_events(match_id: int) -> pd.DataFrame:
    """Load one match's events from cache or StatsBomb Open Data."""
    cache_file = EVENT_CACHE_DIR / f"{match_id}.pkl"

    cached = read_portable_cache(cache_file)
    if cached is not None:
        return cached

    events = retry_call(sb.events, match_id=int(match_id))
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError(f"경기 {match_id}: 이벤트 데이터가 비어 있습니다.")

    events.to_pickle(cache_file)
    write_portable_cache(cache_file, events)
    return events


def load_match_lineups(match_id: int) -> dict[str, pd.DataFrame]:
    """Load one match's lineups from cache or StatsBomb Open Data."""
    cache_file = LINEUP_CACHE_DIR / f"{match_id}.pkl"

    cached = read_portable_cache(cache_file)
    if cached is not None:
        return cached

    lineups = retry_call(sb.lineups, match_id=int(match_id))
    if not isinstance(lineups, dict) or not lineups:
        raise ValueError(f"경기 {match_id}: 라인업 데이터가 비어 있습니다.")

    with cache_file.open("wb") as file:
        pickle.dump(lineups, file)
    write_portable_cache(cache_file, lineups)

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
    """Accumulate period timestamps, including stoppage time but excluding breaks/shootouts."""
    result = events.copy()
    result["_period_minute"] = pd.to_timedelta(result["timestamp"]).dt.total_seconds() / 60.0
    result["_event_minute"] = np.nan
    offset = 0.0
    for period in (1, 2, 3, 4):
        mask = result["period"].eq(period)
        if not mask.any():
            continue
        elapsed = result.loc[mask, "_period_minute"]
        if elapsed.isna().any() or elapsed.min() < 0:
            raise ValueError(f"Invalid timestamps in period {period}")
        result.loc[mask, "_event_minute"] = offset + elapsed
        offset += float(elapsed.max())
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
        raise ValueError("No regulation/extra-time events available")
    return float(match_end)


def lineup_boundary(clock: Any, period: Any, events: pd.DataFrame, default: float) -> float:
    """Map lineup display clock and explicit period to the cumulative event timeline."""
    value = clock_to_minutes(clock)
    if pd.isna(value):
        return default
    if period is None or pd.isna(period):
        raise ValueError("A nonempty lineup clock requires its period")
    period = int(period)
    if period == 5:
        return get_match_end_minute(events)
    nominal_start = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0}[period]
    previous = events.loc[events["period"].lt(period) & events["period"].isin([1, 2, 3, 4])]
    offset = previous.groupby("period")["_period_minute"].max().sum()
    return float(offset + value - nominal_start)


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

    for stint in merged:
        stint["stint_minutes"] = stint["end_minute"] - stint["start_minute"]
    return merged


def extract_goalkeeper_stints(
    lineup_df: pd.DataFrame,
    team_name: str,
    match_end_minute: float,
    events: pd.DataFrame,
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

            start = lineup_boundary(position.get("from"), position.get("from_period"), events, 0.0)
            end = lineup_boundary(position.get("to"), position.get("to_period"), events, match_end_minute)
            # Use exact substitution timestamps rather than rounded lineup seconds.
            substitutions = events.loc[events["team"].eq(team_name) & events["type"].eq("Substitution")]
            for boundary, column in (("start", "substitution_replacement_id"), ("end", "player_id")):
                if column not in substitutions:
                    continue
                candidate = substitutions.loc[substitutions[column].eq(int(row.player_id)), "_event_minute"].dropna()
                expected = start if boundary == "start" else end
                candidate = candidate.loc[(candidate - expected).abs().lt(1.0 / 60.0 + 1e-6)]
                if len(candidate) == 1:
                    if boundary == "start":
                        start = float(candidate.iloc[0])
                    else:
                        end = float(candidate.iloc[0])

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


def filter_buildup_passes(events: pd.DataFrame, play_patterns: set[str] | None = BUILDUP_PLAY_PATTERNS) -> pd.DataFrame:
    """Select team-network passes; this is not a possession-sequence definition of build-up."""
    passes = events.loc[events["type"].eq("Pass")].copy()

    if play_patterns is not None:
        passes = passes.loc[
            passes["play_pattern"].isin(play_patterns)
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
        "timestamp",
        "index",
        "position",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            "현재 statsbombpy 출력에 필요한 열이 없습니다: "
            f"{sorted(missing)}"
        )
    if events["id"].isna().any() or events["id"].duplicated().any():
        raise ValueError("Missing or duplicate source event IDs")
    if events["index"].isna().any() or events["index"].duplicated().any():
        raise ValueError("Missing or duplicate source event ordering indices")


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
    play_patterns: set[str] | None = BUILDUP_PLAY_PATTERNS,
) -> dict[str, Any]:
    """
    Build a directed weighted team pass network for one goalkeeper's stint
    and return goalkeeper-level network/distribution metrics.
    """
    validate_event_columns(events)

    if "_event_minute" not in events.columns:
        events = prepare_event_time(events)

    # Exclude shootout events and restrict analysis to the goalkeeper's stint.
    stint_events = event_window(events, start_minute, end_minute)
    stint_events = stint_events.loc[stint_events["team"].eq(team_name)].copy()

    passes = filter_buildup_passes(stint_events, play_patterns)

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
    pass_lengths = pass_lengths.where(np.isfinite(pass_lengths) & pass_lengths.ge(0))
    known_lengths = int(pass_lengths.notna().sum())
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
        "gk_pass_length_observed": known_lengths,
        "gk_pass_length_coverage": safe_divide(known_lengths, attempted_passes),
        "pass_completion_pct": safe_divide(completed_passes, attempted_passes),
        "long_pass_attempts": long_attempts,
        "long_pass_completed": long_completed,
        "long_pass_share": safe_divide(long_attempts, known_lengths),
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
    """Resolve IDs dynamically and verify that Copa America 2024 is available."""
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
    play_patterns: set[str] | None = BUILDUP_PLAY_PATTERNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all tournament matches and return metrics plus an error log."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for match_row in tqdm(
        matches.sort_values(["match_date", "match_id"], kind="stable").itertuples(index=False),
        total=len(matches),
        desc="Copa America 2024 경기 처리",
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    ):
        match_id = int(match_row.match_id)

        try:
            raw_events = load_match_events(match_id)
            validate_event_columns(raw_events)
            events = prepare_event_time(raw_events)
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
                    events=events,
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
                    if stint["stint_minutes"] <= MIN_STINT_MINUTES:
                        continue

                    metrics = calculate_goalkeeper_stint_metrics(
                        events=events,
                        team_name=team_name,
                        goalkeeper_id=int(stint["goalkeeper_id"]),
                        goalkeeper_name=str(stint["goalkeeper"]),
                        start_minute=float(stint["start_minute"]),
                        end_minute=float(stint["end_minute"]),
                        play_patterns=play_patterns,
                    )

                    metrics.update(add_event_context(
                        events, team_name, int(stint["goalkeeper_id"]),
                        float(stint["start_minute"]), float(stint["end_minute"]), play_patterns,
                    ))
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
            "long_pass_share": safe_divide(long_attempts, group["gk_pass_length_observed"].sum()),
            "long_pass_completion_pct": safe_divide(
                long_completed,
                long_attempts,
            ),
            "average_pass_length": safe_divide(
                group["gk_pass_length_sum"].sum(),
                group["gk_pass_length_observed"].sum(),
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
# 6. Processing and coverage validation
# ---------------------------------------------------------------------

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


def audit_time_and_pass_coverage(matches: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Independently reconcile raw goalkeeper passes and period durations with stint totals."""
    period_rows, coverage_rows, pattern_frames = [], [], []
    for match in matches.itertuples(index=False):
        raw = load_match_events(int(match.match_id))
        raw = raw.loc[raw["period"].isin([1, 2, 3, 4])].copy()
        raw["elapsed"] = pd.to_timedelta(raw["timestamp"]).dt.total_seconds() / 60
        durations = raw.groupby("period")["elapsed"].max()
        for period, duration in durations.items():
            period_rows.append({"match_id": match.match_id, "period": period, "elapsed_minutes": duration})
        gk_passes = raw.loc[raw["type"].eq("Pass") & raw["position"].eq("Goalkeeper")].copy()
        counts = gk_passes.groupby("play_pattern", dropna=False).size().rename("attempts").reset_index()
        counts["match_id"] = match.match_id
        pattern_frames.append(counts)
        for team in (match.home_team, match.away_team):
            observed = metrics.loc[metrics["match_id"].eq(match.match_id) & metrics["team"].eq(team)]
            source = gk_passes.loc[gk_passes["team"].eq(team)]
            selected = filter_buildup_passes(source)
            intervals = observed.sort_values("start_minute")
            boundaries_ok = bool(len(intervals) and abs(intervals.iloc[0]["start_minute"]) < 1e-8
                and abs(intervals.iloc[-1]["end_minute"] - durations.sum()) < 1e-8
                and np.allclose(intervals["end_minute"].to_numpy()[:-1],
                                intervals["start_minute"].to_numpy()[1:], atol=1e-8, rtol=0))
            row = {"match_id": match.match_id, "team": team,
                   "period_minutes": durations.sum(), "stint_minutes": observed["stint_minutes"].sum(),
                   "raw_selected_attempts": len(selected), "processed_attempts": observed["gk_pass_attempts"].sum(),
                   "raw_selected_completed": selected["pass_outcome"].isna().sum(),
                   "processed_completed": observed["gk_completed_passes"].sum(),
                   "intervals_contiguous": boundaries_ok}
            row["passed"] = bool(boundaries_ok and abs(row["period_minutes"] - row["stint_minutes"]) < 1e-8
                and row["raw_selected_attempts"] == row["processed_attempts"]
                and row["raw_selected_completed"] == row["processed_completed"])
            coverage_rows.append(row)
    patterns = pd.concat(pattern_frames).groupby("play_pattern", as_index=False)["attempts"].sum()
    patterns["included_primary"] = True if BUILDUP_PLAY_PATTERNS is None else patterns["play_pattern"].isin(BUILDUP_PLAY_PATTERNS)
    return {"period_durations.csv": pd.DataFrame(period_rows),
            "time_pass_coverage_audit.csv": pd.DataFrame(coverage_rows),
            "raw_goalkeeper_play_pattern_counts.csv": patterns}


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

    window = event_window(events, start, end)
    window = window.loc[window["team"].eq(team_name)].copy()

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
            mutation_scale=16,
            shrinkA=12,
            shrinkB=15,
            linewidth=width,
            alpha=0.55,
            connectionstyle="arc3,rad=0.14",
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
            color="#f4a340" if is_goalkeeper else "#a8cde6",
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
    ax.text(0.5, -0.06,
             "Orange: goalkeeper; blue: teammates. Arrows: passer to recipient; width: completed-pass count.\n"
             "Node area = 130 + 38 x sqrt(in-strength + out-strength); GK area multiplied by 1.45 for visibility.\n"
             "Descriptive match nearest the pair's median participation share; not a goalkeeper type.",
             ha="center", va="top", fontsize=9, transform=ax.transAxes)

    safe_name = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", goalkeeper_name)
    output = FIGURE_DIR / (
        f"network_{safe_name}_{team_name}_{match_id}.png".replace(" ", "_")
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if SHOW_NETWORK_FIGURES:
        plt.show()
    plt.close(fig)
    return output


def validate_matches(matches):
    required = {"match_id", "match_date", "home_team", "away_team", "home_score", "away_score", "competition_stage"}
    if not required.issubset(matches):
        raise ValueError(f"Match table is missing: {sorted(required-set(matches))}")
    dates = pd.to_datetime(matches.match_date, errors="coerce")
    if (len(matches) != 32 or matches.match_id.duplicated().any()
            or dates.isna().any() or not dates.dt.year.eq(2024).all()
            or not dates.between("2024-06-20", "2024-07-15").all()
            or len(set(matches.home_team) | set(matches.away_team)) != 16):
        raise ValueError("Expected the complete, unique 32-match/16-team Copa America 2024 table")
    if "competition_name" in matches and not matches.competition_name.eq(TARGET_COMPETITION_NAME).all():
        raise ValueError("Unexpected competition in match cache")


def input_provenance(matches, match_file, ratings_path=None):
    paths = [("matches", match_file)]
    for match_id in matches.match_id:
        for kind, directory in (("events", EVENT_CACHE_DIR), ("lineups", LINEUP_CACHE_DIR)):
            original = directory / f"{match_id}.pkl"
            portable = original.with_suffix(".portable.pkl.gz")
            used = portable if portable.exists() else original
            paths.append((kind, used))
    source_root = Path(__file__).resolve().parent
    paths += [("code", Path(__file__).resolve())]
    paths += [("strength_source", path) for path in sorted((source_root / "review_sources").glob("*")) if path.is_file()]
    if ratings_path:
        paths.append(("ratings", Path(ratings_path)))
    records = []
    for kind, path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                digest.update(block)
        records.append({"kind": kind, "path": str(path.resolve()),
                        "relative_path": os.path.relpath(path.resolve(), OUTPUT_ROOT).replace("\\", "/"),
                        "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return pd.DataFrame(records)


def main(output_dir=None, cache_dir=None, matches_csv=None, ratings_path=None, networks=True) -> dict[str, Any]:
    initialize_output(output_dir, cache_dir)
    run_status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    (OUTPUT_ROOT / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    if ratings_path is None:
        ratings_path = Path(__file__).resolve().parent / "review_sources" / "opponent_elo_20240619.csv"
    if not Path(ratings_path).is_file():
        raise FileNotFoundError("Pre-tournament opponent ratings are required; provide --opponent-ratings or bundled review_sources/opponent_elo_20240619.csv")
    match_cache = Path(matches_csv) if matches_csv else SOURCE_ROOT / "tables" / "copa_america2024_matches.csv"
    if match_cache.exists() and not FORCE_REFRESH_CACHE:
        matches = pd.read_csv(match_cache)
    else:
        competition_id, season_id, competitions = resolve_target_competition()
        matches = retry_call(sb.matches, competition_id=competition_id, season_id=season_id)
    matches = matches.sort_values(["match_date", "match_id"], kind="stable").reset_index(drop=True)
    validate_matches(matches)

    print(f"공개 경기 수: {len(matches)}")

    matches.to_csv(
        TABLE_DIR / "copa_america2024_matches.csv",
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
            "and processing_errors.csv before interpreting match associations."
        )

    audit_tables = audit_time_and_pass_coverage(matches, goalkeeper_match)
    for filename, table in audit_tables.items():
        table.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")
    if not audit_tables["time_pass_coverage_audit.csv"]["passed"].all():
        raise RuntimeError("Raw-pass/time reconciliation failed; inspect time_pass_coverage_audit.csv")

    goalkeeper_summary = aggregate_goalkeepers(goalkeeper_match)
    goalkeeper_summary.to_csv(
        TABLE_DIR / "goalkeeper_tournament_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    context_rows = []
    for match in matches.itertuples(index=False):
        context_rows.extend(validate_match_context(
            prepare_event_time(load_match_events(int(match.match_id))), match,
            goalkeeper_match.loc[goalkeeper_match.match_id.eq(match.match_id)]))
    context_qc = pd.DataFrame(context_rows)
    context_qc.to_csv(TABLE_DIR / "context_reconciliation.csv", index=False, encoding="utf-8-sig")
    if not context_qc.passed.all():
        raise RuntimeError("Score/time/state-pass/shot reconciliation failed; inspect context_reconciliation.csv")
    results = run_match_analysis(goalkeeper_match, TABLE_DIR, OUTPUT_ROOT, ratings_path)
    if networks:
        # Median participation observation per pair; selection is descriptive,
        # independent of attack outcome and regression significance.
        sample = results["primary_match_sample.csv"]
        selected = []
        for _, group in sample.groupby("pair_id", sort=True):
            row = group.assign(_distance=(group.team_pass_share-group.team_pass_share.median()).abs()).sort_values(
                ["_distance", "match_id", "start_minute"]).iloc[0].drop(labels="_distance")
            selected.append(row)
            plot_goalkeeper_match_network(row)
        pd.DataFrame(selected).to_csv(TABLE_DIR / "descriptive_network_matches.csv", index=False, encoding="utf-8-sig")
    written_matches = TABLE_DIR / "copa_america2024_matches.csv"
    input_provenance(matches, written_matches, ratings_path).to_csv(
        TABLE_DIR / "input_provenance.csv", index=False, encoding="utf-8-sig")
    run_status.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      source_matches=len(matches), source_stints=len(goalkeeper_match),
                      time_pass_audit_passed=True, context_audit_passed=True,
                      network_figures_created=len(selected) if networks else 0)
    (OUTPUT_ROOT / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    print(f"검증 및 파일 저장 완료: {OUTPUT_ROOT}")
    return {"goalkeeper_match": goalkeeper_match,
            "goalkeeper_summary": goalkeeper_summary, "tables": results,
            "output_root": OUTPUT_ROOT}




def paper_number(value, digits=2):
    """Published-scale rounding, including the manuscript's mathematical minus."""
    if not np.isfinite(value):
        raise ValueError('A manuscript coefficient is not estimable on these inputs')
    return f'{float(value):.{digits}f}'.replace('-', '\u2212')


def paper_interval(low, high, digits=2):
    return paper_number(low, digits)+'\u2013'+paper_number(high, digits)


def paper_p(value):
    if value < .001:
        return '<.001'
    text = f'{float(value):.3f}'
    return text[1:] if text.startswith('0.') else text


def calculate_paper_tables(analysis_root):
    """Build the seven manuscript tables from recomputed full-precision values."""
    folder = Path(analysis_root)/'tables'
    d = pd.read_csv(folder/'primary_match_sample.csv')
    models = pd.read_csv(folder/'exploratory_pair_fixed_effects.csv')
    principal = models.loc[models['sample'].eq('primary')]
    diag = pd.read_csv(folder/'model_diagnostics.csv')
    boot = pd.read_csv(folder/'crossed_cluster_bootstrap.csv')
    leave = pd.read_csv(folder/'leave_one_team_out_summary.csv')
    sensitivity = pd.read_csv(folder/'sample_sensitivity_counts.csv').set_index('sample')
    model_order = ['context_participation', 'context_long_distribution', 'context_entropy',
                   'concurrent_territory', 'concurrent_shots', 'concurrent_xg']
    labels = ['Participation', 'Long distribution', 'Entropy (exploratory)',
              'Opponent-half endpoints', 'Shots/90 min', 'xG/90 min']
    primary_diag = diag.loc[diag['sample'].eq('primary')].set_index('model_id')
    rows = []
    for model, label in zip(model_order, labels):
        r = primary_diag.loc[model]
        rows.append([label, *[str(int(r[name])) for name in ['complete_case_n', 'n', 'pairs', 'teams', 'matches']]])
    tables = [rows]

    variables = [
        ('team_pass_share', 'Participation (%)', 100, 2),
        ('betweenness', 'Betweenness', 1, 3),
        ('receiver_diversity', 'Recipient count (players)', 1, 2),
        ('normalized_entropy', 'Normalised entropy', 1, 3),
        ('long_pass_share', 'Long-pass share (%)', 100, 2),
        ('strength_difference', 'Elo difference', 1, 2),
        ('leading_time_share', 'Leading exposure (%)', 100, 2),
        ('drawing_time_share', 'Drawing exposure (%)', 100, 2),
        ('trailing_time_share', 'Trailing exposure (%)', 100, 2),
        ('gk_pressure_share', 'Pressure share (%)', 100, 2),
        ('goal_kick_share', 'Goal-kick-pattern share (%)', 100, 2),
        ('outfield_opponent_half_share', 'Opponent-half endpoints (%)', 100, 2),
        ('team_shots_per90', 'Shots/90 min', 1, 2),
        ('team_xg_per90', 'xG/90 min', 1, 3),
    ]
    rows = []
    for name, label, scale, digits in variables:
        values = d[name]*scale
        rows.append([label, f'{paper_number(values.mean(), digits)} ({paper_number(values.std(ddof=1), digits)})',
                     paper_number(values.median(), digits),
                     paper_interval(values.quantile(.25), values.quantile(.75), digits),
                     paper_interval(values.min(), values.max(), digits)])
    tables.append(rows)

    repeated = d.loc[d.groupby('pair_id').pair_id.transform('size').ge(2)]
    rows = []
    for name, label, scale, digits in [
        ('team_pass_share', 'Participation (pp)', 100, 2),
        ('betweenness', 'Betweenness', 1, 3),
        ('receiver_diversity', 'Recipient count (players)', 1, 2),
        ('normalized_entropy', 'Normalised entropy', 1, 3),
        ('long_pass_share', 'Long-pass share (pp)', 100, 2),
    ]:
        grouped = repeated.groupby('pair_id')[name]
        ranges = (grouped.max()-grouped.min())*scale
        rows.append([label, paper_number(ranges.median(), digits),
                     paper_interval(ranges.quantile(.25), ranges.quantile(.75), digits),
                     paper_interval(ranges.min(), ranges.max(), digits)])
    tables.append(rows)

    predictors = ['leading_time_share', 'trailing_time_share', 'gk_pressure_share', 'goal_kick_share']
    predictor_labels = ['Leading', 'Trailing', 'Pressure', 'Goal-kick pattern']
    rows = []
    ordered_focal = []
    for model, outcome in [('context_participation', 'Participation'),
                           ('context_long_distribution', 'Long pass'), ('context_entropy', 'Entropy\u2020')]:
        subset = principal.loc[principal.model_id.eq(model)].set_index('predictor')
        digits, scale = (4, 1) if model == 'context_entropy' else (2, 100)
        for name, label in zip(predictors, predictor_labels):
            r = subset.loc[name]
            ordered_focal.append((model, name, outcome, label))
            rows.append([outcome, label, paper_number(r.estimate_per_increment*scale, digits),
                         paper_interval(r.ci_low_per_increment*scale, r.ci_high_per_increment*scale, digits),
                         paper_p(r.p_unadjusted), paper_p(r.p_holm_primary_family)])
    tables.append(rows)

    rows = []
    for model, label in zip(model_order[3:], labels[3:]):
        r = principal.loc[principal.model_id.eq(model)].iloc[0]
        digits, scale = (3, 1) if model == 'concurrent_xg' else (2, 100 if model == 'concurrent_territory' else 1)
        rows.append([label, paper_number(r.estimate_per_increment*scale, digits),
                     paper_interval(r.ci_low_per_increment*scale, r.ci_high_per_increment*scale, digits),
                     paper_p(r.p_unadjusted), paper_p(r.p_holm_primary_family)])
        ordered_focal.append((model, 'team_pass_share', label, 'Participation'))
    tables.append(rows)

    rows = []
    for model, predictor, outcome, label in ordered_focal:
        b = boot.loc[boot.model_id.eq(model) & boot.predictor.eq(predictor)].iloc[0]
        l = leave.loc[leave.model_id.eq(model) & leave.predictor.eq(predictor)].iloc[0]
        scale = .1 if model in ['context_entropy', 'concurrent_shots', 'concurrent_xg'] else 10.
        digits = 4 if model == 'context_entropy' else (3 if model == 'concurrent_xg' else 2)
        interval = ('Withheld' if not np.isfinite(b.get('percentile_low', np.nan)) else
                    paper_interval(b.percentile_low*scale, b.percentile_high*scale, digits))
        rows.append([outcome, label, f'{int(b.valid)}/{int(b.attempted)}', interval,
                     paper_interval(l.min_estimate*scale, l.max_estimate*scale, digits),
                     f'{int(l.sign_changes)}/{int(l.omissions)}'])
    tables.append(rows)

    rows = []
    conditions = [('primary', 'Primary'), ('minutes60', 'At least 60 min'),
                  ('attempts10_completed5', 'Attempts \u226510, completed \u22655'),
                  ('attempts20_completed10', 'Attempts \u226520, completed \u226510'),
                  ('no_recorded_dismissals', 'No recorded dismissals'), ('without_opponent_strength', 'Without Elo control')]
    for sample, label in conditions:
        s = sensitivity.loc[sample]
        row = [label, f'{int(s.n)}/{int(s.pairs)}']
        for model in model_order[:3]:
            r = diag.loc[diag['sample'].eq(sample) & diag.model_id.eq(model)].iloc[0]
            mark = '*' if min(r.teams, r.matches) < MIN_CLUSTERS else ''
            row.append(f'{int(r.n)}/{int(r.pairs)}'+mark)
        rows.append(row)
    tables.append(rows)
    return d, tables


def validate_paper_tables(tables):
    """Reference cells never supply calculated data: they are comparison targets."""
    import re
    def normalized(value):
        return re.sub(r'\s+', ' ', str(value)).strip()
    checked = 0
    for index, (actual, reference) in enumerate(zip(tables, PAPER_REFERENCE['tables']), 1):
        expected = reference[1:]
        if len(actual) != len(expected):
            raise ValueError(f'Table {index}: manuscript and calculation row counts differ')
        for row_number, (a, b) in enumerate(zip(actual, expected), 1):
            if len(a) != len(b):
                raise ValueError(f'Table {index}, row {row_number}: column counts differ')
            for column_number, (value, target) in enumerate(zip(a, b), 1):
                if normalized(value) != normalized(target):
                    raise ValueError(f'Table {index}, row {row_number}, column {column_number}: calculated {value!r}, manuscript {target!r}')
                checked += 1
    return checked


def save_paper_outputs(d, tables, folder):
    import math
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    filenames = ['Analytical_samples', 'Descriptive_statistics', 'Within_combination_ranges',
                 'Context_associations', 'Concurrent_attacking_indicators',
                 'Resampling_and_leave_one_out', 'Eligibility_sensitivity']
    workbook = Workbook()
    workbook.remove(workbook.active)
    for index, (rows, suffix) in enumerate(zip(tables, filenames), 1):
        headers = [re.sub(r'\s+', ' ', cell).strip() for cell in PAPER_REFERENCE['tables'][index-1][0]]
        pd.DataFrame(rows, columns=headers).to_csv(folder/f'Table_{index}_{suffix}.csv', index=False, encoding='utf-8-sig')
        sheet = workbook.create_sheet(f'Table {index}')
        width = len(headers)
        sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=width)
        sheet.cell(1, 1, PAPER_REFERENCE['captions'][index-1])
        sheet.cell(1, 1).font = Font(name='Calibri', size=13, bold=True)
        sheet.cell(1, 1).alignment = Alignment(wrap_text=True, vertical='center')
        sheet.row_dimensions[1].height = 40
        for column_index, name in enumerate(headers, 1):
            cell = sheet.cell(3, column_index, name)
            cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='26475D')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
            sheet.column_dimensions[get_column_letter(column_index)].width = 34 if column_index == 1 else 24
        sheet.row_dimensions[3].height = 38
        for row_index, row in enumerate(rows, 4):
            for column_index, value in enumerate(row, 1):
                cell = sheet.cell(row_index, column_index, value)
                cell.font = Font(name='Calibri', size=11)
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            sheet.row_dimensions[row_index].height = 30
        note_row = len(rows)+5
        sheet.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=width)
        sheet.cell(note_row, 1, PAPER_REFERENCE['notes'][index-1])
        sheet.cell(note_row, 1).alignment = Alignment(wrap_text=True, vertical='top')
        sheet.cell(note_row, 1).font = Font(name='Calibri', size=10)
        sheet.row_dimensions[note_row].height = max(55, math.ceil(len(PAPER_REFERENCE['notes'][index-1])/105)*14)
        sheet.freeze_panes = 'B4'
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = 'landscape'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = sheet.page_setup.fitToHeight = 1
        sheet.print_title_rows = '1:3'
        sheet.print_area = f'A1:{get_column_letter(width)}{note_row}'
    workbook.save(folder/'논문 결과표.xlsx')

    groups = list(d.groupby(['team', 'goalkeeper'], sort=True))
    for figure, feature, label in [(1, 'team_pass_share', 'Goalkeeper share of team completed passes (%)'),
                                    (2, 'long_pass_share', 'Goalkeeper long-pass share (%)')]:
        fig, ax = plt.subplots(figsize=(9, 9))
        for position, (_, group) in enumerate(groups):
            values = group[feature].to_numpy()*100
            ax.plot([values.min(), values.max()], [position, position], color='#8BAFC6', lw=1.3)
            ax.scatter(values, position+np.linspace(-.12, .12, len(values)), color='#2C6F9A', s=24, zorder=3)
            ax.scatter(np.median(values), position, color='#BD6A22', marker='D', s=35, zorder=4)
        ax.set_yticks(range(len(groups)), [f'{team} / {name} (n={len(group)})' for (team, name), group in groups], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
        if figure == 2:
            ax.set_xlim(0, 105)
        fig.tight_layout()
        suffix = 'Participation_share' if figure == 1 else 'Long_pass_share'
        fig.savefig(folder/f'Figure_{figure}_{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Additional raw descriptive plot requested alongside the manuscript figures.
    # Use all duration-eligible stints, not the 62-row fixed-effects sample.
    # The 10-pass line marks the entropy-model observation rule, not significance.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(d.gk_completed_passes, d.normalized_entropy,
                    color='#002451', s=28)
    axes[0].axvline(MIN_COMPLETED, ls='--', color='grey')
    axes[0].set(xlabel='Completed goalkeeper passes',
                ylabel='Normalized recipient entropy')
    axes[1].scatter(d.team_pass_share, d.outfield_opponent_half_share,
                    s=28, alpha=.75)
    axes[1].set(xlabel='GK share of team completed passes',
                ylabel='Outfield opponent-half endpoint share')
    fig.suptitle('Observation volume and concurrent territory (raw descriptive associations)')
    fig.tight_layout()
    fig.savefig(folder/'Figure_3_Observation_volume_and_concurrent_territory.png', dpi=200)
    plt.close(fig)


def paper_main():
    import contextlib
    import io
    import shutil
    import tempfile
    parser = argparse.ArgumentParser(description='Recompute seven manuscript tables, two manuscript figures and the requested observation-volume/territory scatter plot')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    print('논문 결과 분석을 실행합니다. 표 7개와 그림 3개를 원자료에서 계산합니다.', flush=True)
    # Full-precision regression/resampling and audit files are private scratch
    # data; only manuscript tables/figures are published into the output folder.
    with tempfile.TemporaryDirectory(prefix='.copa_paper_', dir=Path(__file__).resolve().parent) as scratch:
        internal = Path(scratch)/'analysis'
        staged = Path(scratch)/'published'
        with contextlib.redirect_stdout(io.StringIO()):
            main(output_dir=internal, networks=False)
        data, tables = calculate_paper_tables(internal)
        checked = validate_paper_tables(tables)
        save_paper_outputs(data, tables, staged)
        output.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            shutil.copy2(path, output/path.name)
    print(f'완료: 원고 표의 데이터 {checked}개 셀 일치 확인.', flush=True)
    print(f'표 7개(CSV 및 Excel 통합문서), 그림 3개 저장: {output}', flush=True)


# Frozen manuscript reference: used only to CHECK independently computed results.
PAPER_REFERENCE = {'sha256': 'bd1379210d6550ad08d557b5c98d735dec9d994a89555e1f90a7e727c1f713ca', 'tables': [[['Model', 'Eligible stints', 'Analysed stints', 'Pairs', 'Teams', 'Matches'], ['Participation', '65', '62', '16', '16', '32'], ['Long distribution', '60', '58', '16', '16', '32'], ['Entropy (exploratory)', '41', '36', '11', '11', '28'], ['Opponent-half endpoints', '65', '62', '16', '16', '32'], ['Shots/90 min', '65', '62', '16', '16', '32'], ['xG/90 min', '65', '62', '16', '16', '32']], [['Variable', 'Mean (SD)', 'Median', 'IQR', 'Minimum–maximum'], ['Participation (%)', '7.96 (3.31)', '7.12', '5.83–9.42', '3.06–21.11'], ['Betweenness', '0.046 (0.063)', '0.019', '0.000–0.074', '0.000–0.299'], ['Recipient count (players)', '5.45 (1.85)', '6.00', '4.00–7.00', '1.00–10.00'], ['Normalised entropy', '0.877 (0.131)', '0.898', '0.857–0.948', '0.000–1.000'], ['Long-pass share (%)', '51.26 (20.36)', '48.00', '35.71–68.00', '13.64–100.00'], ['Elo difference', '1.42 (247.06)', '5.00', '−150.00–150.00', '−425.00–425.00'], ['Leading exposure (%)', '22.91 (31.93)', '0.00', '0.00–47.62', '0.00–97.45'], ['Drawing exposure (%)', '54.06 (31.92)', '51.00', '32.09–77.72', '2.55–100.00'], ['Trailing exposure (%)', '23.04 (31.93)', '0.00', '0.00–47.62', '0.00–97.45'], ['Pressure share (%)', '10.36 (8.19)', '10.00', '3.85–14.29', '0.00–37.50'], ['Goal-kick-pattern share (%)', '51.23 (19.62)', '55.56', '38.89–65.00', '0.00–85.71'], ['Opponent-half endpoints (%)', '43.84 (10.64)', '45.03', '36.46–50.63', '20.62–69.27'], ['Shots/90 min', '10.52 (4.73)', '9.85', '7.03–13.67', '1.81–20.31'], ['xG/90 min', '1.045 (0.767)', '0.768', '0.567–1.337', '0.033–3.417']], [['Descriptor', 'Median range', 'IQR of ranges', 'Minimum–maximum range'], ['Participation (pp)', '4.65', '3.14–6.28', '1.63–17.13'], ['Betweenness', '0.074', '0.055–0.140', '0.000–0.207'], ['Recipient count (players)', '4.00', '2.75–5.00', '1.00–6.00'], ['Normalised entropy', '0.118', '0.068–0.165', '0.038–0.330'], ['Long-pass share (pp)', '34.85', '21.63–46.58', '12.50–51.82']], [['Outcome', 'Exposure\n(+10 pp)', 'Estimate', '95% CI', 'Unadjusted\np', 'Holm\np'], ['Participation', 'Leading', '−0.10', '−0.47–0.26', '.555', '1.000'], ['Participation', 'Trailing', '0.31', '−0.05–0.68', '.089', '.886'], ['Participation', 'Pressure', '−1.15', '−2.07–−0.24', '.017', '.203'], ['Participation', 'Goal-kick pattern', '0.63', '−0.03–1.28', '.058', '.643'], ['Long pass', 'Leading', '1.52', '−0.67–3.70', '.160', '1.000'], ['Long pass', 'Trailing', '−1.30', '−4.99–2.40', '.466', '1.000'], ['Long pass', 'Pressure', '2.32', '−3.78–8.42', '.430', '1.000'], ['Long pass', 'Goal-kick pattern', '1.94', '−1.02–4.90', '.183', '1.000'], ['Entropy†', 'Leading', '0.0021', '−0.0113–0.0154', '.736', '1.000'], ['Entropy†', 'Trailing', '0.0076', '−0.0072–0.0224', '.278', '1.000'], ['Entropy†', 'Pressure', '−0.0091', '−0.0352–0.0169', '.453', '1.000'], ['Entropy†', 'Goal-kick pattern', '−0.0030', '−0.0206–0.0145', '.708', '1.000']], [['Concurrent indicator', 'Estimated difference', '95% CI', 'Unadjusted p', 'Holm p'], ['Opponent-half endpoints', '−7.73', '−18.32–2.87', '.141', '.282'], ['Shots/90 min', '−4.30', '−9.29–0.69', '.086', '.258'], ['xG/90 min', '−0.055', '−0.707–0.597', '.860', '.860']], [['Model', 'Predictor', 'Valid /\n1,999', 'Resampling\n95% interval', 'Leave-one-out\nmin–max', 'Sign changes /\nomissions'], ['Participation', 'Leading', '1918/1999', '−0.83–0.47', '−0.18–−0.02', '0/16'], ['Participation', 'Trailing', '1918/1999', '−0.38–1.05', '0.23–0.39', '0/16'], ['Participation', 'Pressure', '1918/1999', '−2.81–0.95', '−1.36–−1.02', '0/16'], ['Participation', 'Goal-kick pattern', '1918/1999', '−0.20–1.84', '0.44–0.73', '0/16'], ['Long pass', 'Leading', '1831/1999', '−1.83–6.58', '1.02–2.14', '0/16'], ['Long pass', 'Trailing', '1831/1999', '−7.89–4.76', '−2.04–0.01', '1/16'], ['Long pass', 'Pressure', '1831/1999', '−11.95–19.38', '0.56–4.13', '0/16'], ['Long pass', 'Goal-kick pattern', '1831/1999', '−4.62–8.37', '1.19–3.05', '0/16'], ['Entropy†', 'Leading', '628/1999', 'Withheld', '−0.0018–0.0058', '2/11'], ['Entropy†', 'Trailing', '628/1999', 'Withheld', '0.0047–0.0115', '0/11'], ['Entropy†', 'Pressure', '628/1999', 'Withheld', '−0.0146–0.0025', '1/11'], ['Entropy†', 'Goal-kick pattern', '628/1999', 'Withheld', '−0.0070–0.0012', '2/11'], ['Opponent-half endpoints', 'Participation', '1909/1999', '−40.30–17.89', '−9.83–−5.84', '0/16'], ['Shots/90 min', 'Participation', '1908/1999', '−15.17–7.47', '−5.65–−3.29', '0/16'], ['xG/90 min', 'Participation', '1895/1999', '−1.363–1.849', '−0.214–0.113', '3/16']], [['Condition', 'Descriptive stints/pairs', 'Participation stints/pairs', 'Long pass stints/pairs', 'Entropy stints/pairs'], ['Primary', '65/19', '62/16', '58/16', '36/11'], ['At least 60 min', '63/18', '61/16', '57/16', '36/11'], ['Attempts ≥10, completed ≥5', '58/17', '57/16', '57/16', '36/11'], ['Attempts ≥20, completed ≥10', '18/10', '13/5*', '13/5*', '13/5*'], ['No recorded dismissals', '50/17', '49/16', '44/15', '24/8'], ['Without Elo control', '65/19', '62/16', '58/16', '36/11']]], 'captions': ['Table 1. Model-specific analytical samples.', 'Table 2. Stint-level descriptors, match context and concurrent attacking indicators.', 'Table 3. Observed within-combination ranges across matches.', 'Table 4. Context associations with participation, long-pass share and exploratory entropy.', 'Table 5. Participation share and concurrent attacking indicators.', 'Table 6. Crossed-resampling intervals and leave-one-national-team-out estimates.', 'Table 7. Eligibility sensitivity and model-specific retained observations.'], 'notes': ['Note. The descriptive sample contains 65 stints, 19 goalkeeper–national-team pairs, 16 teams and 32 matches. Eligible stints follow metric-specific and complete-case rules; analysed stints additionally exclude singleton pairs in that model sample. Repeated stints are not independent participants.', 'Note. N = 65 for every variable; stints receive equal weight. The first five rows describe passing characteristics. The long-pass denominator comprises attempts with known length. Entropy measures evenness among observed recipients. Elo difference is the team-minus-opponent rating; models divide it by 100. Rounded score-state percentages may not sum to 100%. Endpoint share uses selected-pattern outfield passes; shots and xG include all patterns. SD = standard deviation; IQR = interquartile range.', 'Note. For each of 16 pairs observed at least twice, range = maximum minus minimum; these ranges were then summarised across pairs. Three singleton pairs are excluded. Pairs contribute two to six observations. These are observed ranges, not estimates of population variability or tactical adaptability. pp = percentage points.', 'Note. All estimates refer to a 10-percentage-point (pp) increase in the exposure. Participation and long-pass estimates are in pp; entropy estimates are on the 0–1 scale. Models include pair fixed effects, relative Elo strength and the four listed exposures; entropy additionally adjusts for log completed-pass count. Participation uses 62 stints, long distribution 58 and entropy 36. Two-way national-team/match CR1 t reference df = 15, 15 and 10, respectively. Leading and trailing are relative to drawing. Holm adjustment covers all 12 displayed coefficients. Ordinary intervals are not simultaneous, multiplicity-adjusted intervals. Entropy has limited inferential support and is not principal evidence. † Entropy is an additional exploratory analysis.', 'Note. Differences are per 10-percentage-point increase in participation. Endpoint-share differences are in percentage points; shots and xG differences are per 90 elapsed minutes. Each model uses 62 stints, 16 pairs/teams and 32 matches, with pair fixed effects and five context covariates. Holm covers the three participation coefficients. Intervals are ordinary 95% intervals. Opponent-half endpoint share is not post-goalkeeper-pass progression success.', 'Note. Every estimate is per 10-percentage-point predictor increase. Share outcomes use percentage points, entropy its 0–1 scale, and shots/xG rates per 90 elapsed minutes. National-team and match identifiers were independently resampled; selection multiplicities were multiplied for refitting. Model seeds were 20261015–20261020. Intervals are 2.5th–97.5th percentiles, not multiplicity-adjusted intervals, exact few-cluster corrections or bootstrap p values. Entropy intervals were withheld: 628/1,999 valid attempts (31.4%); 1,322 failed the six-cluster rule and 49 failed within-pair information/rank conditions. The reporting rule required at least 80% valid attempts. Leave-one-out ranges cover all estimable omissions; sign changes are relative to the principal fit. Stable signs do not imply precise magnitudes or robust inference. Linear-coefficient intervals are not predictions of feasible share changes. † Entropy is an additional exploratory analysis.', 'Note. Cells give stints/pairs. Descriptive counts precede model-specific filters and singleton removal. All three attacking models have the same counts as participation within each condition. * The attempts ≥20/completed ≥10 condition retained only five national teams, below the minimum-six-cluster rule: counts show remaining observations, not reported estimates. The Elo-omission condition changes adjustment, not descriptive eligibility.']}


if __name__ == '__main__':
    paper_main()

