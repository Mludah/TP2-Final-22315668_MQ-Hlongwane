"""Reproducible PR ingestion and the notebook's local/metro swing estimator.

No notebook execution, fitted artifacts, or network requests are required.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2011, 2016, 2021)
PARTIES = ['ANC', 'DA', 'EFF', 'IFP', 'Other']
PARTY_NAMES = {
    'ANC': 'African National Congress', 'DA': 'Democratic Alliance',
    'EFF': 'Economic Freedom Fighters', 'IFP': 'Inkatha Freedom Party',
    'Other': 'All remaining parties (combined)',
}
ALIASES = {
    'AFRICAN NATIONAL CONGRESS': 'ANC',
    'DEMOCRATIC ALLIANCE': 'DA',
    'DEMOCRATIC ALLIANCE/DEMOKRATIESE ALLIANSIE': 'DA',
    'ECONOMIC FREEDOM FIGHTERS': 'EFF', 'INKATHA FREEDOM PARTY': 'IFP',
}


@dataclass
class ElectionData:
    ward_votes: pd.DataFrame
    ward_stats: pd.DataFrame
    metro_votes: pd.DataFrame
    metro_stats: pd.DataFrame
    audit: pd.DataFrame
    anomalies: pd.DataFrame


def read_results(path: Path, year: int) -> tuple[pd.DataFrame, int]:
    prefix = path.read_bytes()[:2]
    encoding = 'utf-16' if prefix in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    try:
        frame = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding='cp1252')
    frame.columns = frame.columns.str.strip().str.lower()
    required = {'ballottype', 'ward', 'votingdistrict', 'registeredvoters',
                'spoiltvotes', 'partyname', 'totalvalidvotes'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{path.name}: missing columns {sorted(required - set(frame.columns))}')
    frame = frame[frame.ballottype.str.strip().str.upper().eq('PR')].copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f'{path.name}: empty PR data or missing required values')
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    for col in ['registeredvoters', 'spoiltvotes', 'totalvalidvotes']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col]).all() or (frame[col] < 0).any():
            raise ValueError(f'{path.name}: invalid {col}')
    for col in ['ward', 'partyname']:
        frame[col] = frame[col].str.strip()
    if frame.duplicated(['votingdistrict', 'partyname']).any():
        raise ValueError(f'{path.name}: conflicting district/party records')
    invariants = frame.groupby('votingdistrict')[['ward', 'registeredvoters', 'spoiltvotes']].nunique()
    if (invariants > 1).any().any():
        raise ValueError(f'{path.name}: inconsistent district-level totals or ward assignment')
    frame['party'] = frame.partyname.str.upper().map(ALIASES).fillna('Other')
    frame['year'] = year
    return frame, duplicate_count


def aggregate(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Registration and spoilt votes repeat for every party: count each VD once.
    stations = frame.drop_duplicates(['year', 'votingdistrict'])
    stats = stations.groupby(keys)[['registeredvoters', 'spoiltvotes']].sum()
    stats['districts'] = stations.groupby(keys).size()
    stats['valid_votes'] = frame.groupby(keys).totalvalidvotes.sum()
    stats['ballots'] = stats.valid_votes + stats.spoiltvotes
    stats['turnout'] = stats.ballots.div(stats.registeredvoters.replace(0, np.nan)) * 100
    votes = frame.groupby(keys + ['party']).totalvalidvotes.sum().unstack('party')
    votes = votes.reindex(columns=PARTIES).fillna(0).stack().rename('votes').reset_index()
    votes = votes.merge(stats[['valid_votes']].reset_index(), on=keys, validate='many_to_one')
    votes['share'] = votes.votes.div(votes.valid_votes.replace(0, np.nan)) * 100
    return votes, stats.reset_index()


def load_data(directory: Path) -> ElectionData:
    frames, duplicates = {}, {}
    for year in YEARS:
        frames[year], duplicates[year] = read_results(directory / f'ETH_PR_{year}.csv', year)
    mapping = frames[2021].drop_duplicates('votingdistrict').set_index('votingdistrict').ward
    aligned, audits, anomalies = [], [], []
    for year, frame in frames.items():
        frame['reference_ward'] = frame.votingdistrict.map(mapping)
        matched = frame.reference_ward.notna()
        audits.append({
            'Year': year, 'PR rows': len(frame), 'Source wards': frame.ward.nunique(),
            'Voting districts': frame.votingdistrict.nunique(),
            'Mapped vote coverage (%)': 100 * frame.loc[matched, 'totalvalidvotes'].sum() / frame.totalvalidvotes.sum(),
            'Unmapped districts': frame.loc[~matched, 'votingdistrict'].nunique(),
            'Unmapped valid votes': frame.loc[~matched, 'totalvalidvotes'].sum(),
            'Duplicate rows removed': duplicates[year],
        })
        districts = frame.groupby('votingdistrict').agg(
            registered=('registeredvoters', 'first'), spoilt=('spoiltvotes', 'first'),
            valid=('totalvalidvotes', 'sum'))
        bad = districts[districts.valid + districts.spoilt > districts.registered].copy()
        bad['year'] = year
        anomalies.append(bad.reset_index())
        # Unmatched IDs stay in metro history, but are not assigned a guessed ward.
        ward_frame = frame.loc[matched].copy()
        ward_frame['ward'] = ward_frame.reference_ward
        aligned.append(ward_frame)
    all_rows = pd.concat(frames.values(), ignore_index=True)
    metro_votes, metro_stats = aggregate(all_rows, ['year'])
    ward_votes, ward_stats = aggregate(pd.concat(aligned, ignore_index=True), ['year', 'ward'])
    return ElectionData(ward_votes, ward_stats, metro_votes, metro_stats,
                        pd.DataFrame(audits), pd.concat(anomalies, ignore_index=True))


def shares_at(data: ElectionData, year: int) -> pd.DataFrame:
    return data.ward_votes.query('year == @year').pivot(index='ward', columns='party', values='share').reindex(columns=PARTIES)


def predict(data: ElectionData, earlier: int, latest: int, growth: float = 2.,
            turnout_shift: float = 0., spoilt_rate: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed notebook parameters: 50/50 local-metro swing; 70/30 turnout.

Only the two supplied cycles contribute vote/turnout features. A ward missing
an earlier observation carries its latest shares and turnout forward.
"""
    old, recent = shares_at(data, earlier), shares_at(data, latest)
    old = old.reindex(recent.index)
    available = old.notna().all(axis=1)
    old = old.fillna(recent)
    metro = data.metro_votes.pivot(index='year', columns='party', values='share').reindex(columns=PARTIES)
    swing = .5 * (recent - old) + .5 * (metro.loc[latest] - metro.loc[earlier])
    projected = (recent + swing).clip(0, 100)
    projected.loc[~available] = recent.loc[~available]
    projected = projected.div(projected.sum(axis=1), axis=0) * 100
    base = data.ward_stats.query('year == @latest').set_index('ward')
    prior = data.ward_stats.query('year == @earlier').set_index('ward').reindex(base.index)
    stats = pd.DataFrame(index=base.index)
    stats['registeredvoters'] = base.registeredvoters * (1 + growth / 100)
    stats['turnout'] = (.7 * base.turnout + .3 * prior.turnout.fillna(base.turnout) + turnout_shift).clip(0, 100)
    stats['ballots'] = stats.registeredvoters * stats.turnout / 100
    stats['valid_votes'] = stats.ballots * (1 - spoilt_rate / 100)
    stats['fallback'] = ~available
    votes = projected.rename_axis(columns='party').stack().rename('share').reset_index()
    votes = votes.merge(stats[['valid_votes']].reset_index(), on='ward', validate='many_to_one')
    votes['votes'] = votes.share / 100 * votes.valid_votes
    return votes, stats.reset_index()


def metro_projection(votes: pd.DataFrame) -> pd.DataFrame:
    metro = votes.groupby('party', as_index=False).votes.sum()
    metro['share'] = metro.votes / metro.votes.sum() * 100
    return metro


def leaders(votes: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ward, group in votes.groupby('ward'):
        # Other is an aggregate, never an individual party or candidate winner.
        ranking = group[group.party.ne('Other')].sort_values(['share', 'party'], ascending=[False, True])
        first, second = ranking.iloc[0], ranking.iloc[1]
        other = float(group.loc[group.party.eq('Other'), 'share'].iloc[0])
        ambiguous = other >= first.share or np.isclose(first.share, second.share)
        records.append({'ward': ward, 'leader': 'Unresolved' if ambiguous else first.party,
                        'share': first.share, 'margin': first.share - second.share,
                        'other_share': other})
    return pd.DataFrame(records)


def error_metrics(actual, predicted) -> tuple[float, float]:
    errors = np.asarray(predicted) - np.asarray(actual)
    return float(np.abs(errors).mean()), float(np.sqrt(np.square(errors).mean()))


def backtest(data: ElectionData) -> dict:
    projected, projected_stats = predict(data, 2011, 2016)
    actual = data.ward_votes.query('year == 2021')[['ward', 'party', 'share']]
    baseline = data.ward_votes.query('year == 2016')[['ward', 'party', 'share']]
    eligible = set(shares_at(data, 2011).dropna().index) & set(shares_at(data, 2016).dropna().index) & set(shares_at(data, 2021).dropna().index)
    rows = actual[actual.ward.isin(eligible)].rename(columns={'share': 'actual'}).merge(
        projected[['ward', 'party', 'share']].rename(columns={'share': 'predicted'}), on=['ward', 'party']).merge(
        baseline.rename(columns={'share': 'baseline'}), on=['ward', 'party'])
    metrics = []
    for party, group in rows.groupby('party'):
        m, r = error_metrics(group.actual, group.predicted)
        bm, br = error_metrics(group.actual, group.baseline)
        metrics.append({'Party': party, 'Model MAE (pp)': m, 'Baseline MAE (pp)': bm,
                        'Model RMSE (pp)': r, 'Baseline RMSE (pp)': br})
    turnout = data.ward_stats.query('year == 2021 and ward in @eligible')[['ward', 'turnout']].rename(columns={'turnout': 'actual'})
    turnout = turnout.merge(projected_stats[['ward', 'turnout']].rename(columns={'turnout': 'predicted'}), on='ward').merge(
        data.ward_stats.query('year == 2016')[['ward', 'turnout']].rename(columns={'turnout': 'baseline'}), on='ward')
    def winner_frame(column):
        return leaders(rows[['ward', 'party', column]].rename(columns={column: 'share'}))[['ward', 'leader']].rename(columns={'leader': column})
    winners = winner_frame('actual').merge(winner_frame('predicted'), on='ward').merge(winner_frame('baseline'), on='ward')
    classification = []
    labels = sorted(set(winners.actual) | set(winners.predicted) | set(winners.baseline))
    for method in ['predicted', 'baseline']:
        for label in labels:
            tp = ((winners.actual == label) & (winners[method] == label)).sum()
            support = (winners.actual == label).sum()
            predicted_count = (winners[method] == label).sum()
            precision = tp / predicted_count if predicted_count else 0.
            recall = tp / support if support else 0.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            classification.append({'Method': method, 'Party': label, 'Precision': precision,
                                   'Recall': recall, 'F1': f1, 'Support': int(support)})
    confusion = pd.crosstab(winners.actual, winners.predicted).reindex(index=labels, columns=labels, fill_value=0)
    return {'rows': rows, 'party_metrics': pd.DataFrame(metrics), 'turnout': turnout,
            'winners': winners, 'classification': pd.DataFrame(classification),
            'confusion': confusion, 'ward_count': len(eligible)}
