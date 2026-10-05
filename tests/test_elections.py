from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from election_data import PARTIES, backtest, leaders, load_data, metro_projection, predict, read_results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def data():
    return load_data(ROOT)


def test_pr_totals_and_no_double_counting(data):
    totals = data.metro_stats.set_index('year')
    assert totals.valid_votes.to_dict() == {2011: 973728, 2016: 1104552, 2021: 774736}
    assert totals.registeredvoters.to_dict() == {2011: 1666549, 2016: 1919724, 2021: 1909125}
    assert totals.spoiltvotes.to_dict() == {2011: 12864, 2016: 29978, 2021: 18087}
    assert np.allclose(data.metro_votes.groupby('year').share.sum(), 100)
    assert np.allclose(data.ward_votes.groupby(['year', 'ward']).share.sum(), 100)
    assert len(data.anomalies) == 2


def test_unmapped_votes_reconcile_without_guessed_wards(data):
    metro = data.metro_votes.groupby('year').votes.sum()
    wards = data.ward_votes.groupby('year').votes.sum()
    missing = data.audit.set_index('Year')['Unmapped valid votes']
    pd.testing.assert_series_equal(metro - wards, missing, check_names=False, check_dtype=False)
    assert data.ward_stats.query('year == 2021').ward.nunique() == 111


def test_projection_bounds_and_volume_conservation(data):
    for growth, shift, spoilt in [(2, 0, 1.5), (-10, -20, 0), (20, 20, 5)]:
        votes, stats = predict(data, 2016, 2021, growth, shift, spoilt)
        assert votes.share.between(0, 100).all()
        assert stats.turnout.between(0, 100).all()
        assert np.allclose(votes.groupby('ward').share.sum(), 100)
        assert np.isclose(votes.votes.sum(), stats.valid_votes.sum())
        assert np.isclose(metro_projection(votes).share.sum(), 100)


def test_future_vote_values_do_not_leak_into_backtest_prediction(data):
    before, _ = predict(data, 2011, 2016)
    altered = deepcopy(data)
    for frame in [altered.ward_votes, altered.metro_votes]:
        frame.loc[frame.year == 2021, ['share', 'votes']] = 123456
    altered.ward_stats.loc[altered.ward_stats.year == 2021, 'turnout'] = 0
    after, _ = predict(altered, 2011, 2016)
    pd.testing.assert_frame_equal(before, after)


def test_missing_earlier_ward_carries_forward(data):
    altered = deepcopy(data)
    ward = altered.ward_votes.query('year == 2021').ward.iloc[0]
    altered.ward_votes = altered.ward_votes[~((altered.ward_votes.year == 2016) & (altered.ward_votes.ward == ward))]
    altered.ward_stats = altered.ward_stats[~((altered.ward_stats.year == 2016) & (altered.ward_stats.ward == ward))]
    votes, stats = predict(altered, 2016, 2021)
    recent = data.ward_votes.query('year == 2021 and ward == @ward').set_index('party').share.sort_index()
    pd.testing.assert_series_equal(votes.query('ward == @ward').set_index('party').share.sort_index(), recent)
    assert stats.query('ward == @ward').fallback.iloc[0]


def test_backtest_errors_and_confusion_reconcile(data):
    result = backtest(data)
    assert result['ward_count'] == 110
    assert result['confusion'].to_numpy().sum() == 110
    assert len(result['rows']) == 110 * 5
    for party, frame in result['rows'].groupby('party'):
        actual_mae = (frame.predicted - frame.actual).abs().mean()
        reported = result['party_metrics'].set_index('Party').loc[party, 'Model MAE (pp)']
        assert actual_mae == pytest.approx(reported)


def test_other_and_ties_are_never_declared_party_winners():
    votes = pd.DataFrame({'ward': ['A'] * 5, 'party': PARTIES, 'share': [20, 20, 10, 10, 40]})
    assert leaders(votes).leader.iloc[0] == 'Unresolved'
    votes['share'] = [30, 30, 20, 10, 10]
    assert leaders(votes).leader.iloc[0] == 'Unresolved'


def test_inconsistent_station_totals_fail_clearly(tmp_path):
    frame = pd.DataFrame({'BallotType': ['PR', 'PR'], 'Ward': ['Ward 1'] * 2,
                          'VotingDistrict': [1, 1], 'PartyName': ['A', 'B'],
                          'RegisteredVoters': [100, 101], 'SpoiltVotes': [1, 1],
                          'TotalValidVotes': [40, 20]})
    path = tmp_path / 'bad.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='inconsistent district'):
        read_results(path, 2021)


def test_dashboard_navigation_filters_and_scenarios():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'The metro, across elections'
    default_metrics = [item.value for item in app.metric]
    app.multiselect[0].set_value(['DA']).run()
    assert not app.exception
    assert [item.value for item in app.metric] == default_metrics
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any('Select at least one party' in item.value for item in app.info)
    app.multiselect[0].set_value(PARTIES).run()
    app.slider(key='turnout_shift').set_value(10.).run()
    changed_metrics = [item.value for item in app.metric]
    assert changed_metrics[:3] == default_metrics[:3]
    assert changed_metrics[4] != default_metrics[4]
    app.button[0].click().run()
    assert not app.exception
    assert app.slider(key='turnout_shift').value == 0.
    app.radio[0].set_value('Ward explorer').run()
    assert not app.exception
    app.selectbox[1].set_value('Ward 59500111').run()
    app.selectbox[0].set_value(2011).run()
    assert not app.exception
    assert app.title[0].value == 'Ward 111'
    app.radio[0].set_value('Model performance').run()
    assert not app.exception
    performance_metrics = [item.value for item in app.metric]
    app.slider(key='turnout_shift').set_value(-10.).run()
    assert [item.value for item in app.metric] == performance_metrics
    app.radio[0].set_value('Limitations & uncertainty').run()
    assert not app.exception
    assert len(app.dataframe) == 2
