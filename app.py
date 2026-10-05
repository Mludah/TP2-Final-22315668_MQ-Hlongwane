"""Run with: python -m streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from election_data import (PARTIES, PARTY_NAMES, YEARS, backtest, error_metrics,
                           leaders, load_data, metro_projection, predict)

ROOT = Path(__file__).resolve().parent
COLORS = {'ANC': '#168A6D', 'DA': '#2775CA', 'EFF': '#D5515A', 'IFP': '#BD8426', 'Other': '#8792A6'}
st.set_page_config(page_title='eThekwini | Election Observatory', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width:1440px;padding-top:2.2rem;}
h1 {letter-spacing:-1.4px;font-weight:750 !important;}
h2 {letter-spacing:-.6px;}
[data-testid="stMetric"] {background:white;border:1px solid #E2E8F0;border-radius:12px;padding:16px;}
[data-testid="stMetricLabel"] {color:#526581;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:2px;color:#087F8C;margin-bottom:10px;}
.period {padding:12px 16px;border-radius:9px;font-weight:650;margin:12px 0 18px;}
.history {background:#E4F2F4;color:#086575;border-left:4px solid #087F8C;}
.prediction {background:#FFF0DB;color:#855415;border-left:4px solid #C48729;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner='Preparing election results and historical backtest…')
def prepare(signature):
    data = load_data(ROOT)
    return data, backtest(data)


try:
    signature = tuple((ROOT / f'ETH_PR_{y}.csv').stat().st_mtime_ns for y in YEARS)
    data, evaluation = prepare(signature)
except (OSError, ValueError) as exc:
    st.error(f'Could not load election data: {exc}')
    st.info('Place the three original ETH_PR_2011.csv, ETH_PR_2016.csv and ETH_PR_2021.csv files beside app.py.')
    st.stop()

with st.sidebar:
    st.markdown('### ◈ Election Observatory')
    st.caption('ETHEKWINI METRO · PR BALLOTS')
    page = st.radio('Explore', ['Metro overview', 'Ward explorer', 'Model performance', 'Limitations & uncertainty'])
    st.divider()
    selected = st.multiselect('Party filter', PARTIES, default=PARTIES,
                             help='Applies to party charts, result tables and downloads in metro and ward views. Shares always use all valid votes.')
    st.caption('Other combines every party outside ANC, DA, EFF and IFP.')
    year = st.selectbox('Historical election', YEARS, index=2)
    wards = sorted(data.ward_stats.query('year == 2021').ward.unique())
    ward = st.selectbox('Ward selector', wards, format_func=lambda w: f'Ward {int(w.split()[-1]) % 1000} · {w.split()[-1]}')
    with st.expander('2026 scenario assumptions'):
        st.caption('Adjust turnout and vote volumes. Party-share rules stay fixed; metro shares can change when ward turnout weights change.')
        growth = st.slider('Registration growth since 2021 (%)', -10., 20., 2., .5, key='growth')
        turnout_shift = st.slider('Turnout adjustment (percentage points)', -20., 20., 0., .5, key='turnout_shift')
        spoilt = st.slider('Spoilt ballots (%)', 0., 5., 1.5, .1, key='spoilt')
        if st.button('Reset assumptions'):
            # Button callback is unnecessary: remove keys before the next run.
            for key in ['growth', 'turnout_shift', 'spoilt']:
                del st.session_state[key]
            st.rerun()
    st.divider()
    st.caption('Sources: supplied 2011, 2016 and 2021 election files. Forecast geography: 2021 ward reference.')

forecast_votes, forecast_stats = predict(data, 2016, 2021, growth, turnout_shift, spoilt)
metro_forecast = metro_projection(forecast_votes)
scenario = growth != 2. or turnout_shift != 0. or spoilt != 1.5


def filtered(frame):
    return frame[frame.party.isin(selected)].copy()


def chart(fig, key=None):
    fig.update_layout(font=dict(family='Arial', color='#172B4D'), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=12, r=18, t=25, b=20),
                      legend_title_text='', legend=dict(orientation='h', y=-.2), height=350)
    fig.update_yaxes(gridcolor='#E2E8F0')
    st.plotly_chart(fig, width='stretch', key=key, config={'displayModeBar': False})


def bars(frame, projected=False):
    shown = filtered(frame).sort_values('share', ascending=False)
    if shown.empty:
        st.info('Select at least one party in the sidebar to display results.')
        return
    fig = px.bar(shown, x='party', y='share', color='party', color_discrete_map=COLORS,
                 text=shown.share.map(lambda v: f'{v:.1f}%'), labels={'share': 'Share of all valid PR votes (%)', 'party': ''})
    if projected:
        fig.update_traces(marker_pattern_shape='/')
    fig.update_traces(textposition='outside', cliponaxis=False, hovertemplate='%{x}: %{y:.2f}%<extra></extra>')
    fig.update_yaxes(range=[0, min(105, max(60, shown.share.max() + 12))])
    fig.add_hline(y=50, line_dash='dot', line_color='#AAB4C4', annotation_text='50% PR share')
    fig.update_layout(showlegend=False)
    chart(fig)


def title(name, subtitle):
    st.markdown('<div class="eyebrow">ETHEKWINI / ELECTION OBSERVATORY</div>', unsafe_allow_html=True)
    st.title(name)
    st.caption(subtitle)


def period(predicted=False):
    text = '2026 · MODEL PREDICTION' + (' · ADJUSTED SCENARIO' if scenario else '') if predicted else f'{year} · HISTORICAL RESULTS'
    css = 'prediction' if predicted else 'history'
    st.markdown(f'<div class="period {css}">{text}</div>', unsafe_allow_html=True)


def metrics(stats):
    registered, ballots, valid = stats.registeredvoters.sum(), stats.ballots.sum(), stats.valid_votes.sum()
    a, b, c = st.columns(3)
    a.metric('Registered voters', f'{registered:,.0f}')
    b.metric('PR turnout', f'{100 * ballots / registered:.1f}%' if registered else 'Unavailable')
    c.metric('Valid PR votes', f'{valid:,.0f}')


def results_table(history, projection, filename):
    out = history[['party', 'votes', 'share']].merge(projection[['party', 'votes', 'share']], on='party', suffixes=('_history', '_2026'))
    out['change_pp'] = out.share_2026 - out.share_history
    out = filtered(out).rename(columns={'party': 'Party', 'votes_history': f'{year} votes', 'share_history': f'{year} share (%)',
        'votes_2026': '2026 estimated votes', 'share_2026': '2026 projected share (%)', 'change_pp': 'Change (pp)'})
    st.dataframe(out.round(2), hide_index=True, width='stretch')
    export = out.copy()
    export['Forecast status'] = 'Adjusted scenario' if scenario else 'Historical-data model prediction'
    export['Registration growth (%)'] = growth
    export['Turnout adjustment (pp)'] = turnout_shift
    export['Spoilt ballots (%)'] = spoilt
    export['Geography'] = '2021 ward reference'
    st.download_button('Download displayed comparison', export.to_csv(index=False).encode('utf-8-sig'), filename, 'text/csv', disabled=out.empty)
    st.caption('Party filters never change the denominator, overall totals or leading-party calculations. Estimates are rounded for display.')


def history_chart(history):
    shown = filtered(history)
    if shown.empty:
        st.info('Select parties to display historical trends.')
        return
    fig = px.line(shown, x='year', y='share', color='party', markers=True, color_discrete_map=COLORS,
                  labels={'share': 'PR vote share (%)', 'year': ''})
    fig.update_xaxes(tickvals=list(YEARS))
    chart(fig)


def uncertainty(projection, is_metro=False):
    st.subheader('How much could the estimate move?')
    mae = evaluation['party_metrics'].set_index('Party')['Model MAE (pp)']
    view = filtered(projection).copy()
    view['mae'] = view.party.map(mae)
    view['low'] = (view.share - view.mae).clip(0, 100)
    view['high'] = (view.share + view.mae).clip(0, 100)
    if not view.empty:
        fig = go.Figure(go.Scatter(x=view.share, y=view.party, mode='markers', marker=dict(size=12, color='#BD8426'),
            error_x=dict(type='data', symmetric=False, array=view.high - view.share, arrayminus=view.share - view.low),
            customdata=np.column_stack([view.low, view.high]),
            hovertemplate='%{y}: %{x:.1f}%<br>Sensitivity: %{customdata[0]:.1f}–%{customdata[1]:.1f}%<extra></extra>'))
        fig.update_xaxes(range=[0, 100], title='Projected share ± historical ward MAE (percentage points)')
        chart(fig)
    st.caption('Sensitivity bars use each party’s mean absolute ward-level error in the 2021 backtest, clipped to 0–100%. '
               'They are not calibrated confidence intervals or probabilities, and their endpoints do not sum to 100%. '
               + ('Applying ward errors to metro shares is illustrative; these are not measured metro uncertainty bounds.' if is_metro else 'A narrow lead can reverse within these historical error scales.'))


def forecast_warning():
    st.warning('Experimental 2026 projection from 2011–2021 data only. No newer election results, new-party estimates or verified 2026 boundaries are included. PR shares do not establish council control or ward-candidate winners.')


if page == 'Metro overview':
    title('The metro, across elections', 'Historical PR results and a transparent 2026 scenario for eThekwini.')
    forecast_warning()
    history = data.metro_votes.query('year == @year')
    left, right = st.columns(2, gap='large')
    with left:
        period()
        metrics(data.metro_stats.query('year == @year'))
        bars(history)
    with right:
        period(True)
        metrics(forecast_stats)
        bars(metro_forecast, True)
    named = metro_forecast[metro_forecast.party.ne('Other')].sort_values('share', ascending=False).iloc[0]
    st.info(f'Largest projected named party: {named.party} · {named.share:.1f}% of valid PR votes. The 50% guide is a vote-share reference, not a seat or governing-majority prediction.')
    with st.expander('Compare results and download', expanded=True):
        results_table(history, metro_forecast, f'ethekwini_{year}_vs_2026.csv')
    a, b = st.columns(2, gap='large')
    with a:
        st.subheader('Historical party support')
        history_chart(data.metro_votes)
    with b:
        st.subheader('Historical turnout')
        fig = px.bar(data.metro_stats, x='year', y='turnout', text_auto='.1f', labels={'turnout': 'PR turnout (%)', 'year': ''}, color_discrete_sequence=['#087F8C'])
        fig.update_xaxes(type='category')
        fig.update_yaxes(range=[0, 100])
        chart(fig)
    uncertainty(metro_forecast, True)

elif page == 'Ward explorer':
    title(f'Ward {int(ward.split()[-1]) % 1000}', f'{ward} · Historical voting districts mapped to the 2021 ward reference.')
    forecast_warning()
    history = data.ward_votes.query('year == @year and ward == @ward')
    projection = forecast_votes.query('ward == @ward')
    historic_stats = data.ward_stats.query('year == @year and ward == @ward')
    left, right = st.columns(2, gap='large')
    with left:
        period()
        if history.empty:
            st.info('No mapped observations for this ward in the selected historical year.')
        else:
            metrics(historic_stats)
            bars(history)
    with right:
        period(True)
        metrics(forecast_stats.query('ward == @ward'))
        bars(projection, True)
    leader = leaders(projection).iloc[0]
    a, b = st.columns(2)
    a.metric('Projected PR leader among named parties', leader.leader)
    b.metric('Lead over next named party', f'{leader.margin:.1f} pp')
    st.caption('Other is a combined category, never a party winner. Unresolved means a tied lead or Other collectively exceeds the leading named party. These are not ward-councillor predictions.')
    if leader.margin < 10 or leader.leader == 'Unresolved':
        st.warning('Close or unresolved projection. The 10-point flag is a descriptive threshold, not a probability of winning.')
    if not history.empty:
        results_table(history, projection, f'{ward.replace(" ", "_")}_{year}_vs_2026.csv')
    st.subheader('This ward through time')
    history_chart(data.ward_votes.query('ward == @ward'))
    uncertainty(projection)
    with st.expander('Browse all projected wards'):
        summary = leaders(forecast_votes)
        summary['Close lead (<10 pp)'] = summary.margin < 10
        st.dataframe(summary.rename(columns={'ward': 'Ward', 'leader': 'PR leader', 'share': 'Leading named share (%)', 'margin': 'Named-party margin (pp)', 'other_share': 'Other share (%)'}).round(2), hide_index=True, width='stretch')
        st.caption('All-party context: this summary is not restricted by the party filter.')

elif page == 'Model performance':
    title('Tested against the past', 'A fixed 2021 backtest: use 2011 and 2016 vote and turnout data, then compare with observed 2021 results.')
    st.info(f'{evaluation["ward_count"]} wards with observations in all three cycles. Equal weight per ward and party. Sidebar filters and 2026 scenario settings do not change this evaluation.')
    st.caption('Geography is retrospectively aligned using the 2021 district-to-ward lookup. No 2021 vote or turnout values enter prediction features; the geography was not available at the historical forecast origin. This is one retrospective holdout, not rolling validation.')
    rows, winners = evaluation['rows'], evaluation['winners']
    model_mae, model_rmse = error_metrics(rows.actual, rows.predicted)
    baseline_mae, baseline_rmse = error_metrics(rows.actual, rows.baseline)
    a, b, c, d = st.columns(4)
    a.metric('Share MAE', f'{model_mae:.2f} pp')
    b.metric('Persistence MAE', f'{baseline_mae:.2f} pp')
    c.metric('Share RMSE', f'{model_rmse:.2f} pp')
    d.metric('PR leader accuracy', f'{100 * (winners.actual == winners.predicted).mean():.1f}%')
    if model_mae >= baseline_mae:
        st.warning('The swing model does not beat carrying forward the previous election on overall share MAE. Its 2026 output should be treated as an exploratory scenario.')
    else:
        st.success('The swing model has lower share MAE than persistence on this holdout. One cycle is insufficient to establish future reliability.')
    st.subheader('Party-share errors')
    st.caption('MAE = average absolute error. RMSE penalises larger misses more strongly. Both are in percentage points; lower is better.')
    st.dataframe(evaluation['party_metrics'].round(3), hide_index=True, width='stretch')
    a, b = st.columns(2, gap='large')
    with a:
        st.subheader('Predicted versus observed')
        fig = px.scatter(rows, x='actual', y='predicted', color='party', color_discrete_map=COLORS, hover_data=['ward'], labels={'actual': 'Observed 2021 share (%)', 'predicted': 'Predicted 2021 share (%)'})
        fig.add_shape(type='line', x0=0, y0=0, x1=100, y1=100, line=dict(color='#8792A6', dash='dot'))
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])
        chart(fig)
    with b:
        st.subheader('PR leader confusion matrix')
        cm = evaluation['confusion']
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=['#EEF5F7', '#087F8C'], labels=dict(x='Predicted leader', y='Observed leader', color='Wards'), aspect='auto')
        chart(fig)
    st.subheader('Leader classification')
    report = evaluation['classification'].copy()
    report['Method'] = report.Method.map({'predicted': 'Swing model', 'baseline': 'Persistence'})
    st.dataframe(report.round(3), hide_index=True, width='stretch')
    st.caption(f'Persistence leader accuracy: {100 * (winners.actual == winners.baseline).mean():.1f}%. Metrics classify the PR leader among the four named parties, with an Unresolved category when needed; they do not evaluate elected ward candidates. Zero-support classes use zero precision/recall/F1 when undefined.')
    st.subheader('Turnout errors')
    turnout_rows = []
    for method, label in [('predicted', '70% recent + 30% previous'), ('baseline', 'Previous election turnout')]:
        mae, rmse = error_metrics(evaluation['turnout'].actual, evaluation['turnout'][method])
        turnout_rows.append({'Method': label, 'MAE (pp)': mae, 'RMSE (pp)': rmse})
    st.dataframe(pd.DataFrame(turnout_rows).round(3), hide_index=True, width='stretch')
    st.download_button('Download backtest observations', rows.to_csv(index=False).encode('utf-8-sig'), '2021_backtest.csv', 'text/csv')

else:
    title('Read the forecast with care', 'What the data supports, how the model works, and where uncertainty remains.')
    forecast_warning()
    st.subheader('What the model assumes')
    st.markdown('''
- **Party shares:** latest ward share plus 50% of the local swing and 50% of the metro swing. Clip to 0–100%, then rescale all five categories to total 100%.
- **Turnout:** 70% of the latest ward turnout plus 30% of the previous turnout, bounded to 0–100%. Missing earlier wards carry forward their latest observations.
- **Vote volumes:** projected registrations × turnout × the assumed valid-ballot fraction. Defaults are 2% registration growth and 1.5% spoilt ballots; these are assumptions, not measured 2026 facts.
- **Party groups:** ANC, DA, EFF and IFP are kept separate. Every other supplied party is included in Other consistently across years, preserving all votes.
''')
    st.subheader('Limitations & uncertainty')
    st.markdown('''
**Only three election cycles.** A single 2021 holdout cannot establish stable future accuracy. The model has no demographic, polling, campaign or economic inputs. Ward errors are correlated and are not independent future election samples.

**No post-2021 results or new-party estimates.** New entrants absent from the supplied data cannot be estimated separately. Missing support is not evidence of zero support. The 2026 projection cannot capture later political realignment.

**Reference geography, not verified 2026 wards.** Older voting districts are matched by ID to their 2021 ward. This is not a spatial boundary crosswalk and does not prove unchanged district boundaries. Unmapped historical districts are excluded from ward history but retained in metro history. No 2026 boundary files or coordinates were supplied, so the app does not draw a geographical map.

**PR ballots only.** Ward-level views aggregate party-list votes within wards. They do not forecast ward candidates, independent candidates, seat allocations, coalitions or council control. Other combines multiple parties and cannot be declared a winner.

**Sensitivity, not confidence.** Error bars are ± each party’s observed ward MAE in the 2021 backtest. They have no claimed coverage probability, do not quantify all sources of uncertainty and are not simultaneous bounds. Metro bars reuse ward errors for illustration. Scenario controls also do not represent probability distributions.

**Turnout and registration assumptions matter.** A change in turnout changes estimated ballots and party vote volumes. Metro shares are weighted by projected valid votes, so changes to ward turnout can also change the metro mix. Forecast share comparisons retain the full electorate denominator when parties are hidden.
''')
    turnout_mae, _ = error_metrics(evaluation['turnout'].actual, evaluation['turnout'].predicted)
    st.info(f'Historical turnout MAE: {turnout_mae:.2f} percentage points across wards. This large error is a useful scale for stress-testing turnout assumptions, not a forecast confidence interval.')
    st.subheader('Data coverage and quality')
    st.dataframe(data.audit.round(3), hide_index=True, width='stretch')
    st.caption('Source totals are recomputed from the supplied exports and have not been independently reconciled with official published totals. Registered voters and spoilt votes are counted once per district. Exact duplicate rows are removed; conflicting duplicates stop loading.')
    st.markdown(f'**{len(data.anomalies)} district-year records have ballots exceeding registered voters.** These source values are retained in historical aggregates and disclosed below; no historical values are silently capped.')
    st.dataframe(data.anomalies.rename(columns={'votingdistrict': 'Voting district', 'registered': 'Registered voters', 'spoilt': 'Spoilt votes', 'valid': 'Valid votes', 'year': 'Year'}), hide_index=True, width='stretch')
    with st.expander('Provenance and implementation notes'):
        st.markdown('The supplied **22315668_HLONGWANE_MQ.ipynb** supplies the swing and turnout approach. The dashboard uses eThekwini, as identified by the source files, despite an early Cape Town reference in the notebook. It recomputes metrics rather than copying notebook constants.')
        st.markdown('Corrections for the dashboard: keep every non-core party in Other in all years; exclude unmapped districts from ward totals rather than guessing; use the same turnout formula in the backtest and forecast; remove the arbitrary 20% forecast turnout floor; label historical MAE bars as sensitivity rather than confidence intervals.')
        st.caption('The raw files and notebook are read-only inputs. Ingestion is cached until a source file changes. Forecast scoring scales with wards × party categories.')

st.divider()
st.caption('eThekwini Election Observatory · Historical observations 2011–2021 · Experimental projection 2026 · Percentage points abbreviated as pp')
