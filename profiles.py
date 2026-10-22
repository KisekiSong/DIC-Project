"""Stage 4: per-user pattern profiles, weekly-structure metrics and the Plotly report."""
import json
from datetime import timedelta
from math import log2
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import Window, functions as F

from clustering import mean_silhouette
from common import HOURS, load_json, read_table, write_table
from weekly import DAY_CLASSES, Tally, by_share, count_rows, summarise, user_entropies, user_shifts

DOW_ORDER = [2, 3, 4, 5, 6, 7, 1]  # Spark dayofweek: 1 = Sunday
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
LIFT_KEYS = DAY_CLASSES + ['low']
LIFT_NAMES = ['Mon-Fri', 'Sat', 'Sun', 'holiday', 'low day']
STYLE = '''
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 16px;
       color: #222; background: #fff; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }
th { width: 32%; color: #555; font-weight: 600; }
section { margin-top: 28px; }
h2 { font-size: 1.1rem; }
'''
PALETTE = ['#2E6FBA', '#E08A1E', '#3A9D5D', '#C8463D', '#7B5BB5', '#8C6D3F', '#D16BA5', '#5B8A99',
           '#B5B23A', '#444444']


def colour_of(cluster):
    return PALETTE[cluster % len(PALETTE)]


def fmt(value, spec='.3f'):
    return '-' if value is None else format(value, spec)


def day_type():
    """weekday / weekend / holiday: the split used for the stored user profiles."""
    return F.when(F.col('day_class').isin('sat', 'sun'), 'weekend').otherwise(F.col('day_class'))


def user_profiles(assigned):
    counts = assigned.groupBy('user_id', day_type().alias('day_type'), 'cluster').agg(F.count('*').alias('days'))
    total = F.sum('days').over(Window.partitionBy('user_id', 'day_type'))
    return counts.withColumn('share', F.col('days') / total)


def dominant_agreement(assigned):
    """Share of (user, weekday/weekend) pairs whose most frequent pattern is the same in both years."""
    regular = assigned.where(F.col('day_class') != 'holiday')
    counts = regular.groupBy('split', 'user_id', day_type().alias('day_type'), 'cluster').count()
    order = Window.partitionBy('split', 'user_id', 'day_type').orderBy(F.desc('count'), 'cluster')
    top = (counts.withColumn('rank', F.row_number().over(order)).where('rank = 1')
           .select('split', 'user_id', 'day_type', 'cluster'))
    train = top.where("split = 'train'").alias('a')
    holdout = top.where("split = 'holdout'").alias('b')
    same = (F.col('a.cluster') == F.col('b.cluster')).cast('double')
    pairs = train.join(holdout, ['user_id', 'day_type'])
    row = pairs.agg(F.avg(same).alias('agree'), F.count('*').alias('pairs')).first()
    return {'agreement': row.agree, 'pairs': row.pairs}


def cluster_quantiles(assigned):
    """5-25-50-75-95th percentiles of daily kWh per cluster, for the consumption box plot."""
    percentiles = F.expr('percentile_approx(daily_kwh, array(0.05, 0.25, 0.5, 0.75, 0.95))')
    rows = assigned.groupBy('cluster').agg(percentiles.alias('q')).collect()
    return {r.cluster: r.q for r in rows}


def dow_counts(assigned):
    rows = assigned.groupBy('dow', 'cluster').count().collect()
    return {(r.dow, r.cluster): r['count'] for r in rows}


def example_users(shifts, entropies):
    """Three contrasting users: most single-pattern, most mixed, and largest weekday/weekend shift."""
    candidates = [min(entropies, key=entropies.get), max(entropies, key=entropies.get),
                  max(shifts, key=shifts.get)]
    return list(dict.fromkeys(candidates))  # keeps the order, drops duplicates


def user_calendar(assigned, user):
    rows = assigned.where(F.col('user_id') == user).select('date', 'cluster').collect()
    return [(r.date, r.cluster) for r in rows]


def collect_report_data(spark, args, k):
    assigned_all = read_table(spark, args.data_root, 'daily_clusters')
    assigned = assigned_all.where("split = 'train'").cache()
    write_table(user_profiles(assigned), args.data_root, 'user_profiles')
    tally = Tally(count_rows(assigned))
    shifts = user_shifts(tally)
    picks = example_users(shifts, user_entropies(tally))
    data = {'weekly': summarise(tally, k), 'shifts': shifts, 'picks': picks,
            'calendars': {user: user_calendar(assigned, user) for user in picks},
            'by_dow': dow_counts(assigned), 'quantiles': cluster_quantiles(assigned),
            'dominant_agreement': dominant_agreement(assigned_all) if args.holdout_year else None}
    assigned.unpersist()
    return data


def latest_stages(output):
    path = Path(output) / 'run_log.jsonl'
    stages = {}
    if not path.exists():
        return stages
    for line in path.read_text().splitlines():
        record = json.loads(line)
        stages[record['stage']] = record
    return stages


def sweep_figure(result):
    runs, stability = result['runs'], result['stability']
    ks = sorted({run['k'] for run in runs})
    means = [mean_silhouette(runs, k) for k in ks]
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Silhouette', 'Seed agreement (ARI)'))
    fig.add_trace(go.Scatter(x=[run['k'] for run in runs], y=[run['silhouette'] for run in runs],
                             mode='markers', name='seed run', marker=dict(color='#888', size=8)), 1, 1)
    fig.add_trace(go.Scatter(x=ks, y=means, mode='lines+markers', name='mean silhouette',
                             line=dict(color=PALETTE[0])), 1, 1)
    fig.add_trace(go.Scatter(x=[s['k'] for s in stability], y=[s['mean_ari'] for s in stability],
                             mode='lines+markers', name='mean pairwise ARI', line=dict(color=PALETTE[1])), 1, 2)
    fig.add_vline(x=result['k'], line_dash='dash', line_color='#999', row='all', col='all')
    fig.update_xaxes(title_text='k', dtick=1)
    return fig


def centre_figure(result, names):
    fig = go.Figure()
    hours = list(HOURS)
    scale = 100 if result['norm'] == 'share' else 1
    for centre in result['centres']:
        cluster = centre['cluster']
        colour = colour_of(cluster)
        examples = [r for r in result['representatives'] if r['cluster'] == cluster]
        for example in examples:
            fig.add_trace(go.Scatter(x=hours, y=[v * scale for v in example['shape']], mode='lines',
                                     line=dict(color=colour, width=1, dash='dot'), opacity=0.6, showlegend=False,
                                     hovertext=f"{example['user_id']} {example['date']}", hoverinfo='text+y'))
        share = result['train_shares'][str(cluster)]
        fig.add_trace(go.Scatter(x=hours, y=[v * scale for v in centre['centre']], mode='lines+markers',
                                 name=f'{names[cluster]} ({share:.0%} of days)', line=dict(color=colour, width=3)))
    y_title = '% of daily kWh' if result['norm'] == 'share' else 'fraction of daily peak'
    fig.update_layout(xaxis_title='hour of day', yaxis_title=y_title, xaxis=dict(dtick=2))
    return fig


def lift_figure(weekly, names):
    """Day-type lift per pattern. Colour is log2(lift), so 2x and 0.5x look equally strong."""
    clusters = by_share(weekly['clusters'])
    labels, colours, text = [], [], []
    for c in clusters:
        lifts = [c['lift'][key] for key in LIFT_KEYS]
        labels.append(f"{names[c['cluster']]} ({c['share']:.0%})")
        colours.append([log2(v) if v else None for v in lifts])
        text.append([fmt(v, '.2f') for v in lifts])
    colorbar = dict(title='lift', tickvals=[-2, -1, 0, 1, 2], ticktext=['0.25', '0.5', '1', '2', '4'])
    fig = go.Figure(go.Heatmap(z=colours, x=LIFT_NAMES, y=labels, text=text, texttemplate='%{text}',
                               colorscale='RdBu', reversescale=True, zmid=0, zmin=-2, zmax=2, colorbar=colorbar))
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=120 + 45 * len(clusters))
    return fig


def consumption_figure(quantiles, k):
    fig = go.Figure()
    for cluster in range(k):
        q = quantiles[cluster]
        fig.add_trace(go.Box(x=[f'C{cluster}'], name=f'C{cluster}', lowerfence=[q[0]], q1=[q[1]], median=[q[2]],
                             q3=[q[3]], upperfence=[q[4]], marker_color=colour_of(cluster), showlegend=False))
    fig.update_layout(yaxis_type='log', yaxis_title='daily kWh (5-25-50-75-95 percentiles)')
    return fig


def dow_figure(by_dow, names, k):
    totals = {}
    for dow in DOW_ORDER:
        totals[dow] = sum(by_dow.get((dow, cluster), 0) for cluster in range(k))
    fig = go.Figure()
    for cluster in range(k):
        shares = [by_dow.get((dow, cluster), 0) / totals[dow] for dow in DOW_ORDER]
        fig.add_trace(go.Bar(x=DOW_NAMES, y=shares, name=names[cluster], marker_color=colour_of(cluster)))
    fig.update_layout(barmode='stack', yaxis_tickformat='.0%', yaxis_title='share of valid days')
    return fig


def shift_figure(shifts):
    fig = go.Figure(go.Histogram(x=list(shifts.values()), nbinsx=20, marker_color='#888'))
    fig.update_layout(xaxis_title='total-variation distance between weekday and weekend mix (0 = same)',
                      yaxis_title='users')
    return fig


def drift_figure(result, k):
    fig = go.Figure()
    years = [('train_shares', result['year'], '#9AA5B1'), ('holdout_shares', result['holdout_year'], '#3E4C59')]
    for key, year, colour in years:
        shares = [result[key].get(str(cluster), 0) for cluster in range(k)]
        fig.add_trace(go.Bar(x=[f'C{cluster}' for cluster in range(k)], y=shares, name=str(year),
                             marker_color=colour))
    fig.update_layout(barmode='group', yaxis_tickformat='.0%', yaxis_title='share of valid days')
    return fig


def calendar_grid(days):
    """7 x 53 grid (weekday x week of year) of cluster ids; empty cells are excluded days."""
    origin = min(day for day, _ in days).replace(month=1, day=1)
    origin -= timedelta(days=origin.weekday())
    grid = [[None] * 53 for _ in range(7)]
    for day, cluster in days:
        week = (day - origin).days // 7
        if week < 53:
            grid[day.weekday()][week] = cluster
    return grid


def calendar_figure(picks, calendars, k):
    steps = []
    for cluster in range(k):
        steps += [[cluster / k, colour_of(cluster)], [(cluster + 1) / k, colour_of(cluster)]]
    colorbar = dict(tickvals=list(range(k)), ticktext=[f'C{cluster}' for cluster in range(k)])
    fig = make_subplots(rows=len(picks), cols=1, subplot_titles=picks, vertical_spacing=0.12)
    for row, user in enumerate(picks, start=1):
        heatmap = go.Heatmap(z=calendar_grid(calendars[user]), y=DOW_NAMES, x=list(range(1, 54)),
                             zmin=-0.5, zmax=k - 0.5, colorscale=steps, showscale=row == 1, colorbar=colorbar,
                             xgap=1, ygap=1)
        fig.add_trace(heatmap, row, 1)
        fig.update_yaxes(autorange='reversed', row=row, col=1)
    fig.update_layout(height=230 * len(picks))
    return fig


def figures(result, data):
    k = result['k']
    names = {c['cluster']: f"C{c['cluster']} {c['label']}" for c in result['centres']}
    figs = [('Choosing k (dashed line: final k)', sweep_figure(result)),
            ('Cluster centres (solid) and closest representative days (dotted)', centre_figure(result, names)),
            ('Which days fall into each pattern (lift over the overall rate)', lift_figure(data['weekly'], names)),
            ('Overall consumption per pattern', consumption_figure(data['quantiles'], k)),
            ('Pattern mix by day of week', dow_figure(data['by_dow'], names, k)),
            ("How much each user's pattern mix changes at weekends", shift_figure(data['shifts']))]
    if 'holdout_shares' in result:
        figs.append(('Pattern frequencies, training year vs holdout year', drift_figure(result, k)))
    calendar = calendar_figure(data['picks'], data['calendars'], k)
    figs.append(('Daily assignments for three contrasting users (week of year x weekday)', calendar))
    return figs


def facts(result, data, stages):
    ingest = stages.get('ingest', {}).get('report', {})
    exclusions = stages.get('features', {}).get('exclusions', {})
    weekly = data['weekly']
    timestamps = [ingest.get(f'{kind}_timestamps') for kind in ('duplicate', 'invalid', 'gap')]
    outcomes = ', '.join(f'{name}: {days:,}' for name, days in exclusions.items())
    excluded = ', '.join(result['excluded_meters']) or 'none'
    silhouettes = f"train {result['train_silhouette']:.3f}"
    if 'holdout_silhouette' in result:
        silhouettes += f", holdout {result['holdout_silhouette']:.3f}"
    training = (f"{result['training_days']:,} valid days, {result['meters']} meters, year {result['year']}; "
                f"meters excluded: {excluded}")
    information = f"{weekly['weekly_mi_bits']:.3f} of {weekly['weekend_entropy_bits']:.2f} bits"
    shifting = (f"{weekly['users_shift_gt_0_2']} of {weekly['users']} users "
                'shift their pattern mix by more than 0.2')
    holiday_rate = fmt(weekly['holiday_to_sunday_pattern'], '.0%')
    weekday_rate = fmt(weekly['weekday_to_sunday_pattern'], '.0%')
    holidays = (f"{holiday_rate}, against {weekday_rate} of ordinary weekdays ({weekly['switchers']} users whose "
                'Sunday pattern differs from their weekday one)')
    chosen = [s for s in result['stability'] if s['k'] == result['k']]
    restarts = '-'
    if chosen:
        restarts = f"{chosen[0]['best_cost_hits']} of {chosen[0]['seeds']} seeds reach the lowest cost"
    rows = [
        ('Readings ingested', f"{ingest.get('readings', 0):,} from {ingest.get('meters', '?')} meters"),
        ('Duplicate / invalid / gap timestamps', ' / '.join(str(t) for t in timestamps)),
        ('Day outcomes (all years)', outcomes),
        ('Training data', training),
        ('Final model', f"k = {result['k']}, seed {result['seed']}, normalisation = {result['norm']}"),
        ('Restart reproducibility', restarts),
        ('Silhouette', silhouettes),
        ('Weekday/weekend separation within users', f'{information}; {shifting}'),
        ('Weekday holidays taking the Sunday pattern', holidays),
    ]
    check = result.get('normalisation_check')
    if check:
        matches = ', '.join(f'C{cluster}: {overlap:.2f}' for cluster, overlap in sorted(check['jaccard'].items()))
        rows.append((f"Same k with {check['norm']} normalisation",
                     f"ARI {check['ari']:.3f}; best-match Jaccard per pattern: {matches}"))
    agree = data['dominant_agreement']
    if agree:
        rows.append(('Dominant pattern unchanged next year',
                     f"{agree['agreement']:.1%} of {agree['pairs']} user x weekday/weekend pairs"))
    rows.append(('Stage times (s)', ', '.join(f"{name}: {r['seconds']}" for name, r in stages.items())))
    return rows


def render(result, data, stages, path):
    table = ''.join(f'<tr><th>{name}</th><td>{value}</td></tr>' for name, value in facts(result, data, stages))
    sections = []
    for n, (title, fig) in enumerate(figures(result, data)):
        fig.update_layout(template='plotly_white', margin=dict(l=50, r=40, t=50, b=40),
                          legend=dict(orientation='h', y=-0.3, yanchor='top'))
        html = fig.to_html(full_html=False, include_plotlyjs='cdn' if n == 0 else False)
        sections.append(f'<section><h2>{title}</h2>{html}</section>')
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Daily Electricity Patterns</title>
<style>{STYLE}</style></head><body><h1>Daily electricity use patterns</h1>
<table>{table}</table>{''.join(sections)}</body></html>""")


def run(spark, args, log):
    result = load_json(Path(args.output) / 'clustering.json')
    with log.stage('profile') as info:
        data = collect_report_data(spark, args, result['k'])
        info.update(picks=data['picks'], dominant_agreement=data['dominant_agreement'],
                    weekly_mi_bits=data['weekly']['weekly_mi_bits'])
    weekly = data['weekly']
    print(f"  users profiled: {weekly['users']}, median weekday/weekend shift {weekly['shift_median']:.3f}")
    print(f"  weekly MI {weekly['weekly_mi_bits']:.4f} bits, holiday -> Sunday pattern "
          f"{fmt(weekly['holiday_to_sunday_pattern'], '.2f')}")
    print(f"  dominant pattern agreement across years: {data['dominant_agreement']}")
    report = Path(args.output) / 'report.html'
    render(result, data, latest_stages(args.output), report)
    print(f'  report: {report}')
