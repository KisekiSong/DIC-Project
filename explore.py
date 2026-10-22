"""Experiment: which normalisation and k separate weekend and closed days, beyond user identity."""
from collections import defaultdict
from pathlib import Path
from statistics import mean

import plotly.graph_objects as go

from clustering import (best_cost_hits, best_match_jaccard, best_run, contingency, describe, fit_seeds,
                        prepare_features, seed_aris, select_days)
from common import HOURS, save_json
from profiles import LIFT_KEYS, LIFT_NAMES, colour_of, fmt
from weekly import Tally, by_share, count_rows, summarise

SUMMARY = ['silhouette', 'mean_ari', 'best_cost_hits', 'weekly_mi_bits', 'user_explains', 'users_shift_gt_0_2',
           'holiday_to_sunday_pattern', 'weekday_to_sunday_pattern']


def seed_jaccard(preds, best_col, other_cols):
    """Per cluster of the chosen seed: best-match Jaccard against each other seed, averaged over those seeds."""
    scores = defaultdict(list)
    for other_col in other_cols:
        matches = best_match_jaccard(contingency(preds, best_col, other_col))
        for cluster, overlap in matches.items():
            scores[cluster].append(overlap)
    return {cluster: mean(overlaps) for cluster, overlaps in scores.items()}


def run_config(features, norm, k, seeds):
    preds, runs = fit_seeds(features, k, seeds)
    aris = seed_aris(preds, seeds)
    best = best_run(runs)
    best_col = f"c{best['seed']}"
    other_cols = [f"c{run['seed']}" for run in runs if run is not best]
    jaccard = seed_jaccard(preds, best_col, other_cols)
    metrics = summarise(Tally(count_rows(preds, best_col)), k)
    preds.unpersist()
    centres = [c.tolist() for c in best['model'].clusterCenters()]
    for cluster in metrics['clusters']:
        centre = centres[cluster['cluster']]
        cluster.update(centre=centre, jaccard=jaccard.get(cluster['cluster']), **describe(centre))
    return {'norm': norm, 'k': k, 'best_seed': best['seed'], 'silhouette': best['silhouette'],
            'silhouettes': [run['silhouette'] for run in runs],
            'mean_ari': mean(aris) if aris else None, 'min_ari': min(aris, default=None),
            'best_cost_hits': best_cost_hits(runs), 'seeds': len(runs),
            'fit_seconds': sum(run['seconds'] for run in runs), **metrics}


def describe_days(days, args, n_meters):
    rows = days.groupBy('day_class', 'low').count().collect()
    total = sum(r['count'] for r in rows)
    holiday = sum(r['count'] for r in rows if r.day_class == 'holiday')
    low = sum(r['count'] for r in rows if r.low)
    return {'year': args.year, 'meters': n_meters, 'days': total, 'excluded': args.exclude_meters,
            'holiday': holiday, 'low': low / total}


def lift_cell(value):
    """Orange above 1, blue below; full colour at lift 3 or 0."""
    if value is None:
        return '<td>-</td>'
    if value >= 1:
        colour = f'rgba(224,138,30,{min(1.0, (value - 1) / 2):.2f})'
    else:
        colour = f'rgba(46,111,186,{1 - value:.2f})'
    return f'<td style="background:{colour}">{value:.2f}</td>'


def summary_row(r):
    cells = [r['norm'], r['k'], fmt(r['silhouette']),
             f"{fmt(r['mean_ari'])} ({fmt(r['min_ari'])})",
             f"{r['best_cost_hits']} / {r['seeds']}",
             fmt(r['weekly_mi_bits'], '.4f'),
             f"{r['users_shift_gt_0_2']} / {r['users']}",
             fmt(r['user_explains'], '.0%'),
             f"{fmt(r['holiday_to_sunday_pattern'], '.0%')} <small>({r['switchers']} users)</small>",
             fmt(r['weekday_to_sunday_pattern'], '.0%')]
    return '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'


def cluster_row(c):
    swatch = f"<span class=sw style='background:{colour_of(c['cluster'])}'></span>"
    lifts = ''.join(lift_cell(c['lift'][key]) for key in LIFT_KEYS)
    meter = f"{c['top_user']} ({c['top_user_share']:.0%})"
    return (f"<tr><td>{swatch}C{c['cluster']} {c['label']}</td><td>{c['share']:.1%}</td>{lifts}"
            f"<td>{fmt(c['jaccard'], '.2f')}</td><td>{meter}</td></tr>")


def centre_figure(result):
    fig = go.Figure()
    for c in by_share(result['clusters']):
        line = dict(color=colour_of(c['cluster']), width=2.5)
        fig.add_trace(go.Scatter(x=list(HOURS), y=c['centre'], mode='lines', line=line,
                                 name=f"C{c['cluster']} ({c['share']:.0%})"))
    y_title = 'share of daily kWh' if result['norm'] == 'share' else 'fraction of daily peak'
    fig.update_layout(template='plotly_white', height=300, margin=dict(l=50, r=10, t=10, b=40),
                      xaxis=dict(title='hour of day', dtick=3), yaxis_title=y_title)
    return fig


def config_section(result, include_plotlyjs):
    chart = centre_figure(result).to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    columns = ['pattern', 'days', *LIFT_NAMES, 'seed Jaccard', 'largest meter']
    header = ''.join(f'<th>{name}</th>' for name in columns)
    rows = ''.join(cluster_row(c) for c in by_share(result['clusters']))
    title = f"{result['norm']} &middot; k = {result['k']}"
    table = f'<div class=scroll><table><tr>{header}</tr>{rows}</table></div>'
    return f'<section><h2>{title}</h2>{chart}{table}</section>'


SUMMARY_HEADER = ['norm', 'k', 'silhouette<br>(lowest-cost seed)', 'ARI mean<br>(min)',
                  'seeds reaching<br>lowest cost', 'weekly MI<br>bits', 'users with<br>shift &gt; 0.2',
                  'user identity<br>explains', 'holiday &rarr;<br>Sunday pattern',
                  'weekday &rarr;<br>Sunday pattern']

DEFINITIONS = """<ul><li><b>Weekly MI</b>: I(cluster; weekend | user) in bits. It measures weekday/weekend
separation <i>within</i> each user, so a clustering that only groups users scores 0.</li>
<li><b>User identity explains</b>: I(cluster; user) / H(cluster), the share of cluster uncertainty removed by
knowing the user.</li>
<li><b>Holiday &rarr; Sunday pattern</b>: for users whose dominant Sunday pattern differs from their weekday one,
the share of their weekday-holiday days assigned to the Sunday pattern. <b>Weekday &rarr; Sunday</b> is the
same rate on ordinary weekdays (the false-alarm baseline).</li>
<li>Lift tables: P(day type | cluster) / P(day type); orange above 1, blue below. Silhouettes compare k within
one normalisation only, because the two feature spaces differ.</li>
<li><b>Seeds reaching lowest cost</b>: restarts whose K-means cost is within 0.1% of the best one. The chosen
model is the lowest-cost seed, so this count says how reliably restarts reproduce it.</li>
<li><b>Seed Jaccard</b>: overlap between the cluster and its best match under each other seed, averaged;
1 means every seed finds the same days. <b>Largest meter</b>: the meter contributing most of the cluster's
days.</li></ul>"""

STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 16px;
       color: #222; background: #fff; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #eee; }
th { color: #555; font-weight: 600; }
.scroll { overflow-x: auto; }
section { margin-top: 28px; }
h2 { font-size: 1.05rem; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
p, li { line-height: 1.5; }
"""


def render(results, base, path):
    header = ''.join(f'<th>{name}</th>' for name in SUMMARY_HEADER)
    rows = ''.join(summary_row(r) for r in results)
    sections = []
    for n, result in enumerate(results):
        sections.append(config_section(result, 'cdn' if n == 0 else False))
    excluded = ', '.join(base['excluded']) or 'none'
    intro = (f"{base['days']:,} valid days from {base['meters']} meters in {base['year']}; {base['holiday']:,} "
             f"fall on weekday public holidays and {base['low']:.1%} are low days (under half the user's median "
             f"daily kWh). Meters excluded from fitting: {excluded}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Normalisation and k Comparison</title>
<style>{STYLE}</style></head><body><h1>Which normalisation and k separate weekend and closed days?</h1>
<p>{intro}</p>{DEFINITIONS}<div class=scroll><table><tr>{header}</tr>{rows}</table></div>
{''.join(sections)}</body></html>""")


def report_dir(args):
    name = 'explore'
    if args.exclude_meters:
        name += '_excl_' + '_'.join(args.exclude_meters)
    return Path(args.output) / name


def run(spark, args, log):
    days, _, n_meters = select_days(spark, args)
    base = describe_days(days, args, n_meters)
    print(f'  {base}')
    results = []
    for norm in args.norms:
        features = prepare_features(spark, days, norm)
        for k in args.ks:
            with log.stage('explore', norm=norm, k=k, seeds=args.seeds) as info:
                result = run_config(features, norm, k, args.seeds)
                info.update({key: result[key] for key in SUMMARY})
            results.append(result)
            summary = ', '.join(f'{key}={fmt(result[key])}' for key in SUMMARY)
            print(f'  {norm} k={k}: {summary}')
        features.unpersist()
    out = report_dir(args)
    save_json({'base': base, 'results': results}, out / 'explore.json')
    render(results, base, out / 'explore.html')
    print(f"  report: {out / 'explore.html'}")
    return results
