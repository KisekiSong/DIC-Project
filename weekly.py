"""Weekly-structure metrics for a day-level clustering.

Everything here works on a small collected table of (user_id, day_class, low, cluster, days) counts, so the main
report and the normalisation/k experiment use the same definitions."""
from collections import defaultdict
from math import log2

DAY_CLASSES = ['weekday', 'sat', 'sun', 'holiday']
WEEKEND = ('sat', 'sun')


def count_rows(assigned, cluster_col='cluster'):
    grouped = assigned.groupBy('user_id', 'day_class', 'low', cluster_col).count()
    return [tuple(r) for r in grouped.collect()]


class Tally:
    """Day counts indexed the ways the metrics need them."""

    def __init__(self, rows):
        self.total = 0
        self.low = 0
        self.by_cluster = defaultdict(int)
        self.by_class = defaultdict(int)
        self.by_cluster_class = defaultdict(int)
        self.low_by_cluster = defaultdict(int)
        self.by_user_cluster = defaultdict(int)
        self.mixes = {}  # user -> day class -> {cluster: days}
        for row in rows:
            self.add(*row)

    def add(self, user, day_class, low, cluster, days):
        self.total += days
        self.by_cluster[cluster] += days
        self.by_class[day_class] += days
        self.by_cluster_class[(cluster, day_class)] += days
        self.by_user_cluster[(user, cluster)] += days
        if low:
            self.low += days
            self.low_by_cluster[cluster] += days
        user_mixes = self.mixes.setdefault(user, {})
        mix = user_mixes.setdefault(day_class, defaultdict(int))
        mix[cluster] += days

    def mix(self, user, *day_classes):
        """One user's {cluster: days} over the given day classes."""
        combined = defaultdict(int)
        for day_class in day_classes:
            for cluster, days in self.mixes[user].get(day_class, {}).items():
                combined[cluster] += days
        return combined


def entropy(counts):
    counts = list(counts)
    total = sum(counts)
    return -sum(n / total * log2(n / total) for n in counts if n)


def mutual_information(joint):
    """I(X;Y) in bits from {(x, y): count}."""
    xs, ys = defaultdict(int), defaultdict(int)
    for (x, y), n in joint.items():
        xs[x] += n
        ys[y] += n
    return entropy(xs.values()) + entropy(ys.values()) - entropy(joint.values())


def total_variation(mix_a, mix_b):
    """Half the L1 distance between two {cluster: days} mixes after turning each into shares (0 = same mix)."""
    total_a = sum(mix_a.values())
    total_b = sum(mix_b.values())
    distance = 0.0
    for cluster in set(mix_a) | set(mix_b):
        distance += abs(mix_a.get(cluster, 0) / total_a - mix_b.get(cluster, 0) / total_b)
    return distance / 2


def dominant(mix):
    """The cluster with the most days, or None for an empty mix."""
    if not mix:
        return None
    return max(mix, key=mix.get)


def ratio(part, whole):
    return part / whole if whole else None


def lift(part, whole, base_part, base_whole):
    """P(type | cluster) / P(type): above 1 means the day type is over-represented in the cluster."""
    if not whole or not base_part:
        return None
    return (part / whole) / (base_part / base_whole)


def quantile(sorted_values, q):
    return sorted_values[int(q * (len(sorted_values) - 1))]


def by_share(clusters):
    """Cluster summaries, largest first."""
    return sorted(clusters, key=lambda c: c['share'], reverse=True)


def largest_meters(tally):
    """cluster -> (meter, days) for the meter contributing the most days to that cluster."""
    top = {}
    for (user, cluster), days in tally.by_user_cluster.items():
        if days > top.get(cluster, (None, 0))[1]:
            top[cluster] = (user, days)
    return top


def cluster_stats(tally, k):
    """Per cluster: share of days, lift on each day class and on low days, and its largest meter."""
    top = largest_meters(tally)
    clusters = []
    for cluster in range(k):
        size = tally.by_cluster[cluster]
        lifts = {}
        for day_class in DAY_CLASSES:
            in_cluster = tally.by_cluster_class[(cluster, day_class)]
            lifts[day_class] = lift(in_cluster, size, tally.by_class[day_class], tally.total)
        lifts['low'] = lift(tally.low_by_cluster[cluster], size, tally.low, tally.total)
        meter, meter_days = top.get(cluster, (None, 0))
        # A cluster mostly made of one meter's days is an outlier sink rather than a shared pattern.
        clusters.append({'cluster': cluster, 'share': size / tally.total, 'lift': lifts,
                         'top_user': meter, 'top_user_share': ratio(meter_days, size)})
    return clusters


def user_shifts(tally):
    """user -> total-variation distance between the user's weekday and weekend pattern mixes."""
    shifts = {}
    for user in tally.mixes:
        weekday = tally.mix(user, 'weekday')
        weekend = tally.mix(user, *WEEKEND)
        if weekday and weekend:
            shifts[user] = total_variation(weekday, weekend)
    return shifts


def user_entropies(tally):
    """user -> entropy (bits) of the user's pattern mix over all days; 0 = always the same pattern."""
    per_user = defaultdict(list)
    for (user, _), days in tally.by_user_cluster.items():
        per_user[user].append(days)
    return {user: entropy(days) for user, days in per_user.items()}


def weekly_information(tally):
    """I(cluster; weekend | user) in bits: weekday/weekend separation *within* users, so user types score 0."""
    regular_days = tally.total - tally.by_class['holiday']
    information = 0.0
    for user in tally.mixes:
        joint = {}
        for cluster, days in tally.mix(user, 'weekday').items():
            joint[('weekday', cluster)] = days
        for cluster, days in tally.mix(user, *WEEKEND).items():
            joint[('weekend', cluster)] = days
        user_days = sum(joint.values())
        if user_days:
            information += user_days / regular_days * mutual_information(joint)
    return information


def holiday_check(tally):
    """For users whose dominant Sunday pattern differs from their weekday one: how often do weekday holidays take
    the Sunday pattern, compared with ordinary weekdays (the false-alarm rate)?"""
    switchers = holidays = holiday_sunday = holiday_weekday = weekdays = weekday_sunday = 0
    for user in tally.mixes:
        weekday = tally.mix(user, 'weekday')
        weekday_top = dominant(weekday)
        sunday_top = dominant(tally.mix(user, 'sun'))
        if weekday_top is None or sunday_top is None or weekday_top == sunday_top:
            continue
        holiday = tally.mix(user, 'holiday')
        switchers += 1
        holidays += sum(holiday.values())
        holiday_sunday += holiday.get(sunday_top, 0)
        holiday_weekday += holiday.get(weekday_top, 0)
        weekdays += sum(weekday.values())
        weekday_sunday += weekday.get(sunday_top, 0)
    return {'switchers': switchers, 'switcher_holiday_days': holidays,
            'holiday_to_sunday_pattern': ratio(holiday_sunday, holidays),
            'holiday_to_weekday_pattern': ratio(holiday_weekday, holidays),
            'weekday_to_sunday_pattern': ratio(weekday_sunday, weekdays)}


def summarise(tally, k):
    shifts = sorted(user_shifts(tally).values())
    weekend_days = tally.by_class['sat'] + tally.by_class['sun']
    return {'clusters': cluster_stats(tally, k),
            'user_explains': mutual_information(tally.by_user_cluster) / entropy(tally.by_cluster.values()),
            'weekly_mi_bits': weekly_information(tally),
            'weekend_entropy_bits': entropy([tally.by_class['weekday'], weekend_days]),
            'shift_median': quantile(shifts, 0.5), 'shift_p90': quantile(shifts, 0.9),
            'users_shift_gt_0_2': sum(s > 0.2 for s in shifts), 'users': len(shifts),
            **holiday_check(tally)}
