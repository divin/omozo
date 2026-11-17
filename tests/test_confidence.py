from omozo.confidence import keep_confident


def test_absolute_floor_drops_low_similarity():
    ids = [1, 2]
    sims = {1: 0.8, 2: 0.2}
    out = keep_confident(ids, sims, fts_matched=set(), min_sim=0.35, gap_ratio=0.0)
    assert out == [1]


def test_relative_gap_drops_imposters_behind_a_strong_hit():
    ids = [1, 2, 3]
    sims = {1: 0.95, 2: 0.5, 3: 0.5}
    out = keep_confident(ids, sims, fts_matched=set(), min_sim=0.35, gap_ratio=0.75)
    assert out == [1]  # 0.5 < 0.75 * 0.95


def test_relative_gap_disabled_with_zero():
    ids = [1, 2]
    sims = {1: 0.95, 2: 0.5}
    out = keep_confident(ids, sims, fts_matched=set(), min_sim=0.35, gap_ratio=0.0)
    assert out == [1, 2]


def test_fts_matches_are_exempt_from_both_rules():
    ids = [1, 2, 3]
    sims = {1: 0.95, 2: 0.1, 3: 0.5}
    out = keep_confident(ids, sims, fts_matched={2}, min_sim=0.35, gap_ratio=0.75)
    # 2 is FTS-exempt (kept despite sim 0.1); 3 falls under the relative gap.
    assert out == [1, 2]


def test_everything_pruned_returns_empty():
    ids = [1, 2]
    sims = {1: 0.2, 2: 0.1}
    out = keep_confident(ids, sims, fts_matched=set(), min_sim=0.35, gap_ratio=0.75)
    assert out == []


def test_order_is_preserved():
    ids = [5, 1, 9]
    sims = {5: 0.9, 1: 0.85, 9: 0.8}
    out = keep_confident(ids, sims, fts_matched=set(), min_sim=0.35, gap_ratio=0.0)
    assert out == [5, 1, 9]


def test_missing_similarity_treated_as_zero():
    ids = [1, 2]
    sims = {1: 0.9}
    out = keep_confident(ids, sims, fts_matched=set(), min_sim=0.35, gap_ratio=0.0)
    assert out == [1]
