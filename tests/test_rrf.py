from omozo.rrf import fuse


def test_fuse_prefers_items_high_in_multiple_rankings():
    assert fuse([[1, 2, 3], [2, 3, 4]]) == [2, 3, 1, 4]


def test_fuse_empty_rankings():
    assert fuse([]) == []
    assert fuse([[], []]) == []


def test_fuse_single_ranking_preserves_order():
    assert fuse([[5, 1, 9]]) == [5, 1, 9]


def test_fuse_tie_breaks_by_lower_id():
    # Both appear once at the same rank position -> same score.
    assert fuse([[7], [9]]) == [7, 9]


def test_fuse_custom_k_changes_scores():
    # With huge k, both channels weigh almost equally.
    assert fuse([[1], [1]])[:1] == [1]
