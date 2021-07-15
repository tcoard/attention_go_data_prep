import pytest
from .. import deep_to_attention
from ..deep_to_attention import AA_TO_ONE_HOT
import numpy as np


@pytest.mark.parametrize(
    "seq,max_len,expected",
    [
        (
            ["ATP"],
            4,
            np.array(np.array([[AA_TO_ONE_HOT["A"], AA_TO_ONE_HOT["T"], AA_TO_ONE_HOT["P"], AA_TO_ONE_HOT["empty"]]])),
        ),
        (
            ["BB", "A"],
            2,
            np.array(
                [
                    np.array([AA_TO_ONE_HOT["X"], AA_TO_ONE_HOT["X"]]),
                    np.array([AA_TO_ONE_HOT["A"], AA_TO_ONE_HOT["empty"]]),
                ]
            ),
        ),
    ],
)
def test_get_one_hot_encodings(seq: str, max_len: int, expected: np.ndarray):
    deep_to_attention.MAX_LEN = max_len
    print("actual")
    print(deep_to_attention.get_one_hot_encodings(seq))
    print("expected")
    print(expected)
    assert np.array_equal(deep_to_attention.get_one_hot_encodings(seq), expected)


@pytest.mark.parametrize(
    "seq_annos,go_term_namespace,expected",
    [
        (
            [["GO:1"], ["GO:3", "GO:2"]],
            {"GO:1": "biological_process", "GO:2": "biological_process", "GO:3": "molecular_function"},
            ([["GO:1"], ["GO:3", "GO:2"]], {"biological_process": ["GO:1", "GO:2"], "molecular_function": ["GO:3"]}),
        ),
    ],
)
def test_count_go_terms(seq_annos: list[list[str]], go_term_namespace: dict[str, str], expected: dict[str, list[str]]):
    deep_to_attention.MIN_GO = 1
    deep_to_attention.count_go_terms(seq_annos, go_term_namespace) == expected


@pytest.mark.parametrize(
    "seq_annos,go_term_namespace,expected",
    [
        (
            [["GO:1"], ["GO:3", "GO:2"]],
            {"GO:1": "biological_process", "GO:2": "biological_process", "GO:3": "molecular_function"},
            [np.array([True, True]), np.array([False, True]), np.array([False, False])],
        ),
    ],
)
def test_get_valid_list_index(seq_annos: list[list[str]], go_term_namespace: dict[str, list[str]], expected: list[np.ndarray]):
    actual = deep_to_attention.get_valid_list_index(seq_annos, go_term_namespace)
    for act, exp in zip(actual, expected):
        assert np.array_equal(act, exp)
