# -*- coding: utf-8 -*-

import numpy as np
import pytest

from trx.streamlines_ops import (perform_streamlines_operation,
                                 intersection, union, difference)

streamlines_ori = [np.ones(90).reshape((30, 3)),
                   np.arange(90).reshape((30, 3)) + 0.3333]


@pytest.mark.parametrize(
    "precision, noise, expected",
    [
        (0, 0.0001, [4]),
        (1, 0.0001, [4]),
        (2, 0.0001, [4]),
        (4, 0.0001, []),
        (0, 0.01, [4]),
        (1, 0.01, [4]),
        (2, 0.01, []),
        (0, 1, []),
    ],
)
def test_intersection(precision, noise, expected):
    streamlines_new = []
    for i in range(5):
        if i < 4:
            streamlines_new.append(streamlines_ori[1] +
                                   np.random.random((30, 3)))
        else:
            streamlines_new.append(streamlines_ori[1] +
                                   noise * np.random.random((30, 3)))
    # print(streamlines_new)
    _, indices_uniq = perform_streamlines_operation(intersection,
                                                    [streamlines_new,
                                                     streamlines_ori],
                                                    precision=precision)
    indices_uniq = indices_uniq.tolist()
    assert indices_uniq == expected


@pytest.mark.parametrize(
    "precision, noise, expected",
    [
        (0, 0.0001, 6),
        (1, 0.0001, 6),
        (2, 0.0001, 6),
        (4, 0.0001, 7),
        (0, 0.01, 6),
        (1, 0.01, 6),
        (2, 0.01, 7),
        (0, 1, 7),
    ],
)
def test_union(precision, noise, expected):
    streamlines_new = []
    for i in range(5):
        if i < 4:
            streamlines_new.append(streamlines_ori[1] +
                                   np.random.random((30, 3)))
        else:
            streamlines_new.append(streamlines_ori[1] +
                                   noise * np.random.random((30, 3)))

    unique_streamlines, _ = perform_streamlines_operation(union,
                                                          [streamlines_new,
                                                           streamlines_ori],
                                                          precision=precision)
    assert len(unique_streamlines) == expected


@pytest.mark.parametrize(
    "precision, noise, expected",
    [
        (0, 0.0001, 4),
        (1, 0.0001, 4),
        (2, 0.0001, 4),
        (4, 0.0001, 5),
        (0, 0.01, 4),
        (1, 0.01, 4),
        (2, 0.01, 5),
        (0, 1, 5),
    ],
)
def test_difference(precision, noise, expected):
    streamlines_new = []
    for i in range(5):
        if i < 4:
            streamlines_new.append(streamlines_ori[1] +
                                   np.random.random((30, 3)))
        else:
            streamlines_new.append(streamlines_ori[1] +
                                   noise * np.random.random((30, 3)))

    unique_streamlines, _ = perform_streamlines_operation(difference,
                                                          [streamlines_new,
                                                           streamlines_ori],
                                                          precision=precision)
    assert len(unique_streamlines) == expected
