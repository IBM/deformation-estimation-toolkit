#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
"""
This is a template test to demonstrate how tests in this project are written
"""
import os

import matplotlib.pyplot as plt


def test_plots(savefig):
    """
    If your tests generate plots, we don't want the test to be blocked by a `plt.show()` call.
    Instead, use the `savefig` fixture: wherever you would have written `plt.show()`, write `savefig()` instead.
    This will generate a PNG with the test's name in the `tests/figures` folder. If more than one plot are generated, they are numbered from 0.
    """

    plt.plot([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "c")
    plt.legend(["A cyan sawtooth"])
    savefig()

    plt.clf()
    plt.plot([0, 1, 2, 3, 4], [0, 1, 0, 1, 0], "r")
    plt.legend(["A red triangle wave"])
    savefig()

    expected_filenames = [
        "tests/figures/test_plots0.png",
        "tests/figures/test_plots1.png",
    ]
    # If everything worked fine, then the above paths should exist
    assert all([os.path.exists(os.path.abspath(ef)) for ef in expected_filenames])


def test_import_endeform():
    """
    You can import endeform just like that.
    """
    try:
        import endeform
    except ImportError:
        assert 0
        #
    assert 1
