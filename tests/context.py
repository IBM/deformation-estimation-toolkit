#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
# Adds the Repo folder explicitly to the path Then imports that stuff that needs to be tested This makes sure the
# tests find the module code without requiring the user to add the module path to $PYTHONPATH or similar
import os
import sys
import re

# add the main project folder to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# this is the absolute path to the `tests` folder, so you can refer to it for, e.g., the sample data
TEST_FOLDER_ABSPATH = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DATA_FOLDER = os.path.abspath(
    os.path.join(TEST_FOLDER_ABSPATH, "..", "sample_data")
)
# import the main package
# import endeform  #  not necessary when using pytest

def extract_class_name(parameter_value):
    """Return the characters after the last '.' in `str(type(parameter_value))`, remove "'<>" and "class " first.
    If there's no '.', then process the whole output, obvi"""
    return re.sub("['<>]|(class )", "", str(type(parameter_value)).split(".")[-1])
