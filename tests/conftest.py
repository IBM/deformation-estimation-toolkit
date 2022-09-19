import glob
import os

import cv2
import matplotlib.pyplot as plt
import pytest

from .context import TEST_FOLDER_ABSPATH, SAMPLE_DATA_FOLDER


#
# Plot-related helpers
#
def _replace_fixture_abspath(nodename, repsep="_"):
    # Replaces the path separator with `repsep`, so the nodename can be used as a filename
    #     fixture_match_pattern = "[[][^]].*[]]"  #  pattern that matches [path/to/file] as provided by e.g. the endoscopy_image
    #     def _replace_dirsep(match, repsep='_'):
    #         return match.group(0).replace(os.path.sep, repsep)
    #     return re.sub(fixture_match_pattern, _replace_dirsep, nodename)
    # ^^ This was probably too complicated, a simple replacing of path separators should do it
    return nodename.replace(os.path.sep, repsep)


@pytest.fixture
def savefig(request, store_folder=TEST_FOLDER_ABSPATH):
    """
    Use `savefig()` in place of `plt.show()` to store the figure as a PNG in `store_folder`.
    The filename will be `store_folder`/figures/[test_name][call_counter].png
    `store_folder` is a parameter, default is the folder in which the tests are located.
    call_counter starts at 0 and is increased everytime savefig is called.
    plt.clf() is _not_ called implicitly
    """
    # print(savefig.call_counter)
    def _savefig():
        try:
            _savefig.call_counter += 1
        except AttributeError:
            _savefig.call_counter = 0
        cls_name = "" if request.cls is None else request.cls.__name__ + "__"
        filename = _replace_fixture_abspath(cls_name + request.node.name)
        abs_filename = os.path.join(
            store_folder,
            "figures",
            "{:s}{:d}.png".format(filename, _savefig.call_counter),
        )
        plt.savefig(abs_filename)

    return _savefig


##
#
# Parametrizations
#
# List all the images that are Frame_*
#
sample_frame_paths = glob.glob(os.path.join(SAMPLE_DATA_FOLDER, "Frame_*"))


@pytest.fixture(scope="module", params=sample_frame_paths)
def endoscopy_image(request):
    "A fixture provding all the endoscopy images in sample_data"
    img_path = request.param
    # print('[FIXTURE] Using path {}'.format(img_path))
    yield cv2.imread(img_path)
    # teardown code would go here.
