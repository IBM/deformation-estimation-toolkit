# Testing
## Framework
We're using `pytest`, but maybe other frameworks would work too.  
Tests can be run from the Repo root folder with 

```bash
pytest
```

or selected tests with

```bash
pytest tests/template_test.py
```

Add `-rA` if you have `print` statements inside the tests and want to see the output at the end of the test results:

```bash
pytest -rA tests/template_test.py
```

## Folder structure and imports
All tests should be in the folder `tests`. Due to the way `pytest` works, the presence of `__init__.py` in the `tests` folder means that the repository's root folder is added to the front of `sys.path`, and hence you can use

```python
import endeform
```

and similar statements in your test scripts; see [pytest docs](https://docs.pytest.org/en/stable/pythonpath.html#import-modes) for details.

Put all additional acrobatics concerning absolute paths etc in the `context.py` module. Currently, it provides constants `TEST_FOLDER_ABSPATH` and `SAMPLE_DATA_FOLDER` which can be used inside the test scripts like

```python
from .context import TEST_FOLDER_ABSPATH
```

Note that the tests are not intended to be run as scripts, so things like

```bash
python tests/template_test.py
```
will likely fail.

## Plotting inside of tests
Is probably not a great idea in general, but sometimes necessary. Instead of the slow (and blocking) `plt.show()` call, use the provided fixture `showfig` (defined in [`conftest.py`](conftest.py)),  by specifiying it in the test function definition, and then using `savefig()` in place of `plt.show()`:

```python
def test_plots(savefig):
    plt.plot([0,0,1,1,2,2,3,3,4,4],[0,1,0,1,0,1,0,1,0,1],'c')
    plt.legend(['A cyan sawtooth'])
    savefig()
```

That will store the output figures as png files in `tests/figures`.

## Examples
See [the test template file](test_template.py) for example usage.

