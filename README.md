# bloch-mcconnell-splitter
A matlab class that implements a matrix representation and solver for the Bloch-McConnell equations of a three-pool system. The class uses the operator-splitting approach.

## Running tests ✅

This repository includes a Python translation and tests. You can run the test suite locally or rely on the included GitHub Actions workflow which runs tests on push and pull requests.

To run tests locally (no extra tools required beyond Python and NumPy):

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package and development dependencies (recommended):

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

3. Run the tests:

```bash
python -m unittest -q
```

The repository also includes a GitHub Actions workflow at `.github/workflows/python-tests.yml` that runs the tests on Python 3.10 and 3.11 for pushes and pull requests.
