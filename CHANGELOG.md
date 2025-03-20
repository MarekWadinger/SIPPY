## v1.1.3 (2025-03-20)

### Fix

- **tf2ss**: do not remove last row from numerator even if zeros
- **deps**: add missing jupyter dep for docs

## v1.1.2 (2025-03-20)

### Feat

- **tf2ss**: implement minreal in tf2ss
- **all-MIMO-related-files**: say goodbye to slycot and hello to python-only SIPPY!

### Fix

- **poetry.lock**: update old lock
- **tf2ss**: match shape of denominator and sort tests
- **pytest**: prevent pytest from running pages generator and add missing dep
- **mkdocs**: remove legacy code which is not useful
- **examples**: minor fixes in type hints
- **sippy/***: circular import
- **pypi**: add trigger
- **ci**: remove slycot installation
- **deps**: include missing dependency sympy
- seed in CST and some alignment with old examples
- random seed setup
- match name in pyproject with PyPI
- artifacts from subdirectories could not be read

### Refactor

- **Makefile**: remove redundant code
- **examples/***: convert all the examples from .py -> .ipynb
- **sippy**: further refactorization and structure revision
- **sippy**: refactor code to follow standard package structure
- **examples/***: moves examples to docs/
- rename typing -> _typing
- rename project to sippy_unipy for poetry compatibility

## v1.0.2 (2025-03-07)

### Fix

- remove dependency on cibuildwheel
- build failed on poetry install

## "v1.0.1" (2025-03-07)

### Fix

- cz cannot find version and pre-commit warnings about version
