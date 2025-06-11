## v1.4.3 (2025-06-11)

### Fix

- **validation**: use validate_data during predict()
- **examples**: add TransformedTargetRegressor for MIMO examples
- **mypy**: type validation issues

## v1.4.2 (2025-06-10)

### Fix

- **examples**: multiple revisions and fixtures throughout examples

## v1.4.1 (2025-06-10)

### Fix

- **examples**: multiple revisions and fixtures throughout examples

## v1.4.0 (2025-05-29)

### Feat

- **model_selection**: GridSearchCV wrapper defaulting cv to TimeSeriesSplit
- **tf2ss**: use external package to manage MIMO tf2ss
- **datasets**: enhance input generation and noise handling
- **autoregressive**: Source of autoregressive methods
- **datasets**: update data generators

### Fix

- **examples**: align examples with latest API
- **io**: predict takes keyword argument E as noise input for backward compatibility
- **datasets**: shapes and arguments

### Refactor

- **plot**: make module plot from example utils
- *****: project-wide cleanup
- **datasets**: typing and replacement of control.matlab wrappers
- **utils, typing**: remove legacy code
- **io.base**: rename to BaseInputOutput and add some common logic
- **rlls**: rename rls -> rlls, remove class and keep the algorithm
- **ills**: rename armax -> ills and keep the algorithm
- **lls**: rename arx -> lls and keep the algorithm

## v1.3.0 (2025-05-16)

### Feat

- **model_selection**: support pipelines

### Fix

- remove legacy module model.py and functions from utils

### Refactor

- **all**: minor refactorization and formatting throughout the project
- **identification**: drop support for system_identification function
- **ss**: update prediction API

### Perf

- **io**: add feasibility checks and parameter constraints

## v1.2.0 (2025-05-14)

### Feat

- **ss.base**: create base class of SSModel and increase API consistency
- **parsim**: remove variable order (use GridSearchIC instead) + scaling argument
- **olsim**: remove select_order (use GridSearchIC instead + introduce scaling argument (as the select_order was not scaled but fit was)

### Fix

- **model_selection**: fix wrong arg name
- **identification**: not changing the input data
- **bumpversion**: add token to release action
- **bumpversion**: add write permissions
- **bumpversion**: roll back to fix autoupdating CHANGELOG.md file
- **bumpversion**: try to fix autoupdating CHANGELOG.md file

### Refactor

- **io**: minor alignment of variable names
- **arx, io.base**: standardize predict() and _fit() methods for IOModel subclasses
- **validation**: create wrapper for validate_data which recognizes specific requirements of SIPPY
- **olsim**: remove redundant code
- **ss**: update API to match better with standard
- **olsim**: convert olsim function into classes aligned with sklearn
- **parsim**: convert parsim function into classes aligned with sklearn

### Perf

- **ss**: replace impile with vstack
- **olsim**: use new information criterion + remove unnecessary computation

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

## v0.2.0 (2024-12-16)
