# Makefile for mkdocs docs generation

.PHONY: help init format build serve pdf clean destroy

# Ensure PROJECT_PATH is provided for relevant commands
# PROJECT_PATH ?=
# ifneq ($(filter build serve pdf format,$(MAKECMDGOALS)),)
#     ifeq ($(PROJECT_PATH),)
#         $(error PROJECT_PATH is required. Please specify it as an argument, e.g., make build PROJECT_PATH=/path/to/project)
#     endif
# endif
PROJECT_PATH := "."
# If we want to have PROJECT_PATH as a required argument, we can use the following code
# PROJECT_PATH := $(word 2, $(MAKECMDGOALS))
# ifeq ($(PROJECT_PATH),)
# 	$(error PROJECT_PATH is required. Please specify it as the second argument, e.g. make build /path/to/project)
# endif
SITE_NAME := $(shell basename $(PROJECT_PATH))
CONFIG_PATH := $(CURDIR)/mkdocs.yml
ENV_PATH := $(CURDIR)/.venv
MKDOCS_PATH := $(ENV_PATH)/bin/mkdocs

help: ## Show help
	@echo "Available commands:"
	@echo "  init      			- Set up documentation environment"
	@echo "  format PROJECT_PATH  	- Format code"
	@echo "  build PROJECT_PATH		- Build static docs"
	@echo "  serve PROJECT_PATH		- Serve docs locally"
	@echo "  pdf PROJECT_PATH		- Generate PDF from markdown"
	@echo "  clean     			- Clean generated docs"
	@echo "  destroy   			- Remove virtual environment"
	@echo "  help      			- Show this help"

init:
	@if [ ! -d ".venv-docs" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv --upgrade-deps .venv-docs; \
	else \
		echo "Virtual environment already exists."; \
	fi

	.venv-docs/bin/pip install --require-virtualenv -q -r docs/requirements.txt; \
	env PLAYWRIGHT_BROWSERS_PATH=$(ENV_PATH) .venv-docs/bin/playwright install  --only-shell chromium-headless-shell

format:
	. .venv-docs/bin/activate && cd $(PROJECT_PATH) && pre-commit run -a -c $(CURDIR)/.pre-commit-config.yaml

build:
	cd $(PROJECT_PATH) && \
		env PYTHONPATH=$(PROJECT_PATH) SITE_NAME=$(SITE_NAME) mkdocs build --config-file $(CONFIG_PATH)

serve:
	cd $(PROJECT_PATH) && \
	    env PYTHONPATH=$(PROJECT_PATH) SITE_NAME=$(SITE_NAME) mkdocs serve --config-file $(CONFIG_PATH) -o

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/examples/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown docs/*/*.ipynb

clean:
	rm -r site/

destroy:
	rm -r .venv-docs
