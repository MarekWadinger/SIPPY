# Makefile for mkdocs docs generation

.PHONY: help init format build serve pdf clean destroy

help: ## Show help
	@echo "Available commands:"
	@echo "  format		  		- Format code"
	@echo "  build				- Build static docs"
	@echo "  serve				- Serve docs locally"
	@echo "  pdf				- Generate PDF from markdown"
	@echo "  clean     			- Clean generated docs"
	@echo "  help      			- Show this help"


format:
	pre-commit run -a

build:
	mkdocs build

serve:
	mkdocs serve -o

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/examples/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown docs/*/*.ipynb

clean:
	rm -r site/
