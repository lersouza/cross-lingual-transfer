PY:=python
PIP:=pip

SRC_PATH:=./crosslangt
TOOLS_PATH:=./scripts

.PHONY=dev deps test lint

dev: deps
	$(PY) setup.py develop

deps:
	$(PIP) install -r requirements-dev.txt --quiet
	$(PIP) install -r requirements.txt --quiet

test:
	$(PY) -m unittest -v

lint: deps
	 pycodestyle $(SRC_PATH)
	 pycodestyle $(TOOLS_PATH)
	 
