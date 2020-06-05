PY:=python
PIP:=pip

SRC_PATH:=./crosslangt


.PHONY=develop deps test lint

develop: deps
	$(PY) setup.py develop

deps:
	$(PIP) install -r requirements-dev.txt --quiet
	$(PIP) install -r requirements.txt --quiet

test:
	$(PY) -m unittest -v

lint: deps
	 pycodestyle $(SRC_PATH)
	 
