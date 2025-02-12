VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV)/bin/activate: 
	python3 -m venv $(VENV)

venv: $(VENV)/bin/activate

all: venv
	$(PIP) install .

representation:
	$(PYTHON) directory_tree.py

clean:
	rm -rf build
	rm -rf src/opencllib.egg-info
	rm -rf sample_opencl_lib.egg-info
	rm -rf dist
	rm -rf $(VENV)
	$(PIP) uninstall -y sample_opencl_lib

test: venv
	$(PYTHON) -m unittest test.py