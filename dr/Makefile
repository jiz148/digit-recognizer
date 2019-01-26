.PHONY: clean clean-cache predict training

PROJECT ?= dr

COVERAGE_DIR := htmlcov
COVERAGE_REPORT := $(COVERAGE_DIR)/index.html
PYTEST_ARGS := --flakes --pep8 --pylint -s -vv --cov-report term-missing
PYVENV_MAKE := ../tools/make_venv.sh
PYVENV_NAME ?= .venv


clean clean-cache:
	cd .. && PROJECT=dr make clean-cache

check-tools:
	cd .. && make check-tools

prediction pr:
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	PYTHONPATH=.. python prediction.py
else
	USE_PYTHON3=true VENV_NAME=$(PYVENV_NAME) $(PYVENV_MAKE) "$@"
endif
	@echo
	@echo “- DONE: $@”


training tr:
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	PYTHONPATH=.. python training.py
else
	USE_PYTHON3=true VENV_NAME=$(PYVENV_NAME) $(PYVENV_MAKE) "$@"
endif
	@echo
	@echo “- DONE: $@”

test pytest: clean-cache check-tools
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	@echo "--- Starting pytest for unit tests ..."
	@echo
	pytest -c setup.cfg -m "not functest" --flakes --pep8 --pylint -s -vv
	@echo
	@echo "- DONE: $@"
else
	VENV_NAME=$(PYVENV_NAME) $(PYVENV_MAKE) "$@"
endif

test-all-only:
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	# @echo "--- Setup $(PROJECT) develop [$@] ..."
	# python setup.py develop
	# @echo
	pip list
	@echo
	@echo "--- Starting pytest for all tests ..."
	pytest -c setup.cfg --flakes --pep8 --pylint -s -vv
else
	VENV_NAME=$(PYVENV_NAME) $(PYVENV_MAKE) "$@"
endif

test-all: clean-all dev-setup test-all-only
	@echo
	@echo "- DONE: $@"

unittest: check-tools
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	@echo "--- Starting unittest discover ..."
	@echo
	python -m unittest discover -bcfv
	@echo
	@echo "- DONE: $@"
else
	VENV_NAME=$(PYVENV_NAME) $(PYVENV_MAKE) dev-setup "$@"
endif
