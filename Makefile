.PHONY: clean clean-cache

PROJECT ?= dr

COVERAGE_DIR := htmlcov
COVERAGE_REPORT := $(COVERAGE_DIR)/index.html

SYSTOOLS := find rm pip tee xargs zip

PYTEST_ARGS := --flakes --pep8 --pylint -s -vv --cov-report term-missing
PYVENV_MAKE := tools/make_venv.sh
PYVENV_NAME ?= .venv


check-tools:
	@echo
	@echo "--- Checking for presence of required tools: $(SYSTOOLS)"
	$(foreach tool,$(SYSTOOLS),\
	$(if $(shell which $(tool)),$(echo "boo"),\
	$(error "ERROR: Cannot find '$(tool)' in system $$PATH")))
	@echo
	@echo "- DONE: $@"


clean clean-cache:
	@echo
	@echo “--- Removing pyc and log files”
	find . -name ‘.DS_Store’ -type f -delete
	find . -name \*.pyc -type f -delete -o -name \*.log -delete
	rm -Rf .cache
	rm -Rf .vscode
	rm -Rf $(PROJECT)/.cache
	rm -Rf $(PROJECT)/__pycache__
	rm -Rf $(PROJECT)/tests/__pycache__
	rm -Rf tests/__pycache__
	@echo
	@echo “--- Removing coverage files”
	find . -name ‘.coverage’ -type f -delete
	rm -rf .coveragerc
	rm -rf cover
	rm -rf $(PROJECT)/cover
	rm -Rf $(PROJECT)/$(COVERAGE_DIR)
	rm -Rf $(COVERAGE_DIR)
	@echo
	@echo “--- Removing *.egg-info”
	rm -Rf *.egg-info
	rm -Rf $(PROJECT)/*.egg-info
	@echo
	@echo “--- Removing tox virtualenv”
	rm -Rf $(PROJECT)/.tox*
	@echo
	@echo “--- Removing build”
	rm -rf $(PROJECT)_build.tee
	@echo
	@echo “- DONE: $@”

clean-all: clean-cache
ifneq ("$(VIRTUAL_ENV)", "")
	@echo "--- Cleaning up pip list in $(VIRTUAL_ENV) ..."
	pip freeze | grep -v "^-e" | xargs pip uninstall -y || true
else
	@echo "--- Removing virtual env"
	rm -Rf $(PYVENV_NAME)
	rm -Rf .venv*
endif
	@echo
	@echo "- DONE: $@"


# setup and dev-setup targets
$(PYVENV_NAME)/bin/activate dev-setup-venv: check-tools requirements-dev.txt
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	@echo "--- Cleaning up pip list in $(VIRTUAL_ENV) ..."
	pip install --upgrade pip || true
	pip freeze | grep -v "^-e" | xargs pip uninstall -y || true
	@echo
	@echo "--- Setting up $(PROJECT) develop ..."
	cd $(PROJECT) && python setup.py develop
	@echo
	@echo "--- Installing required dev packages ..."
	# running setup.py in upper level of `$(PROJECT)` folder to register the package
	pip install -r requirements-dev.txt
	@echo
	pip list
else
	@echo "Checking python venv: $(PYVENV_NAME)"
	@echo "----------------------------------------------------------------------"
	USE_PYTHON3=true VENV_NAME=$(PYVENV_NAME) $(PYVENV_MAKE) "$@"
	@echo
	# touch $(PYVENV_NAME)/bin/activate
endif
	@echo
	@echo "- DONE: $@"

dev-setup: dev-setup-venv
	@echo "----------------------------------------------------------------------"
	@echo "Python environment: $(PYVENV_NAME)"
	@echo "- Activate command: source $(PYVENV_NAME)/bin/activate"
	@echo


############################################################
# sub-projects Makefile redirection
############################################################
digit-prediction dpr:
	cd dr && make prediction
	@echo “- DONE: $@”

digit-training dtr:
	cd dr && make training
	@echo “- DONE: $@”


############################################################
# venv
############################################################
dvenv:
	@echo
ifneq ("$(VIRTUAL_ENV)", "")
	@echo "----------------------------------------------------------------------"
	@echo "Python environment: $(VIRTUAL_ENV)"
	@echo "- Activate command: source $(VIRTUAL_ENV)/bin/activate"
	@echo "- Deactivating cmd: deactivate"
	@echo "----------------------------------------------------------------------"
else
	@echo "Cleaning up python venv: $(PYVENV_NAME)"
	rm -rf $(PYVENV_NAME)
endif
	@echo ""
	@echo "- DONE: $@"
	@echo ""

venv: check-tools
	@echo
ifeq ("$(VIRTUAL_ENV)", "")
	@echo "Preparing for venv: [$(PYVENV_NAME)] ..."
	python3 -m venv $(PYVENV_NAME)
	@echo "----------------------------------------------------------------------"
	@echo "Python environment: $(PYVENV_NAME)"
	@echo "- Activate command: source $(PYVENV_NAME)/bin/activate"
else
	@echo "----------------------------------------------------------------------"
	@echo "- Activated python venv: $(VIRTUAL_ENV)"
endif
	@echo "----------------------------------------------------------------------"
	@echo "- DONE: $@"
	@echo ""
