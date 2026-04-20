.PHONY: test test-e2e install-dev

VENV := .venv
PY   := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
REPORT := reports/test-report.html

install-dev:
	$(PY) -m pip install -e ".[dev]" --quiet

test:
	@mkdir -p reports
	$(PYTEST) -m "not e2e" \
		--html=$(REPORT) --self-contained-html \
		-q
	@echo ""
	@echo "Report: $(REPORT)"

test-e2e:
	@mkdir -p reports
	$(PYTEST) \
		--html=reports/test-report-e2e.html --self-contained-html \
		-q
	@echo ""
	@echo "Report: reports/test-report-e2e.html"
