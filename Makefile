.PHONY: help setup start clean

# Virtual environment settings
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make setup   - Create virtual environment and install dependencies"
	@echo "  make start   - Start the FastAPI RAG Cache service with Uvicorn auto-reload"
	@echo "  make clean   - Purge virtual environment and cached byte files"

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment in $(VENV)..."
	python3 -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Run 'make start' to run the service."

start:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Running setup first..."; \
		make setup; \
	fi
	@echo "Starting FastAPI RAG caching service on port 8000..."
	$(PYTHON) -m uvicorn main:app --reload --port 8000

clean:
	@echo "Purging virtual environment and cached folders..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean completed."
