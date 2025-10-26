.PHONY: help install-local run run-local clean docker-build docker-run test docker-shell

help:
	@echo "RoboSumo PyTorch - Available commands:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make run           - Run demo in Docker (default settings, no video)"
	@echo "  make test          - Run test suite in Docker"
	@echo "  make clean         - Remove __pycache__ and output files"
	@echo ""
	@echo "Docker run options:"
	@echo "  make docker-run ENV=RoboSumo-Bug-vs-Spider-v0 - Run with specific environment"
	@echo "  make docker-run RECORD=--record-video          - Run with video recording"
	@echo "  make docker-run EPISODES=5                     - Run with N episodes"
	@echo ""
	@echo "Interact with the container:"
	@echo "  make docker-shell   - Open a shell inside the Docker container (for debugging/development)"
	@echo ""
	@echo "Local development (not recommended):"
	@echo "  make install-local  - Install dependencies locally"
	@echo "  make run-local     - Run demo locally"

# Default: build and run in Docker
run: docker-build
	docker run --rm \
		-v $(PWD)/out:/app/out \
		robosumo-torch:latest \
		python play.py --max_episodes 2

# Run tests in Docker
test: docker-build
	@echo "=== Running Test Suite ==="
	@echo ""
	@echo "Test 1: Default (Ant vs Ant, MLP)"
	@docker run --rm robosumo-torch:latest python play.py --max_episodes 2
	@echo ""
	@echo "Test 2: Ant vs Bug with LSTM"
	@docker run --rm robosumo-torch:latest python play.py --env RoboSumo-Ant-vs-Bug-v0 --policy-names lstm lstm --max_episodes 2
	@echo ""
	@echo "Test 3: Spider vs Spider mixed policies"
	@docker run --rm robosumo-torch:latest python play.py --env RoboSumo-Spider-vs-Spider-v0 --policy-names mlp lstm --max_episodes 2
	@echo ""
	@echo "Test 4: With video recording"
	@docker run --rm -v $(PWD)/out:/app/out robosumo-torch:latest python play.py --record-video --max_episodes 2
	@echo ""
	@echo "=== All Tests Passed ==="

# Configurable Docker run with parameters
docker-run: docker-build
	docker run --rm \
		-v $(PWD)/out:/app/out \
		robosumo-torch:latest \
		python play.py --record-video --env RoboSumo-Ant-vs-Ant-v0

# Open an interactive shell in the Docker container
run-dev: docker-build
	docker run --rm -it \
		-v $(PWD)/out:/app/out \
		-v $(PWD):/app \
		-w /app \
		robosumo-torch:latest \
		bash

# Build Docker image
build:
	docker build -t robosumo-torch:latest .

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Note: Output videos require root to delete (created by Docker). Use: docker run --rm -v \$$(PWD)/out:/app/out alpine sh -c 'rm -rf /app/out/*'"

clear-out:
	docker run --rm -v $(PWD)/out:/app/out alpine sh -c 'rm -rf /app/out/*'