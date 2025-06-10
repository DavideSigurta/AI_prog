install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	@echo "Installation complete. You can now run the project."

lint:
	pylint --disable=R,C src/*.py tests/*.py
	@echo "Linting complete."

test:
	python -m pytest -vv --cov=src tests/
	@echo "Testing complete."

build:
	python -m build
	@echo "Build complete. Check the dist/ directory."

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__
	@echo "Clean complete."

run-eda:
	python -m src.eda
	@echo "EDA completata. Visualizzazioni salvate nella cartella data/"

train:
	python -m src.model_training
	@echo "Training completato. Modello salvato nella cartella models/"

predict:
	@read -p "Inserisci un testo: " TEXT; \
	python -c "from src.model_training import predict_emotion; print(f'Emozione predetta: {predict_emotion(\"$$TEXT\")}')"

# Standard Docker commands (x86_64)
docker-build:
	docker build -t emotion-classifier .
	@echo "Docker image built successfully (x86_64)."

docker-train:
	docker run --rm -v $(PWD)/models:/app/models:cached emotion-classifier
	@echo "Training completed in Docker. Model saved in models/ directory."

docker-eda:
	docker run --rm -v $(PWD)/data:/app/data:cached emotion-classifier python -m src.eda
	@echo "EDA completed in Docker. Visualizations saved in data/ directory."

docker-predict:
	@read -p "Inserisci un testo: " TEXT; \
	docker run --rm -v $(PWD)/models:/app/models:cached emotion-classifier python -c "from src.model_training import predict_emotion; print(f'Emozione predetta: {predict_emotion(\"$$TEXT\")}')"

# M1-optimized Docker commands (arm64)
docker-build-m1:
	docker build --platform linux/arm64 -t emotion-classifier-m1 .
	@echo "Docker image built successfully for M1 (arm64)."

docker-train-m1:
	@echo "Avvio training su M1 senza mount di volumi..."
	docker run --platform linux/arm64 --cpus=8 --memory=8g --ipc=host --name emotion-train emotion-classifier-m1
	@echo "Training completato, copiando i modelli..."
	docker cp emotion-train:/app/models/. ./models/
	docker rm emotion-train
	@echo "Training completato in Docker (M1). Modello copiato dalla directory del container."

docker-eda-m1:
	docker run --rm --platform linux/arm64 -v $(PWD)/data:/app/data:cached emotion-classifier-m1 python -m src.eda
	@echo "EDA completed in Docker (M1). Visualizations saved in data/ directory."

docker-predict-m1:
	@read -p "Inserisci un testo: " TEXT; \
	docker run --rm --platform linux/arm64 -v $(PWD)/models:/app/models:cached emotion-classifier-m1 python -c "from src.model_training import predict_emotion; print(f'Emozione predetta: {predict_emotion(\"$$TEXT\")}')"

# Cleanup
docker-clean:
	docker system prune -f
	@echo "Docker system cleaned successfully."

# Workflows
docker-workflow: docker-build docker-eda docker-train
	@echo "Docker workflow completed: build, EDA and training executed."

docker-workflow-m1: docker-build-m1 docker-eda-m1 docker-train-m1
	@echo "Docker M1 workflow completed: build, EDA and training executed."