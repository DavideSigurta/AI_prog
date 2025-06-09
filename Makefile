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