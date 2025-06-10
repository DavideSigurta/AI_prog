# Emotion Classifier

Un classificatore di emozioni basato su DistilBERT per testi in lingua inglese.

## Descrizione

Questo progetto implementa un sistema di classificazione delle emozioni utilizzando un modello DistilBERT fine-tuned. Il sistema è in grado di analizzare un testo e classificarlo in una delle sei emozioni: tristezza, gioia, amore, rabbia, paura e sorpresa.

### Dataset

Il progetto utilizza il dataset [emotion](https://huggingface.co/datasets/emotion) di Hugging Face, che contiene testi brevi classificati in sei categorie emotive.

### Modello

Il classificatore si basa su DistilBERT, una versione distillata di BERT che mantiene il 97% delle sue performance con una dimensione ridotta e una velocità di inferenza maggiore.

## Struttura del Progetto

```
AI_prog/
├── src/                 # Codice sorgente
│   ├── eda.py           # Analisi esplorativa dei dati
│   └── model_training.py # Training e valutazione del modello
├── tests/               # Test unitari
├── data/                # Directory per i dati e le visualizzazioni (generata)
├── models/              # Directory per i modelli salvati (generata)
├── Dockerfile           # Configurazione Docker
├── Makefile             # Comandi per gestire il progetto
├── pyproject.toml       # Configurazione del pacchetto
└── requirements.txt     # Dipendenze del progetto
```

## Requisiti

- Python 3.9 o superiore
- Librerie specificate in `requirements.txt`

## Installazione

Clona il repository e installa le dipendenze:

```bash
git clone https://github.com/DavideSigurta/AI_prog.git
cd AI_prog
make install
```

## Utilizzo

### Analisi Esplorativa dei Dati (EDA)

Esegui l'analisi esplorativa dei dati per generare visualizzazioni sulla distribuzione delle emozioni e le caratteristiche del dataset:

```bash
make run-eda
```

Le visualizzazioni saranno salvate nella directory `data/`.

### Training del Modello

Addestra il modello di classificazione DistilBERT:

```bash
make train
```

Il modello addestrato sarà salvato nella directory `models/`.

### Predizione delle Emozioni

Predici l'emozione associata a un testo:

```bash
make predict
```

Ti verrà chiesto di inserire un testo, e il sistema risponderà con l'emozione predetta.

## Utilizzo con Docker

Il progetto supporta l'esecuzione in container Docker, con opzioni ottimizzate sia per architetture x86_64 standard che per Apple Silicon (M1/M2).

### Architettura x86_64

```bash
# Costruisci l'immagine Docker
make docker-build

# Esegui l'EDA
make docker-eda

# Addestra il modello
make docker-train

# Fai una predizione
make docker-predict
```

### Architettura ARM64 (Apple M1/M2)

```bash
# Costruisci l'immagine Docker ottimizzata per M1
make docker-build-m1

# Esegui l'EDA
make docker-eda-m1

# Addestra il modello
make docker-train-m1

# Fai una predizione
make docker-predict-m1
```

## Testing

Esegui i test unitari:

```bash
make test
```

## Linting

Esegui il linting del codice:

```bash
make lint
```

## Pulizia

Pulisci i file generati:

```bash
make clean
```

Pulisci Docker:

```bash
make docker-clean
```

## Licenza

Questo progetto è distribuito sotto licenza MIT.

## Autore

- Davide Sigurta - [sigurtadavide@gmail.com](mailto:sigurtadavide@gmail.com)
