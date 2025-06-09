"""
Training di un modello DistilBERT per la classificazione delle emozioni.
Questo modulo contiene funzioni per caricare, addestrare e valutare un modello
di classificazione delle emozioni basato su DistilBERT.
"""

import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def create_models_dir():
    """
    Crea la directory models/ se non esiste.

    """
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Directory 'models/' creata.")

def load_model(num_labels=6):
    """
    Carica il modello DistilBERT pre-addestrato.
    
    Args:
        num_labels: Numero di classi/emozioni (default: 6 per il dataset emotion).
        
    Returns:
        tuple: (model, tokenizer) - Il modello e il tokenizer caricati.
    """
    print("Caricamento del modello DistilBERT...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    print(f"Modello {model_name} caricato con {num_labels} classi.")
    return model, tokenizer

def preprocess_data(dataset, tokenizer, max_length=128):
    """
    Preprocessa il dataset tokenizzando i testi.
    """
    print("Preprocessamento del dataset...")
    
    # Funzione di tokenizzazione applicata in batch
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Gestione diversa se il dataset è un dizionario o un oggetto Dataset
    if isinstance(dataset, dict):
        tokenized_dataset = {}
        for key, ds in dataset.items():
            tokenized_dataset[key] = ds.map(tokenize_function, batched=True)
            # Rimuovi colonne non necessarie e rinomina per la compatibilità con il trainer
            tokenized_dataset[key] = tokenized_dataset[key].remove_columns(["text"])
            if "label" in tokenized_dataset[key].column_names:
                tokenized_dataset[key] = tokenized_dataset[key].rename_column("label", "labels")
            # Imposta il formato torch
            tokenized_dataset[key].set_format("torch")
    else:
        # Codice per un singolo dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
    
    print("Preprocessamento completato.")
    return tokenized_dataset

def compute_metrics(pred):
    """
    Calcola metriche di valutazione per il modello.
    
    Questa funzione è usata dal Trainer per calcolare metriche durante la valutazione.
    
    Args:
        pred: Oggetto EvalPrediction con predictions e label_ids.
        
    Returns:
        dict: Dizionario con le metriche calcolate.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calcola precision, recall, f1 per ogni classe e la media
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    
    # Calcola l'accuratezza
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_model(model, tokenized_dataset):
    """
    Addestra il modello sul dataset tokenizzato.
    """
    print("Configurazione dell'addestramento...")
    
    # Configurazione semplificata compatibile con versioni precedenti di Transformers
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./models/logs",
        logging_steps=10,
    )
    
    print(f"Addestramento configurato con {training_args.num_train_epochs} epoche.")
    
    # Inizializza il Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    
    # Addestra il modello
    print("Inizio addestramento del modello...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Addestramento completato in {training_time:.2f} secondi.")
    
    return trainer

def evaluate_model(trainer, tokenized_dataset):
    """
    Valuta il modello addestrato sul test set.
    
    Args:
        trainer: Oggetto Trainer con il modello addestrato.
        tokenized_dataset: Dataset tokenizzato.
        
    Returns:
        dict: Risultati della valutazione.
    """
    print("\nValutazione del modello sul test set...")
    
    # Valuta il modello sul test set
    test_results = trainer.evaluate(tokenized_dataset["test"])
    
    # Stampa i risultati
    print("\n===== RISULTATI DELLA VALUTAZIONE =====")
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return test_results

def save_model(trainer, tokenizer):
    """
    Salva il modello addestrato e il tokenizer.
    
    Args:
        trainer: Oggetto Trainer con il modello addestrato.
        tokenizer: Tokenizer da salvare.
    """
    model_path = "./models/emotion_classifier"
    
    print(f"Salvataggio del modello in {model_path}...")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print("Modello salvato con successo.")

def run_training():
    """
    Esegue l'intero processo di training del modello.
    
    1. Preparazione (directory, etc.)
    2. Caricamento dei dati
    3. Preprocessing 
    4. Training
    5. Valutazione
    6. Salvataggio
    
    Returns:
        dict: Risultati della valutazione.
    """
    # Crea la directory per i modelli
    create_models_dir()
    
    # Carica il dataset
    print("Caricamento del dataset...")
    dataset = load_dataset("emotion")
    emotion_labels = dataset['train'].features['label'].names
    num_labels = len(emotion_labels)
    print(f"Dataset caricato. Emozioni disponibili: {emotion_labels}")
    
    # Carica il modello e il tokenizer
    model, tokenizer = load_model(num_labels=num_labels)
    
    # Preprocessa il dataset
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # Addestra il modello
    trainer = train_model(model, tokenized_dataset)
    
    # Valuta il modello
    test_results = evaluate_model(trainer, tokenized_dataset)
    
    # Salva il modello
    save_model(trainer, tokenizer)
    
    print("\nProcesso di training completato. Il modello è pronto per l'uso.")
    
    return test_results

def predict_emotion(text, model_path="./models/emotion_classifier"):
    """
    Predice l'emozione per un testo dato usando il modello addestrato.
    
    Args:
        text: Testo da classificare.
        model_path: Percorso del modello salvato.
        
    Returns:
        str: L'emozione predetta.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Il modello non è stato trovato in {model_path}. "
                              "Eseguire prima l'addestramento con 'make train'.")
    
    # Carica il modello e il tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Tokenizza il testo
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Predici
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    
    # Mappa l'indice all'etichetta
    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    predicted_emotion = emotion_labels[predictions.item()]
    
    return predicted_emotion

if __name__ == "__main__":
    run_training()