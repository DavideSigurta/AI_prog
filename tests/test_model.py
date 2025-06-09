"""
Test unitari per il modello di classificazione delle emozioni.
"""

import os
import shutil
import pytest
from datasets import load_dataset
from src.model_training import create_models_dir, load_model, preprocess_data

def test_create_models_dir():
    """Test che verifica la creazione della directory models/."""
    # Rimuovi la directory se già esiste
    if os.path.exists("models"):
        # Salva temporaneamente eventuali file nella directory
        if os.path.isdir("models") and os.listdir("models"):
            os.rename("models", "models_temp")
        else:
            shutil.rmtree("models")
    
    # Esegui la funzione da testare
    create_models_dir()
    
    # Verifica che la directory esista
    assert os.path.exists("models")
    assert os.path.isdir("models")
    
    # Ripristina i file originali se necessario
    if os.path.exists("models_temp"):
        if os.path.exists("models"):
            shutil.rmtree("models")
        os.rename("models_temp", "models")

def test_load_model():
    """Test che verifica il caricamento del modello DistilBERT."""
    model, tokenizer = load_model(num_labels=6)
    
    # Verifica che il modello e il tokenizer siano stati caricati
    assert model is not None
    assert tokenizer is not None
    
    # Verifica che il modello abbia il numero corretto di etichette
    assert model.config.num_labels == 6

def test_preprocess_data():
    """Test che verifica il preprocessing dei dati."""
    # Carica solo 10 esempi per un test rapido
    dataset = load_dataset("emotion", split="train[:10]")
    dataset = {"train": dataset}
    
    # Carica il tokenizer
    _, tokenizer = load_model()
    
    # Esegui il preprocessing
    tokenized_data = preprocess_data(dataset, tokenizer)
    
    # Verifica che il preprocessing abbia creato le features corrette
    assert "input_ids" in tokenized_data["train"].features
    assert "attention_mask" in tokenized_data["train"].features
    assert "labels" in tokenized_data["train"].features