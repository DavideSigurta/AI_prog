"""
Test unitari per le funzioni di Exploratory Data Analysis (EDA).
"""

import os
import shutil
import pytest
from src.eda import create_data_dir, load_data, analyze_data

def test_create_data_dir():
    """Test che verifica la creazione della directory data/."""
    # Rimuovi la directory se già esiste
    if os.path.exists("data"):
        # Salva temporaneamente eventuali file nella directory
        if os.path.isdir("data") and os.listdir("data"):
            os.rename("data", "data_temp")
        else:
            shutil.rmtree("data")
    
    # Esegui la funzione da testare
    create_data_dir()
    
    # Verifica che la directory esista
    assert os.path.exists("data")
    assert os.path.isdir("data")
    
    # Ripristina i file originali se necessario
    if os.path.exists("data_temp"):
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.rename("data_temp", "data")

def test_load_data():
    """Test che verifica il caricamento del dataset emotion."""
    dataset = load_data()
    
    # Verifica le partizioni del dataset
    assert 'train' in dataset
    assert 'validation' in dataset
    assert 'test' in dataset
    
    # Verifica che contenga dati
    assert len(dataset['train']) > 0
    
    # Verifica le colonne attese
    assert 'text' in dataset['train'].features
    assert 'label' in dataset['train'].features

def test_analyze_data():
    """Test che verifica l'analisi dei dati."""
    # Caricamento del dataset
    dataset = load_data()
    
    # Esecuzione dell'analisi
    train_df = analyze_data(dataset)
    
    # Verifiche sul DataFrame risultante
    assert 'text' in train_df.columns
    assert 'label' in train_df.columns
    assert 'emotion' in train_df.columns
    assert 'text_length' in train_df.columns
    
    # Verifiche sulle statistiche
    assert all(train_df['text_length'] >= 0)  # Lunghezza non negativa