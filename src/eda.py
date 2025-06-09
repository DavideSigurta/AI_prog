"""
Exploratory Data Analysis (EDA) per il dataset emotion.
Questo modulo contiene funzioni per caricare, analizzare e visualizzare
il dataset emotion di Hugging Face.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def create_data_dir():
    """
    Crea la directory data/ se non esiste.
    
    Nelle dispense del professore è visibile l'importanza di separare 
    i dati dal codice, ma assicurarsi che la directory esista.
    """
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Directory 'data/' creata.")

def load_data():
    """
    Carica il dataset emotion da Hugging Face.
    
    Returns:
        Dataset: Il dataset emotion contenente train, validation e test.
    """
    print("Caricamento del dataset emotion...")
    dataset = load_dataset("emotion")
    print(f"Dataset caricato. Parti disponibili: {list(dataset.keys())}")
    return dataset

def analyze_data(dataset):
    """
    Esegue un'analisi esplorativa sul dataset.
    
    Args:
        dataset: Il dataset emotion caricato.
        
    Returns:
        pd.DataFrame: DataFrame con i dati di train per ulteriori analisi.
    """
    # Converti in DataFrame per una più facile manipolazione
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Statistiche di base
    print("\n===== STATISTICHE DI BASE =====")
    print(f"Dimensione train set: {len(train_df)} esempi")
    print(f"Dimensione validation set: {len(val_df)} esempi")
    print(f"Dimensione test set: {len(test_df)} esempi")
    
    # Mappa i label numerici alle emozioni
    emotion_labels = dataset['train'].features['label'].names
    print(f"Emozioni disponibili: {emotion_labels}")
    
    # Aggiungi una colonna con l'etichetta testuale
    train_df['emotion'] = train_df['label'].map(lambda i: emotion_labels[i])
    
    # Statistiche sulla lunghezza dei testi
    train_df['text_length'] = train_df['text'].apply(len)
    print("\n===== LUNGHEZZA DEI TESTI =====")
    print(f"Lunghezza media dei testi: {train_df['text_length'].mean():.2f} caratteri")
    print(f"Lunghezza minima: {train_df['text_length'].min()} caratteri")
    print(f"Lunghezza massima: {train_df['text_length'].max()} caratteri")
    
    return train_df

def visualize_data(train_df):
    """
    Crea e salva visualizzazioni del dataset.
    
    Args:
        train_df: DataFrame contenente i dati di training.
    """
    # Impostazioni per le visualizzazioni
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Distribuzione delle emozioni
    plt.figure(figsize=(10, 6))
    emotion_counts = train_df['emotion'].value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribuzione delle emozioni nel dataset')
    plt.xlabel('Emozione')
    plt.ylabel('Conteggio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/emotion_distribution.png')
    print("Salvato grafico: data/emotion_distribution.png")
    
    # 2. Lunghezza del testo per emozione
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='emotion', y='text_length', data=train_df)
    plt.title('Lunghezza del testo per emozione')
    plt.xlabel('Emozione')
    plt.ylabel('Lunghezza del testo (caratteri)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/text_length_by_emotion.png')
    print("Salvato grafico: data/text_length_by_emotion.png")
    
    # 3. Istogramma delle lunghezze dei testi
    plt.figure(figsize=(12, 6))
    sns.histplot(train_df['text_length'], bins=50, kde=True)
    plt.title('Distribuzione della lunghezza dei testi')
    plt.xlabel('Lunghezza del testo (caratteri)')
    plt.ylabel('Frequenza')
    plt.tight_layout()
    plt.savefig('data/text_length_histogram.png')
    print("Salvato grafico: data/text_length_histogram.png")
    
    # 4. Esempio di testi per ogni emozione
    plt.figure(figsize=(12, 8))
    examples = {}
    for emotion in train_df['emotion'].unique():
        examples[emotion] = train_df[train_df['emotion'] == emotion]['text'].iloc[0]
    
    # Crea una tabella con esempi
    table_data = [
        [emotion, text[:80] + "..." if len(text) > 80 else text] 
        for emotion, text in examples.items()
    ]
    
    plt.table(cellText=table_data, colLabels=["Emozione", "Esempio di testo"], 
              cellLoc='left', loc='center', colWidths=[0.2, 0.7])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('data/example_texts.png')
    print("Salvato grafico: data/example_texts.png")

def run_eda():
    """
    Esegue l'intero processo di EDA.
    """
    create_data_dir()
    dataset = load_data()
    train_df = analyze_data(dataset)
    visualize_data(train_df)
    print("\nAnalisi esplorativa completata. Tutti i grafici sono stati salvati nella cartella 'data/'.")
    return dataset

if __name__ == "__main__":
    run_eda()