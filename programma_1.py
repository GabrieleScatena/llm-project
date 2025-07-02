import sys
import nltk
import string
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def open_read(f): # svolge le operazioni preliminari di apertura e lettura
    with open(f, mode='r', encoding='utf-8') as file_input:
        return file_input.read()


def tokenizzazione(raw): # svolge sentence splitting e tokenizzazione restituisce tokens in totale e frasi tokenizzate
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = sent_tokenizer.tokenize(raw)
    frasi_tok = [nltk.word_tokenize(frase) for frase in frasi]
    tokens = [token for frase in frasi_tok for token in frase]
    return tokens, frasi_tok


def stampa_numero_frasi_token(frasi_tok, tokens, file): # Stampo il numero di frase ed il numero di Token (Con punteggiatura)
    print(f'\nFILE: {file}')
    print(f'Numero di frasi: {len(frasi_tok)}')
    print(f'Numero di token (inclusa punteggiatura): {len(tokens)}')


def stampa_lunghezze_medie(frasi_tok, tokens, file): # Stampa la lunghezza media rimuovendo la punteggiatura
    tokens_no_punt = rimuovi_punteggiatura(tokens)
    media_token_len = sum(len(t) for t in tokens_no_punt) / len(tokens_no_punt)
    media_frase_len = sum(len(frase) for frase in frasi_tok) / len(frasi_tok)
    print(f'Lunghezza media dei token (escl. punteggiatura): {media_token_len:.2f} caratteri')
    print(f'Lunghezza media delle frasi: {media_frase_len:.2f} token')


def rimuovi_punteggiatura(tokens):
    return [t for t in tokens if t not in string.punctuation]


def pos_distribution(tokens, file):  # Restituisce la distribuzione PoS dei primi 1000 token (senza punteggiatura)
    # Considero unicamente i primi 1000 caratteri
    tokens_no_punt = rimuovi_punteggiatura(tokens[:1000])
    pos_tags = nltk.pos_tag(tokens_no_punt) # Applica il PoS tagging: restituisce una lista di tuple (parola, PoS)

    # Dizionario per contare la frequenza di ciascun tag
    tag_counts = {}
    for word, tag in pos_tags:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1

    # Ordina i tag in base alla frequenza decrescente
    sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)

    # Stampa la distribuzione dei PoS
    print(f'\nDistribuzione PoS (primi 1000 token) - {file}:')
    for tag, count in sorted_tags:
        print(f'{tag}: {count}')


def ttr_incrementale(tokens, file): # Calcola e restituisce il TTR ogni 200 tokens
    print(f'\nTTR incrementale (ogni 200 token) - {file}:')
    for i in range(200, len(tokens)+1, 200):
        segment = tokens[:i]
        types = set(segment)
        ttr = len(types) / len(segment)
        print(f'Primi {i} token: TTR = {ttr:.4f}')


def get_wordnet_pos(tag): # Resituisce la casistica del tag analizzato: Aggettivo, Verbo, nomi, avverbi
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

''' Prende in ingresso una lista di tokens, determina il suo PoS, lemmatizza tramite il WordNetLemmatizer ed infine restituisce una lista di lemmi '''
def lemmatizzazione(tokens):
    # Inizializza il lemmatizzatore di WordNet (serve per ridurre le parole alla loro forma base o lemma)
    lemmatizer = WordNetLemmatizer()
    # Applica il PoS tagging ai token, assegnando a ciascuna parola la sua categoria grammaticale (nome, verbo, ecc.)
    pos_tags = nltk.pos_tag(tokens)
    # Per ogni coppia (token, tag), ottiene il lemma usando il lemmatizzatore e la funzione get_wordnet_pos per mappare il tag in un formato compatibile con WordNet
    lemmi = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    return lemmi

''' Rimuove la punteggiatura, esegue la lemmatizzazione dei tokens senza punteggiatura, ottengo gli elementi unici ed 
    il numero di lemmi per frase con la quale calcolo la media di lemmi per frase'''
def stampa_vocabolario_lemmi(frasi_tok, tokens, file):
    tokens_no_punct = rimuovi_punteggiatura(tokens)
    lemmi = lemmatizzazione(tokens_no_punct)
    vocabolario_lemmi = set(lemmi) # Ottengo gli elementi unici tramite la set
    print(f'\nNumero di lemmi distinti - {file}: {len(vocabolario_lemmi)}')

    num_lemmi_per_frase = [len(set(lemmatizzazione(rimuovi_punteggiatura(frase)))) for frase in frasi_tok]
    media = sum(num_lemmi_per_frase) / len(num_lemmi_per_frase)
    print(f'Numero medio di lemmi per frase: {media:.2f}')


def main(file):
    # Apro il file
    raw = open_read(file)
    # Eseguo la tokenizzazione, il sentence splitting e vado a restituire i token totali e le frasi tokenizzate
    tokens, frasi_tok = tokenizzazione(raw)

    # Stampo il numero di frasi e di token (Con punteggiatura)
    stampa_numero_frasi_token(frasi_tok, tokens, file)

    # Stampo la lunghezza media di frasi e di tokens (rimuovendo la punteggiatura)
    stampa_lunghezze_medie(frasi_tok, tokens, file)

    # Restituisco la distribuzione PoS dei primi 1000 token (senza punteggiatura) con formato tag: conteggio
    pos_distribution(tokens, file)

    # Calcolo e restituisco il TTR ogni 200 tokens
    ttr_incrementale(tokens, file)

    # Stampa il numero di lemmi distinti ed il numero medio di lemmi per frase
    stampa_vocabolario_lemmi(frasi_tok, tokens, file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 programma_1_OLD.py <file1.txt> <file2.txt>")
        sys.exit(1)

    main(sys.argv[1])
    main(sys.argv[2])