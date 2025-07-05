import sys
import nltk
import string
import os
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

'''
Decommentare questo codice in caso non siano stati già scaricati
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('vader_lexicon')
'''

def lettura_file(f): # svolge le operazioni preliminari di apertura e lettura
    with open(f, mode='r', encoding='utf-8') as file_input:
        return file_input.read()

def scrittura_output(nomefile, contenuto): # Svolge la scrittura del risultato finale del file
    with open(nomefile, mode='w', encoding='utf-8') as f:
        f.write(contenuto)

def tokenizzazione(file): # svolge sentence splitting e tokenizzazione restituisce tokens in totale e frasi tokenizzate
    tokenizzatore = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = tokenizzatore.tokenize(file)
    frasi_tok = [nltk.word_tokenize(frase) for frase in frasi]
    tokens = [token for frase in frasi_tok for token in frase]
    return tokens, frasi_tok


def stampa_numero_frasi_token(frasi_tok, tokens, file): # Stampo il numero di frase ed il numero di Token (Con punteggiatura)
    output = f'\nFILE: {file}'
    output += f'\n\nNumero di frasi: {len(frasi_tok)}'
    output += f'\nNumero di token (inclusa punteggiatura): {len(tokens)}'

    return output

def stampa_lunghezze_medie(frasi_tok, tokens, file): # Stampa la lunghezza media rimuovendo la punteggiatura
    tokens_no_punt = rimuovi_punteggiatura(tokens)
    media_token_len = sum(len(t) for t in tokens_no_punt) / len(tokens_no_punt)
    media_frase_len = sum(len(frase) for frase in frasi_tok) / len(frasi_tok)

    output = f'\nLunghezza media dei token (escl. punteggiatura): {media_token_len:.2f} caratteri' #Mi fermo al secondo valore dopo la virgola, anche nel campo successivo
    output += f'\nLunghezza media delle frasi: {media_frase_len:.2f} token'

    return output

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
    output = f'\n\nDistribuzione PoS (primi 1000 token) - {file}:'
    for tag, count in sorted_tags:
        output += f'\n{tag}: {count}'

    return output

def ttr_incrementale(tokens, file): # Calcola e restituisce il TTR ogni 200 tokens
    output = f'\n\nTTR incrementale (ogni 200 token) - {file}:'
    for i in range(200, len(tokens)+1, 200):
        segment = tokens[:i]
        types = set(segment)
        ttr = len(types) / len(segment)
        output += f'\nPrimi {i} token: TTR = {ttr:.4f}' # Mi fermo al 4° valore dopo la virgola

    return output

def distinzione_wordnet_pos(tag): # Resituisce la casistica del tag analizzato: Aggettivo, Verbo, nomi, avverbi
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
    lemmi = [lemmatizer.lemmatize(token, distinzione_wordnet_pos(tag)) for token, tag in pos_tags]
    return lemmi


''' Rimuove la punteggiatura, esegue la lemmatizzazione dei tokens senza punteggiatura, ottengo gli elementi unici ed 
    il numero di lemmi per frase con la quale calcolo la media di lemmi per frase'''
def stampa_vocabolario_lemmi(frasi_tok, tokens, file):
    tokens_no_punct = rimuovi_punteggiatura(tokens)
    lemmi = lemmatizzazione(tokens_no_punct)
    vocabolario_lemmi = set(lemmi) # Ottengo gli elementi unici tramite la set

    output = f'\n\nNumero di lemmi distinti - {file}: {len(vocabolario_lemmi)}'

    num_lemmi_per_frase = [len(set(lemmatizzazione(rimuovi_punteggiatura(frase)))) for frase in frasi_tok]
    media = sum(num_lemmi_per_frase) / len(num_lemmi_per_frase)
    output += f'\nNumero medio di lemmi per frase: {media:.2f}' # Mi fermo al secondo valore dopo la virgola

    return output

def analisi_sentiment(frasi_tok, file): # Effettua l'analisi del sentiment tramite VADER
    sia = SentimentIntensityAnalyzer()
    positive_contatore = 0
    negative_contatore = 0
    neutre_contatore = 0
    polarita_documento = 0.0

    for frase in frasi_tok:
        frase_testo = ' '.join(frase)
        risultato = sia.polarity_scores(frase_testo)
        compound = risultato['compound']

        if compound >= 0.05:
            positive_contatore += 1
        elif compound <= -0.05:
            negative_contatore += 1
        else:
            neutre_contatore += 1

        polarita_documento += compound

    output = f"\n\nDistribuzione frasi POS/NEG/NEU - {file}:"
    output += f"\nPositive: {positive_contatore}"
    output += f"\nNegative: {negative_contatore}"
    output += f"\nNeutre: {neutre_contatore}"
    output += f"\nPolarità complessiva del documento: {polarita_documento:.4f}"

    return output

def main(file):
    # Apro il file
    fileLetto = lettura_file(file)
    # Eseguo la tokenizzazione, il sentence splitting e vado a restituire i token totali e le frasi tokenizzate
    tokens, frasi_tok = tokenizzazione(fileLetto)

    # Stampo il numero di frasi e di token (Con punteggiatura)
    output = stampa_numero_frasi_token(frasi_tok, tokens, file)

    # Stampo la lunghezza media di frasi e di tokens (rimuovendo la punteggiatura)
    output += stampa_lunghezze_medie(frasi_tok, tokens, file)

    # Restituisco la distribuzione PoS dei primi 1000 token (senza punteggiatura) con formato tag: conteggio
    output += pos_distribution(tokens, file)

    # Calcolo e restituisco il TTR ogni 200 tokens
    output += ttr_incrementale(tokens, file)

    # Stampa il numero di lemmi distinti ed il numero medio di lemmi per frase
    output += stampa_vocabolario_lemmi(frasi_tok, tokens, file)

    # Effettua l'analisi sentiment rispetto alle frasi passate ed il file
    output += analisi_sentiment(frasi_tok, file)

    scrittura_output(f'programma1_{os.path.basename(file)}', output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usare: python3 programma_1.py <file1.txt> <file2.txt>")
        sys.exit(1)

    main(sys.argv[1])
    main(sys.argv[2])