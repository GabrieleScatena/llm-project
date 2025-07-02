import sys
import nltk

nltk.download('punkt_tab')

def open_read(f):  # svolge le operazioni preliminari di apertura e lettura
    file_input = open(f, mode='r', encoding='utf-8')
    raw = file_input.read()

    return raw

def media(valori):
    somma = 0
    for valore in valori: somma += valore
    media = somma/len(valori)
    return media

def lunghezze(lista): # calcola la lunghezza di ogni frase / parola
    lunghezze =  []
    for elemento in lista:
        lunghezza = len(elemento)
        lunghezze.append(lunghezza)
    return lunghezze

def stampa_titolo(file1, file2):
    titolo = 'PROGRAMMA 1 - ANALISI LINGUISTICA DEI FILE: ' + str(file1) + ', ' + str(file2)
    print(titolo)
    print('_' * len(titolo) + '\n\n')


def stampa_lunghezza_media(oggetto1, oggetto2, file1, file2, valore1, valore2): # stampa lunghezza media parole / frasi

    print('LUNGHEZZA MEDIA', oggetto1.upper(), 'IN TERMINI DI', oggetto2.upper())
    print()
    print()
    print('Lunghezza media', oggetto1, 'del file', file1, ':', valore1)
    print('Lunghezza media', oggetto1, 'del file', file2, ':', valore2)
    print()

    if valore1>valore2:
        print('Il file', file1, 'contiene', oggetto1, 'mediamente più lunghe')
    elif valore2>valore1:
        print('Il file', file2, 'contiene', oggetto1, 'mediamente più lunghe')
    else:
        print('I file', file1, 'e il file', file2, 'hanno la stessa lunghezza media di', oggetto1)

    print()
    print()
    print()
    print()

def tokenizzazione(raw, sent_tokenizer):  # svolge sentence splitting e tokenizzazione restituisce tokens in totale e frasi tokenizzate

    frasi = sent_tokenizer.tokenize(raw)

    frasi_tok = []  # tokens divisi per frase
    tokens_tot = []

    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        frasi_tok.append(tokens)  # appende list con la frase tokenizzata
        tokens_tot = tokens_tot + tokens  # concatena liste per avere i tokens in totale senza la suddivisione in frase

    return tokens_tot, frasi_tok

def stampa_numero(oggetto, file1, file2, valore1, valore2): # stampa numero di caratteri / frasi

    print('NUMERO DI', oggetto.upper())
    print()
    print()
    print('Il file', file1, 'contiene', valore1, oggetto)
    print('Il file', file2, 'contiene', valore2, oggetto)
    print()

    if valore1 > valore2: print('Il file', file1, 'contiene più', oggetto)
    elif valore2 > valore1: print('Il file', file2, 'contiene più', oggetto)
    else: print ('Il file', file1, 'e il file', file2, 'contengono lo stesso numero di', oggetto)
    print()
    print()
    print()
    print()


def main(file1, file2):

    stampa_titolo(file1, file2)
    raw1 = open_read(file1)
    raw2 = open_read(file2)
    # Utilizzo un tokenizzatore di frasi pre-addestrato
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    tokens1, frasi_tok1 = tokenizzazione(raw1, sent_tokenizer)
    tokens2, frasi_tok2 = tokenizzazione(raw2, sent_tokenizer)

    # numero frasi (1)
    n_frasi1 = len(frasi_tok1)
    n_frasi2 = len(frasi_tok2)
    stampa_numero('frasi', file1, file2, n_frasi1, n_frasi2)

    # numero tokens (1)
    n_tokens1 = len(tokens1)
    n_tokens2 = len(tokens2)
    stampa_numero('token', file1, file2, n_tokens1, n_tokens2)

    #lunghezza parole (2)
    lunghezze_tokens1 = lunghezze(tokens1)
    lunghezze_tokens2 = lunghezze(tokens2)
    media_len_tokens1 = media(lunghezze_tokens1)
    media_len_tokens2 = media(lunghezze_tokens2)
    stampa_lunghezza_media('parole', 'caratteri', file1, file2, media_len_tokens1, media_len_tokens2)

    # lunghezza frasi (2)
    lunghezze_frasi1 = lunghezze(frasi_tok1)
    lunghezze_frasi2 = lunghezze(frasi_tok2)
    media_len_frasi1 = media(lunghezze_frasi1)
    media_len_frasi2 = media(lunghezze_frasi2)
    stampa_lunghezza_media('frasi', 'parole', file1, file2, media_len_frasi1, media_len_frasi2)


if __name__ == "__main__":

    print(f"${sys.argv}")
    if len(sys.argv) != 3:
        print("Usage: python3 programma_1.py <file1.txt> <file2.txt>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
