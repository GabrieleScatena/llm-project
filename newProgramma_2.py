import sys
import nltk

def tokenizzazione(file):
    tokenizzatore = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = tokenizzatore.tokenize(file)
    frasi_tok = [nltk.word_tokenize(frase) for frase in frasi]
    tokens = [token for frase in frasi_tok for token in frase]
    return tokens, frasi_tok


def main(file):
    print("")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usare: python3 programma_2.py <file1.txt> ")
        sys.exit(1)

    main(sys.argv[1])