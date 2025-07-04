import sys
import nltk
import math
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

def lettura_file(f): # svolge le operazioni preliminari di apertura e lettura
    with open(f, mode='r', encoding='utf-8') as file_input:
        return file_input.read()


def scrittura_output(nomefile, contenuto): # Svolge la scrittura del risultato finale del file
    with open(nomefile, mode='w', encoding='utf-8') as f:
        f.write(contenuto)


def tokenizzazione(file):
    tokenizzatore = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = tokenizzatore.tokenize(file)
    frasi_tok = [nltk.word_tokenize(frase) for frase in frasi]
    tokens = [token for frase in frasi_tok for token in frase]
    return tokens, frasi_tok


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


''' Prende in ingresso una lista di tokens, determina il PoS e lemmatizza tramite WordNetLemmatizer e la lemmatizzazione '''
def lemmatizzazione(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    lemmi = [lemmatizer.lemmatize(token, distinzione_wordnet_pos(tag)) for token, tag in pos_tags]
    return pos_tags, lemmi


def estrai_top_parole(pos_tags, tipi, top_n=50):
    parole_filtrate = [token.lower() for token, tag in pos_tags if tag.startswith(tipi)]
    frequenze = Counter(parole_filtrate) # utilizzo counter che va esattamente a contare le ricorrenze
    return frequenze.most_common(top_n)


def genera_output_frequenze(pos_tags, top_n=50): # Estraggo le parole più frequenti
    top_sostantivi = estrai_top_parole(pos_tags, 'N', top_n)
    top_aggettivi = estrai_top_parole(pos_tags, 'J', top_n)
    top_avverbi = estrai_top_parole(pos_tags, 'R', top_n)

    # Restituisco i risultati
    output = "Top 50 Sostantivi:\n"
    output += '\n'.join([f"{parola}: {freq}" for parola, freq in top_sostantivi]) + '\n\n'

    output += "Top 50 Aggettivi:\n"
    output += '\n'.join([f"{parola}: {freq}" for parola, freq in top_aggettivi]) + '\n\n'

    output += "Top 50 Avverbi:\n"
    output += '\n'.join([f"{parola}: {freq}" for parola, freq in top_avverbi]) + '\n'

    return output


def estrai_top_ngrammi(tokens, n_range=(1, 2, 3), top_n=20):
    risultati = {}
    for n in n_range:
        ngrammi = ngrams(tokens, n)
        frequenze = Counter([' '.join(gramma) for gramma in ngrammi])
        risultati[n] = frequenze.most_common(top_n)
    return risultati


def genera_output_ngrammi(tokens):
    ngrammi = estrai_top_ngrammi(tokens, n_range=[1, 2, 3], top_n=20)
    output = "\n\nTop 20 N-grammi:\n"

    for n in [1, 2, 3]:
        output += f"\n--- {n}-grammi ---\n"
        for frase, freq in ngrammi[n]:
            output += f"{frase}: {freq}\n"

    return output


def estrai_top_ngrammi_pos(pos_tags, n_range=(1, 2, 3, 4, 5), top_n=20):
    risultati = {}
    # Estrai solo la sequenza di PoS (es. ['DT', 'NN', 'VBZ'])
    pos_sequence = [tag for _, tag in pos_tags]

    for n in n_range:
        ngrammi = ngrams(pos_sequence, n)
        frequenze = Counter([' '.join(gramma) for gramma in ngrammi])
        risultati[n] = frequenze.most_common(top_n)
    return risultati


def genera_output_ngrammi_pos(pos_tags):
    ngrammi_pos = estrai_top_ngrammi_pos(pos_tags, n_range=[1, 2, 3, 4, 5], top_n=20)
    output = "\n\nTop 20 N-grammi di PoS:\n"

    for n in [1, 2, 3, 4, 5]:
        output += f"\n--- {n}-grammi di PoS ---\n"
        for sequenza, freq in ngrammi_pos[n]:
            output += f"{sequenza}: {freq}\n"

    return output


def estrai_bigrammi_VN(pos_tags):
    bigrammi_vn = []
    for i in range(len(pos_tags) - 1):
        (w1, t1), (w2, t2) = pos_tags[i], pos_tags[i+1]
        if t1.startswith('V') and t2.startswith('N'):
            bigrammi_vn.append((w1.lower(), w2.lower()))
    return bigrammi_vn


def calcola_metriche_bigrammi(bigrammi, pos_tags, top_n=10):
    bigram_freq = Counter(bigrammi)
    tot_bigrammi = sum(bigram_freq.values())

    # Frequenze singole
    unigram_freq = Counter([word.lower() for word, _ in pos_tags])
    pos_dict = {word.lower(): tag for word, tag in pos_tags}

    risultati = []

    for (v, n), f_vn in bigram_freq.items():
        f_v = unigram_freq[v]
        f_n = unigram_freq[n]

        # Prob condizionata
        p_cond = f_vn / f_v if f_v > 0 else 0

        # Prob congiunta
        p_joint = f_vn / tot_bigrammi if tot_bigrammi > 0 else 0

        # MI
        p_v = f_v / tot_bigrammi
        p_n = f_n / tot_bigrammi
        mi = math.log2(p_joint / (p_v * p_n)) if p_v > 0 and p_n > 0 and p_joint > 0 else 0

        # LMI
        lmi = f_vn * mi

        risultati.append({
            'bigramma': f"{v} {n}",
            'freq': f_vn,
            'p_cond': p_cond,
            'p_joint': p_joint,
            'mi': mi,
            'lmi': lmi
        })

    # Ordinamenti
    ordinati = {
        'frequenza': sorted(risultati, key=lambda x: x['freq'], reverse=True)[:top_n],
        'p_cond': sorted(risultati, key=lambda x: x['p_cond'], reverse=True)[:top_n],
        'p_joint': sorted(risultati, key=lambda x: x['p_joint'], reverse=True)[:top_n],
        'mi': sorted(risultati, key=lambda x: x['mi'], reverse=True)[:top_n],
        'lmi': sorted(risultati, key=lambda x: x['lmi'], reverse=True)[:top_n],
    }

    # Intersezione
    top_mi_set = set([r['bigramma'] for r in ordinati['mi']])
    top_lmi_set = set([r['bigramma'] for r in ordinati['lmi']])
    intersezione = top_mi_set & top_lmi_set

    return ordinati, intersezione


def genera_output_bigrammi_vn(pos_tags):
    bigrammi = estrai_bigrammi_VN(pos_tags)
    ordinati, intersezione = calcola_metriche_bigrammi(bigrammi, pos_tags, top_n=10)

    output = "\n\nTop 10 Bigrammi Verbo + Sostantivo:\n"

    for criterio, lista in ordinati.items():
        output += f"\n--- Ordinati per {criterio.upper()} ---\n"
        for item in lista:
            output += f"{item['bigramma']} ({criterio}: {item[criterio]:.4f})\n"

    output += f"\nNumero di bigrammi comuni tra top-10 MI e LMI: {len(intersezione)}\n"
    output += f"Comuni: {', '.join(intersezione)}\n"

    return output


def main(file):
    fileLetto = lettura_file(file)

    tokens, frasi_tok = tokenizzazione(fileLetto)
    pos_tags, lemmi = lemmatizzazione(tokens)

    # Prima richiesta, top 50 sostantivi, aggettivi, avverbi
    output = genera_output_frequenze(pos_tags)

    # Seconda richiesta, top 20 n-grammi più frequenti
    output += genera_output_ngrammi(tokens)

    # Terza richiesta, top 20 n-grammi PoS più frequenti
    output += genera_output_ngrammi_pos(pos_tags)

    # Quarta richiesta, top 10 bigrammi composti da verbo e sostantivo
    output += genera_output_bigrammi_vn(pos_tags)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usare: python3 programma_2.py <file1.txt> ")
        sys.exit(1)

    main(sys.argv[1])