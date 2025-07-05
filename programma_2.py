import os
import sys
import nltk
import math
from nltk import ne_chunk
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

'''
Decommentare questo codice in caso non siano stati già scaricati
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker_tab')
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


def distinzione_wordnet_pos(tag):  # Resituisce la casistica del tag analizzato: Aggettivo, Verbo, nomi, avverbi
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


''' 
    Prende in ingresso una lista di tokens, determina il PoS e lemmatizza tramite WordNetLemmatizer e la lemmatizzazione 
'''
def lemmatizzazione(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    # lemmi = [lemmatizer.lemmatize(token, distinzione_wordnet_pos(tag)) for token, tag in pos_tags]
    lemmi = []

    for token, tag in pos_tags:
        pos = distinzione_wordnet_pos(tag)
        if pos is not None:
            lemma = lemmatizer.lemmatize(token, pos)
            lemmi.append(lemma)
        else:
           continue # In questo modo, evito di inserire la punteggiatura
    return pos_tags, lemmi


''' Ricevendo i pos_tags, il tipo che si desidera e la cifra (50 per tutti e tre i casi nello specifico)
    vado ad estrarmi le top parole per Nome, aggettivi ed avverbi
'''
def estrai_top_parole(pos_tags, tipi, top_n=50):
    parole_filtrate = [token.lower() for token, tag in pos_tags if tag.startswith(tipi)]
    frequenze = Counter(parole_filtrate) # utilizzo counter che va esattamente a contare le ricorrenze
    return frequenze.most_common(top_n)


''' Richiamo la funzione per l'estrazione delle parole passando i casi specifici e successivamente
    vado a formare le frasi di output
'''
def genera_output_frequenze(pos_tags, top_n=50):
    top_sostantivi = estrai_top_parole(pos_tags, 'N', top_n)
    top_aggettivi = estrai_top_parole(pos_tags, 'J', top_n)
    top_avverbi = estrai_top_parole(pos_tags, 'R', top_n)

    '''
        Restituisco i risultati
        Le join funzionano nel seguente modo: formo delle liste di parole e frequenze, invece di
        utilizzare un ciclo for, con una singola linea di codice, vado ad unirle tutte tra loro
        e le divido con un "a capo"
    '''
    output = "Top 50 Sostantivi:\n"
    output += '\n'.join([f"{parola}: {freq}" for parola, freq in top_sostantivi]) + '\n\n'

    output += "Top 50 Aggettivi:\n"
    output += '\n'.join([f"{parola}: {freq}" for parola, freq in top_aggettivi]) + '\n\n'

    output += "Top 50 Avverbi:\n"
    output += '\n'.join([f"{parola}: {freq}" for parola, freq in top_avverbi]) + '\n'

    return output


''' In maniera analoga all'estrazione delle top_parole, vado ad estrarre i top n-grammi.
    Ho reso parametrica i parametri passabili ed impostato ad il default in base a quanto
    richiesto dall'esercizio
'''
def estrai_top_ngrammi(tokens, n_range=(1, 2, 3), top_n=20):
    risultati = {}
    for n in n_range:
        ngrammi = ngrams(tokens, n)
        frequenze = Counter([' '.join(gramma) for gramma in ngrammi]) # utilizzo counter che va esattamente a contare le ricorrenze
        risultati[n] = frequenze.most_common(top_n) # Una funzione derivata dall'utilizzo della Counter, mi da le ricorrenze più comuni
    return risultati


''' Genera una stringa formattata con i top 20 n-grammi più frequenti.
    Prende in input una lista di token e restituisce una rappresentazione testuale leggibile
    dei risultati suddivisi per tipo di n-gramma. 
'''
def genera_output_ngrammi(tokens):
    ngrammi = estrai_top_ngrammi(tokens, n_range=[1, 2, 3], top_n=20)
    output = "\n\nTop 20 N-grammi:\n"

    for n in [1, 2, 3]:
        output += f"\n--- {n}-grammi ---\n"
        for frase, freq in ngrammi[n]:
            output += f"{frase}: {freq}\n"

    return output


''' Dati i pos_tags, vado ad estrare i top n-grammi  '''
def estrai_top_ngrammi_pos(pos_tags, n_range=(1, 2, 3, 4, 5), top_n=20):
    risultati = {}
    # Estrae solo la sequenza di PoS
    pos_sequence = [pos for word, pos in pos_tags]

    for n in n_range:
        ngrammi = ngrams(pos_sequence, n)
        ''' 
        utilizzo counter che va esattamente a contare le ricorrenze. 
        Tramite la join: Ho delle liste di n-grammi, invece di
        utilizzare un ciclo for, con una singola linea di codice, vado ad unire tutte le parole
        tra loro e le divido con una spaziatura
        
        '''
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
    '''
    Tramite questo for (Che scorre fino al penultimo elemento), analizzo una parola con la coppia
    token - postag ed la coppia successiva, in questo modo controllo se ho un verbo ed un nome
    a seguirlo
    '''
    for i in range(len(pos_tags) - 1):
        (token1, postag1), (token2, postag2) = pos_tags[i], pos_tags[i+1]
        if postag1.startswith('V') and postag2.startswith('N'):
            bigrammi_vn.append((token1.lower(), token2.lower()))
    return bigrammi_vn


def calcola_metriche_bigrammi(bigrammi, pos_tags, top_n=10):
    # Utilizzo counter che va esattamente a contare le ricorrenze
    bigram_freq = Counter(bigrammi)
    # Mi ricavo la somme per avere il totale dei bigrammi
    tot_bigrammi = sum(bigram_freq.values())

    # Frequenze unigrammi, calcolati tramite la Counter, funzione integrata di Collection
    unigram_freq = Counter([val.lower() for val, indice in pos_tags])

    risultati = []

    '''
    Il risultato di bigram_freq.items() è un elemento di questo tipo:
    (('love', 'florida'), 1) --> Trami il ciclo flor vado a prendere di conseguenza
    gli elementi delle tuple come verbo e nome e la relativa frequenza (f_vn)
    '''
    for (v, n), f_vn in bigram_freq.items():
        # Ricerco nella variabile le frequenze del verbo e del nome
        f_v = unigram_freq[v]
        f_n = unigram_freq[n]

        # Prob condizionata, equivalente a zero se la condizione f_v > 0 non è rispettata
        p_cond = f_vn / f_v if f_v > 0 else 0

        # Prob congiunta, equivalente a zero se la condizione tot_bigrammi > 0 non è rispettata
        p_v_n = f_vn / tot_bigrammi if tot_bigrammi > 0 else 0

        # MI, vado a calcolarmi la Mutual Information con la relativa formula, sempre che
        # p_v, p_n e p_v_n sia maggiore di 0
        p_v = f_v / tot_bigrammi
        p_n = f_n / tot_bigrammi
        mi = math.log2(p_v_n / (p_v * p_n)) if p_v > 0 and p_n > 0 and p_v_n > 0 else 0

        # LMI ricavo Local Mutual Information
        lmi = f_vn * mi

        ''' 
        Invece di restituire più variabili, vado a crearmi un oggetto che viene inserito dentro
        all'array dichiarata in precedenza 
        '''
        risultati.append({
            'bigramma': f"{v} {n}",
            'frequenza': f_vn,
            'p_cond': p_cond,
            'p_v_n': p_v_n,
            'mi': mi,
            'lmi': lmi
        })

    '''
    Ordinamenti 
    
    Quello che vado a fare qui sono le seguenti cose:
      - lambda: è una funzione anonima che riceve un elemento (cioè un dizionario della lista risultati) 
        e restituisce il valore associato alla chiave ricevuta.
      
      - sorted: ordina la lista in base a un criterio specifico (il parametro key passato), 
        in ordine decrescente perché è stato passato reverse=True
        
      - [:top_n] vuol dire che dopo averlo ordinato prendo fino all'elemento in 20° posizione
    '''
    ordinati = {
        'frequenza': sorted(risultati, key=lambda x: x['frequenza'], reverse=True)[:top_n],
        'p_cond': sorted(risultati, key=lambda x: x['p_cond'], reverse=True)[:top_n],
        'p_v_n': sorted(risultati, key=lambda x: x['p_v_n'], reverse=True)[:top_n],
        'mi': sorted(risultati, key=lambda x: x['mi'], reverse=True)[:top_n],
        'lmi': sorted(risultati, key=lambda x: x['lmi'], reverse=True)[:top_n],
    }

    # Intersezione
    top_mi_set = set([elem['bigramma'] for elem in ordinati['mi']])
    top_lmi_set = set([elem['bigramma'] for elem in ordinati['lmi']])
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


'''
    Vado a selezionare solo quelle frasi che hanno una lunghezza di parole che va da 10 a 20 
    (Comprese) e per cui i token occorrono almeno 2 volte nel corpus. 
'''
def filtra_frasi_valide(frasi_tok, tokens):
    token_freq = Counter(tokens)
    frasi_valide = []

    for frase in frasi_tok:
        if 10 <= len(frase) <= 20:
            non_hapax = [t for t in frase if token_freq[t] >= 2]
            if len(non_hapax) >= len(frase) // 2:
                frasi_valide.append(frase)

    return frasi_valide, token_freq


def media_frequenza(frase, token_freq):
    return sum(token_freq[token] for token in frase) / len(frase)


def costruisci_modello_markov(tokens):
    modello = defaultdict(Counter)
    for i in range(len(tokens) - 2):
        contesto = (tokens[i], tokens[i+1])
        successivo = tokens[i+2]
        modello[contesto][successivo] += 1
    return modello


def probabilita_markov(frase, modello):
    prob = 1.0
    for i in range(len(frase) - 2):
        contesto = (frase[i], frase[i+1])
        successivo = frase[i+2]
        contatore = modello.get(contesto)
        if not contatore or contatore[successivo] == 0:
            prob *= 1e-6  # smoothing
        else:
            total = sum(contatore.values())
            prob *= contatore[successivo] / total
    return prob


def analizza_frasi(frasi_valide, token_freq, modello):
    if not frasi_valide:
        return None, None, None

    max_media = max(frasi_valide, key=lambda f: media_frequenza(f, token_freq))
    min_media = min(frasi_valide, key=lambda f: media_frequenza(f, token_freq))
    max_prob = max(frasi_valide, key=lambda f: probabilita_markov(f, modello))

    return max_media, min_media, max_prob


def genera_output_analisi_frasi(frasi_valide, token_freq, modello):
    max_media, min_media, max_prob = analizza_frasi(frasi_valide, token_freq, modello)

    output = "\n\nAnalisi Frasi con vincoli su lunghezza e hapax:\n"

    if max_media:
        output += f"\na. Frase con media di frequenza più alta:\n{' '.join(max_media)}\n"
        output += f"    Media: {media_frequenza(max_media, token_freq):.2f}\n"

    if min_media:
        output += f"\nb. Frase con media di frequenza più bassa:\n{' '.join(min_media)}\n"
        output += f"    Media: {media_frequenza(min_media, token_freq):.2f}\n"

    if max_prob:
        output += f"\nc. Frase con probabilità massima (Markov ordine 2):\n{' '.join(max_prob)}\n"
        output += f"    Probabilità stimata: {probabilita_markov(max_prob, modello):.6e}\n"

    return output


def calcola_percentuale_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    num_stopwords = sum(1 for token in tokens if token.lower() in stop_words)
    percentuale = (num_stopwords / len(tokens)) * 100 if tokens else 0
    return percentuale, num_stopwords, len(tokens)


def genera_output_stopwords(tokens):
    percentuale, n_stopwords, n_tokens = calcola_percentuale_stopwords(tokens)
    output = "\n\nAnalisi Stopwords:\n"
    output += f"Numero totale token: {n_tokens}\n"
    output += f"Numero stopwords: {n_stopwords}\n"
    output += f"Percentuale stopwords: {percentuale:.2f}%\n"
    return output


def calcola_pronomi_personali(pos_tags, frasi_tok):
    pronomi = [val for val, tag in pos_tags if tag in ('PRP', 'PRP$')]
    num_pron = len(pronomi)
    total_tokens = len(pos_tags)
    percentuale = (num_pron / total_tokens) * 100 if total_tokens else 0

    # Pronomi per frase
    pronomi_per_frase = []
    for frase in frasi_tok:
        count = sum(1 for token, tag in nltk.pos_tag(frase) if tag in ('PRP', 'PRP$'))
        pronomi_per_frase.append(count)
    media_per_frase = sum(pronomi_per_frase) / len(pronomi_per_frase) if frasi_tok else 0

    return num_pron, total_tokens, percentuale, media_per_frase


def genera_output_pronomi(pos_tags, frasi_tok):
    num_pron, total_tokens, perc, media = calcola_pronomi_personali(pos_tags, frasi_tok)
    output = "\n\nAnalisi Pronomi Personali:\n"
    output += f"Totale token: {total_tokens}\n"
    output += f"Numero pronomi personali: {num_pron}\n"
    output += f"Percentuale pronomi personali: {perc:.2f}%\n"
    output += f"Media pronomi personali per frase: {media:.2f}\n"
    return output


def estrai_named_entities(pos_tags):
    # Eseguo il Named Entity Recognition, tramite il binary a false mi va a fare una distinzione con etichette specifiche
    chunk = ne_chunk(pos_tags, binary=False)
    ne_freq = {}

    for elemento in chunk:
        # Dentro l'elemento label trovo l'entità individuata dal chunk
        if hasattr(elemento, 'label'):
            ''' 
            Tramite leaver, essendo elemento un albero, ricevo una lista di tuple che mi indica
            la parola ed il pos tag. Tramite la join, essendo gli elementi che potrebbero essere molteplici
            per una singola entry dell'array, li vado ad unire insereno una spaziatura " " vuota tra di loro
            '''
            entity = " ".join([token for token, el in elemento.leaves()])

            label = elemento.label()
            if label not in ne_freq:
                ne_freq[label] = []
            ne_freq[label].append(entity)

    return ne_freq


def calcola_frequenze_ne(ne_dict):
    frequenze = {}

    for label, entita in ne_dict.items():
        conta = Counter(entita)
        totale = sum(conta.values())
        top_15 = conta.most_common(15)
        frequenze[label] = [(entita, freq, (freq / totale) * 100) for entita, freq in top_15]

    return frequenze


def genera_output_named_entities(pos_tags):
    ne_dict = estrai_named_entities(pos_tags)
    frequenze = calcola_frequenze_ne(ne_dict)

    output = "\n\nAnalisi Entità Nominate (Named Entities):\n"
    for label, entita_list in frequenze.items():
        output += f"\n-- Categoria: {label} --\n"
        for entita, freq, perc in entita_list:
            output += f"{entita}: {freq} ({perc:.2f}%)\n"
    return output


def main(file):
    fileLetto = lettura_file(file)

    tokens, frasi_tok = tokenizzazione(fileLetto)
    pos_tags, lemmi = lemmatizzazione(tokens)

    # Prima richiesta, top 50 sostantivi, aggettivi, avverbi
    output = genera_output_frequenze(pos_tags, 50)

    # Seconda richiesta, top 20 n-grammi più frequenti
    output += genera_output_ngrammi(tokens)

    # Terza richiesta, top 20 n-grammi PoS più frequenti
    output += genera_output_ngrammi_pos(pos_tags)

    # Quarta richiesta, top 10 bigrammi composti da verbo e sostantivo
    output += genera_output_bigrammi_vn(pos_tags)

    # Quinta richiesta, Media di distribuzione
    frasi_valide, token_freq = filtra_frasi_valide(frasi_tok, tokens)
    modello_markov = costruisci_modello_markov(tokens)
    output += genera_output_analisi_frasi(frasi_valide, token_freq, modello_markov)

    # Sesta richiesta, percentuale Stopwords nel corpus
    output += genera_output_stopwords(tokens)

    # Settiman richiesta, frequenza di uso dei pronomi personali
    output += genera_output_pronomi(pos_tags, frasi_tok)

    # Ottava richiesta, estrae le entità nominate del testo
    output += genera_output_named_entities(pos_tags)

    # Scrittura finale del file di risultato
    scrittura_output(f'programma2_{os.path.basename(file)}', output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usare: python3 programma_2.py <file1.txt> ")
        sys.exit(1)

    main(sys.argv[1])