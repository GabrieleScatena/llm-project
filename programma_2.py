import sys
import nltk
from collections import Counter
from math import log2

# nltk.download('punkt_tab')

def open_read(f): # svolge le operazioni preliminari di apertura e lettura
    with open(f, mode='r', encoding='utf-8') as file_input:
        return file_input.read()


def write_line(out, text=""): # Svolge l'operazione di scrittura su di un file, ogni riga va a capo
    out.write(str(text) + '\n')


def tokenizer_tag(text):
    sentences = nltk.sent_tokenize(text)
    tokens = []
    pos_tags = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(words)
        tokens.extend([w.lower() for w in words])
        pos_tags.extend([(w.lower(), t) for (w, t) in tags])
    return tokens, pos_tags


def ricava_top_pos(pos_tags, allowed_prefixes, top_n=50):
    counter = Counter()
    for word, tag in pos_tags:
        if any(tag.startswith(prefix) for prefix in allowed_prefixes):
            counter[word] += 1
    total = sum(counter.values())
    return [(w, c, c / total) for w, c in counter.most_common(top_n)]


def get_top_ngrams(tokens, n, top_n=20):
    ngrams = nltk.ngrams(tokens, n)
    counter = Counter(ngrams)
    total = sum(counter.values())
    return [(ng, c, c / total) for ng, c in counter.most_common(top_n)]


def get_top_pos_ngrams(pos_tags, n, top_n=20):
    tags = [tag for tag in pos_tags]
    return get_top_ngrams(tags, n, top_n)


def get_verb_noun_bigrams(pos_tags):
    bigram_counter = Counter()
    verb_counter = Counter()
    noun_counter = Counter()
    total_bigrams = 0

    for i in range(len(pos_tags) - 1):
        (w1, t1), (w2, t2) = pos_tags[i], pos_tags[i + 1]
        if t1.startswith('VB') and t2.startswith('NN'):
            bigram = (w1, w2)
            bigram_counter[bigram] += 1
            verb_counter[w1] += 1
            noun_counter[w2] += 1
            total_bigrams += 1

    results = []
    for (w1, w2), freq in bigram_counter.items():
        p_joint = freq / total_bigrams
        p_w1 = verb_counter[w1] / total_bigrams
        p_w2 = noun_counter[w2] / total_bigrams
        p_cond = freq / verb_counter[w1]
        mi = log2(p_joint / (p_w1 * p_w2)) if p_w1 * p_w2 > 0 else 0
        lmi = freq * mi
        results.append({
            'bigram': (w1, w2),
            'freq': freq,
            'rel_freq': p_joint,
            'p_cond': p_cond,
            'p_joint': p_joint,
            'mi': mi,
            'lmi': lmi
        })

    top_freq = sorted(results, key=lambda x: x['freq'], reverse=True)[:10]
    top_pcond = sorted(results, key=lambda x: x['p_cond'], reverse=True)[:10]
    top_pjoint = sorted(results, key=lambda x: x['p_joint'], reverse=True)[:10]
    top_mi = sorted(results, key=lambda x: x['mi'], reverse=True)[:10]
    top_lmi = sorted(results, key=lambda x: x['lmi'], reverse=True)[:10]

    mi_set = set([el['bigram'] for el in top_mi])
    lmi_set = set([el['bigram'] for el in top_lmi])
    intersection = len(mi_set & lmi_set)

    return {
        'freq': top_freq,
        'pcond': top_pcond,
        'pjoint': top_pjoint,
        'mi': top_mi,
        'lmi': top_lmi,
        'intersection': intersection
    }


def main(file):
    text = open_read(file)
    tokens, pos_tags = tokenizer_tag(text)

    ''' Creazione del file di output con relativo output '''
    with open("output_programma2.txt", "w", encoding="utf-8") as out:

        # 1. Top-50 Nouns, Adjectives, Adverbs
        write_line(out, "=== Top-50 NOUNs, ADJs, and ADVs ===")
        for word, count, rel in ricava_top_pos(pos_tags, ['NN', 'JJ', 'RB'], 50):
            write_line(out, f"{word}\t{count}\t{rel:.4f}")

        # 2. Top-20 n-grams for n=1,2,3
        for n in [1, 2, 3]:
            write_line(out, f"\n=== Top-20 {n}-grams ===")
            for ng, count, rel in get_top_ngrams(tokens, n):
                write_line(out, f"{' '.join(ng)}\t{count}\t{rel:.4f}")

        # 3. Top-20 PoS n-grams for n=1,2,3,4,5
        for n in [1, 2, 3, 4, 5]:
            write_line(out, f"\n=== Top-20 POS {n}-grams ===")
            for ng, count, rel in get_top_pos_ngrams(pos_tags, n):
                write_line(out, f"{' '.join(ng)}\t{count}\t{rel:.4f}")

        # 4. Bigrammi VERBO + SOSTANTIVO
        write_line(out, f"\n=== Top-10 Verb-Noun Bigrams ===")
        bigram_data = get_verb_noun_bigrams(pos_tags)

        for key in ['freq', 'pcond', 'pjoint', 'mi', 'lmi']:
            write_line(out, f"\n-- Ordered by {key.upper()} --")
            for item in bigram_data[key]:
                b = item['bigram']
                write_line(out, f"{b[0]} {b[1]}\t{item['rel_freq']:.4f}\tPcond={item['p_cond']:.4f}\tPjoint={item['p_joint']:.4f}\tMI={item['mi']:.4f}\tLMI={item['lmi']:.4f}")

        write_line(out, f"\n=== Common Elements between Top-10 MI and Top-10 LMI: {bigram_data['intersection']} ===")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usare: python3 programma_2.py <corpus.txt>")
        sys.exit(1)

    main(sys.argv[1])
