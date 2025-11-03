This repository contains a Python application conceived as an LLM‑centric experiment in text understanding. The code analyses English corpora with a workflow that mirrors how a large language model reasons about language: it reads raw text, segments it into sentences and tokens, tags parts of speech, lemmatizes terms, estimates sentiment at sentence level, and surfaces statistically grounded signals such as bigram associations and named entities. The intent is to study how LLM‑style insights can be approximated or complemented with transparent, reproducible steps, while keeping the door open to plugging an actual LLM for higher‑level interpretation.

The project ships with two independent programs. The first compares two corpora and produces a narrative report that synthesizes distributional properties, salient bigrams scored by mutual information and its local variant, Markov‑based plausibility of sentences, stopword coverage and pronoun usage, ending with a consolidated sentiment profile. The second digs into a single corpus to extract its lexical backbone: it lemmatizes and tags the stream, highlights the most characteristic nouns, adjectives and adverbs, and enumerates named entities by type to support stylistic and topical inspection. Both programs write their results to human‑readable text files so that analysis can be inspected, shared and versioned.

The repository includes example corpora that are long‑form political speeches saved in UTF‑8 and suitable for experimentation. The code does not hard‑wire assumptions about the domain and will operate on any sufficiently sized English text. By design, the workflow emphasizes clarity over cleverness: every step is explicit and auditable, echoing the goal of explaining how a model reaches its conclusions rather than treating the process as a black box.

## How to run

The application is plain Python and depends on NLTK only. A recent Python 3 interpreter is recommended. It is practical to create a virtual environment, install the single dependency, and download the required linguistic resources once before running the programs. The code contains commented instructions to let NLTK fetch its models; if a LookupError appears, simply download the listed resources and re‑run.

Create and activate an environment, install dependencies, and fetch the data. On Unix‑like systems:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install nltk
python - <<'PY'
import nltk
# Use the exact resource identifiers referenced in the source files.
for pkg in [
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "vader_lexicon",
    # The following are commonly needed by named entity chunking and stopword utilities.
    "maxent_ne_chunker_tab",
    "words",
    "stopwords",
]:
    try:
        nltk.download(pkg)
    except Exception as e:
        print(f"Warning: could not download {pkg}: {e}")
print("NLTK setup complete.")
PY
```

Run the comparative analysis on two corpora with the first program. It expects two UTF‑8 text files and writes a report named after the inputs. From the project root:

```
python programma_1.py RobertKennedySpeech.txt TrumpSpeech.txt
```

Run the single‑corpus analysis with the second program. It expects one UTF‑8 text file and produces a textual report with named entities and top lexical items:

```
python programma_2.py TrumpSpeech.txt
```

Output files are placed in the same directory and are named with a descriptive prefix, for example `programma1_RobertKennedySpeech.txt`, `programma1_TrumpSpeech.txt`, and `programma2_TrumpSpeech.txt`. These reports are designed to be read in a text editor and committed to version control alongside the source, so results can be reproduced across environments.

## Project structure

The root contains the two entry points, the sample corpora, and a short Italian brief that documents the academic assignment the implementation follows. The logic is written in straightforward functions that can be imported in notebooks if an interactive exploration is preferred. No external services are required and the code runs fully offline once NLTK data have been downloaded.

## What “LLM‑centric” means here

Although the implementation relies on NLTK rather than on an API‑backed transformer, the analysis is framed through an LLM lens. The steps emulate how a large language model decomposes a prompt: tokenization and part‑of‑speech tagging expose grammatical roles, lemmatization abstracts surface variation, association measures approximate multi‑token dependencies, and sentence‑level polarity offers a compact affective prior. The outputs make the reasoning explicit and can be used as scaffolding to integrate an actual LLM for summarization, stylistic comparison or zero‑shot classification while keeping a transparent baseline as a reference.

## Reproducibility notes

Results depend on the versions of Python, NLTK and its resource packages. Pinning these in a requirements file or capturing them with an environment manager will keep analyses stable over time. The included corpora are plain text in UTF‑8; if you substitute your own documents, ensure they meet the same encoding and contain enough sentences for statistics to be meaningful.
