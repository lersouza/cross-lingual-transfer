import nltk
import sys


NLTK_LANG_MAP = {
    'en': 'english',
    'pt': 'portuguese'
}


language = sys.argv[1] if len(sys.argv) > 1 else 'en'
nltk_language = NLTK_LANG_MAP[language]

nltk.download('punkt', quiet=True)

for line in sys.stdin:
    line = line.rstrip()

    if not line:
        print("")  # Preserve doc separation
    else:
        sentences = nltk.sent_tokenize(line, nltk_language)

        for sentence in sentences:
            print(sentence)
