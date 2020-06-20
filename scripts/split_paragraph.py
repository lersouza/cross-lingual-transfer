import nltk
import sys
import six


NLTK_LANG_MAP = {
    'en': 'english',
    'pt': 'portuguese'
}


def convert_to_unicode(text):
    """
    Extracted from: https://github.com/facebookresearch/XLM
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


language = sys.argv[1] if len(sys.argv) > 1 else 'en'
nltk_language = NLTK_LANG_MAP[language]

nltk.download('punkt', quiet=True)

for line in sys.stdin:
    line = convert_to_unicode(line.rstrip())

    if not line:
        print("")  # Preserve doc separation
    else:
        sentences = nltk.sent_tokenize(line, nltk_language)

        for sentence in sentences:
            sentence = sentence.rstrip()

            # Only print non empty sentences
            if sentence:
                print(u'%s' % sentence)
