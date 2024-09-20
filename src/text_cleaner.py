import re
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Some languages supported by SnowballStemmer from NLTK
LANGUAGE_NLTK_MAP = {
    'en': 'english',
    'it': 'italian',
    'fr': 'french',
    'es': 'spanish',
    'pt': 'portuguese',
    'de': 'german'
}

def remove_text_punctuation(document: str) -> str:
    """
    Removes punctuation from a given string.

    Parameters:
    document (str): The input string to be processed.

    Returns:
    str: The input string with all punctuation removed.
    """
    # If not a string, or empty, returns an empty text
    if not document or not isinstance(document, str):
        return ''

    # Return text without any punctuation
    return re.sub(r'[,.;@#?!&$ "\n\(\)]+', ' ', document).strip()


class TextCleaner:  # pylint: disable=too-few-public-methods
    def __init__(self, language='en') -> None:
        """
        Initializes the Text Cleaner object.

        Parameters:
        - language (str): The language of the code. Default is 'en' (English).

        Raises:
        - Exception: If the specified language is not supported.
        """
        # Assert supported languages
        if language not in LANGUAGE_NLTK_MAP.keys():
            raise Exception(f'Language {language} not supported.')

        # Map language code and full language name
        self.language = language
        lang_name = LANGUAGE_NLTK_MAP[language]

        # Initialize class objects
        self.stemmer = SnowballStemmer(lang_name)
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]\w+')
        self.stop_word = set(stopwords.words(lang_name))


    def __call__(self, text: str) -> str:
        """
        Tokenizes and stems the input text, returning a list of stemmed words.

        Parameters:
        - text (str): The input text to be tokenized and stemmed.

        Returns:
        - str: The text after clean up process.
        """
        return ' '.join([
            self.stemmer.stem(word)
            for word in self.tokenizer.tokenize(remove_text_punctuation(text))
            if word.isalpha() and not word in self.stop_word and len(word) > 1])
