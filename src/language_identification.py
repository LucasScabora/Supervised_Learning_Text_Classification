import os
import fasttext
import urllib.request
from typing import Tuple
fasttext.FastText.eprint = lambda x: None

class LanguageIdentification:
    def __init__(self):
        """
        Initializes the Language Identifier object.
        """
        # Download Language Identification Model (in gitignore)
        if not os.path.exists('lid.176.bin'):
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin',
                'lid.176.bin')
        
        # Load the model
        self.model = fasttext.load_model('lid.176.bin')

    def get_language_and_score(self, text: str) -> Tuple[str, float]:
        """
        Classifies the language of the given document text and returns a tuple containing the language code and its score.

        Parameters:
        - doc_text (str): The document text to classify.

        Returns:
        - Tuple[str, float]: A tuple containing the language code and its respective score. 
        The language code is a string, and the score is a float value between 0 and 1 
        indicating the confidence level of the classification.
        """
        # Replace Linebreaks, if any
        singleline_text = text.replace('\n', ' ')

        # Get language predictions
        labels, scores = self.model.predict(singleline_text, k=-1)
        language_scores = {label.replace('__label__', ''): score
                           for label, score in zip(labels, scores) }

        # Assign language to the highest score, and return it
        classified_lang_code  = max(language_scores, key=language_scores.get)
        return ( classified_lang_code, language_scores[classified_lang_code] )
