import nltk

class Analyzer():
    """Implements sentiment analysis."""

    def __init__(self, positives, negatives):
        """Initialize Analyzer."""
        self.tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        self.positives = self.read_words(positives)
        self.negatives = self.read_words(negatives)
    
    def read_words(self, filename):
        """Return list of words in the given file, removing the comments"""
        return [line.rstrip('\n') for line in open(filename) if line != '\n' and ';' not in line]

    def analyze(self, text):
        """Analyze text for sentiment, returning its score."""
        tokens = self.tokenizer.tokenize(text)
        positive_words = set(tokens) & set(self.positives)
        negative_words = set(tokens) & set(self.negatives)
        return len(positive_words) - len(negative_words)
