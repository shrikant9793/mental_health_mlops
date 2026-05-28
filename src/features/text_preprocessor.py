import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove numbers from text."""
    return re.sub(r"\d+", "", text)


def remove_extra_spaces(text: str) -> str:
    """Remove extra whitespace."""
    return " ".join(text.split())


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text."""
    return " ".join([w for w in text.split() if w not in stop_words])


def lemmatize_text(text: str) -> str:
    """Lemmatize each word in text."""
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])


def clean_text(text: str) -> str:
    """Full text cleaning pipeline."""
    text = text.lower()
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = remove_extra_spaces(text)
    return text