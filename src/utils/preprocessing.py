import re

def clean_text(text):
    """Clean text"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text
