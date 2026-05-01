"""Parse confidence scores (1-10) from varied model output formats."""

import re


def parse_confidence(text):
    """Parse a 1-10 confidence score from model output.

    Handles formats like:
    - "7"
    - "7/10"
    - "7 out of 10"
    - "about 7"
    - "I would say 8/10"
    - "My confidence is 6 out of 10"
    - "Confidence: 8"

    Returns int 1-10. Defaults to 5 if unparseable.
    """
    if text is None:
        return 5

    text = str(text).strip()

    # Pure digit
    if text.isdigit() and 1 <= int(text) <= 10:
        return int(text)

    # "N/10" or "N out of 10"
    match = re.search(r'(\d+)\s*(?:/|out of)\s*10', text)
    if match:
        val = int(match.group(1))
        return max(1, min(val, 10))

    # "confidence: N" or "confidence is N" or "confidence level: N"
    match = re.search(r'confidence[\s:]+(?:is\s+|level[\s:]+)?(\d+)', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 10:
            return val

    # "about/around/roughly/say N"
    match = re.search(r'(?:about|around|roughly|say|approximately)\s*(\d+)', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 10:
            return val

    # "I would rate/give N"
    match = re.search(r'(?:rate|give|assign|say)\s*(?:it\s*)?(?:a\s*)?(\d+)', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 10:
            return val

    # Generic: first standalone 1-10 number
    match = re.search(r'\b(\d{1,2})\b', text)
    if match and 1 <= int(match.group(1)) <= 10:
        return int(match.group(1))

    return 5  # default medium confidence
