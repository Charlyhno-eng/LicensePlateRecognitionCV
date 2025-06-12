def normalize_plate_characters(text: str) -> str:
    """
    Replace commonly misread characters in license plates (e.g., OCR confusion)
    """
    substitutions = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '5': 'S',
        '6': 'G',
        '8': 'B',
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6',
    }

    corrected = ''
    for char in text.upper():
        if char in substitutions:
            corrected += substitutions[char]
        else:
            corrected += char

    return corrected
