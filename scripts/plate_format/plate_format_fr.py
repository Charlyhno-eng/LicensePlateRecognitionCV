import re

# Current French license plate format (since 2009): AB-123-CD or AB 123 CD
pattern_french = re.compile(r"^[A-Z]{2}\s\d{3}\s[A-Z]{2}$")

def normalize_plate_format(plate: str) -> str:
    """
    Attempts to correctly format a poorly spaced or misformatted plate.
    Example: 'AB123CD'   -> 'AB 123 CD'
             'AB-123-CD' -> 'AB 123 CD'
    """
    plate = plate.strip().upper()
    plate = re.sub(r'[-]', '', plate)
    plate = re.sub(r'\s+', '', plate)

    if len(plate) == 7:
        return f"{plate[:2]} {plate[2:5]} {plate[5:]}"
    return plate

def is_valid_plate(plate: str) -> bool:
    """
    Checks whether a license plate strictly matches the French format.
    Applies minimal normalization before validation.
    """
    plate = plate.strip().upper()
    plate = re.sub(r'\s+', ' ', plate)
    plate = normalize_plate_format(plate)

    return bool(pattern_french.match(plate))
