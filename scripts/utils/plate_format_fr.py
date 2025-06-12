import re

FR_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}$")

def is_valid_fr_plate(plate: str) -> bool:
    """Check if a plate matches the French format (SIV since 2009)"""
    plate = plate.strip().upper().replace("-", " ")
    plate = re.sub(r"\s+", " ", plate)
    return bool(FR_PLATE_PATTERN.match(plate))
