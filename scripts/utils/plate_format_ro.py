import re

RO_PREFIXES = {
    "AB", "AR", "AG", "BC", "BH", "BN", "BR", "BT", "BV", "BZ",
    "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ",
    "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT",
    "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN",
    "B"
}

PLATE_PATTERNS = [
    re.compile(r"^B\s?\d{3}\s?[A-Z]{3}$"),  # Ex: B 123 ABC
    re.compile(rf"^({'|'.join(RO_PREFIXES)})\s?\d{{2,3}}\s?[A-Z]{{3}}$")  # Ex: CJ 12 XYZ
]

def is_valid_ro_plate(plate: str) -> bool:
    """Check if a plate matches the Romanian format"""
    plate = plate.strip().upper().replace("-", " ")
    plate = re.sub(r'\s+', ' ', plate)  # Normalize spaces
    return any(pattern.match(plate) for pattern in PLATE_PATTERNS)
