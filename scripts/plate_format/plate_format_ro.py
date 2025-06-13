import re

RO_PREFIXES = {
    "AB", "AR", "AG", "BC", "BH", "BN", "BR", "BT", "BV", "BZ",
    "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ",
    "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT",
    "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN",
    "B"
}

# Format 1: B 123 ABC
pattern_bucharest = re.compile(r"^B\s\d{3}\s[A-Z]{3}$")

# Format 2: CJ 12 XYZ
pattern_regional = re.compile(rf"^({'|'.join(RO_PREFIXES - {'B'})})\s\d{{2}}\s[A-Z]{{3}}$")

def normalize_plate_format(plate: str) -> str:
    """
    Attempts to correctly shape a poorly spaced but valid plate.
    For exemple : 'B865MHQ' => 'B 865 MHQ'
                  'CJ12XYZ' => 'CJ 12 XYZ'
    """
    plate = plate.strip().upper().replace("-", "").replace(" ", "")
    if len(plate) == 7 and plate[0] == 'B':
        return f"{plate[0]} {plate[1:4]} {plate[4:]}"
    elif len(plate) == 7 and plate[:2] in RO_PREFIXES:
        return f"{plate[:2]} {plate[2:4]} {plate[4:]}"
    return plate

def is_valid_plate(plate: str) -> bool:
    """
    Checks if a plate exactly matches the RO format.
    Applies minimal normalization before testing.
    """
    plate = plate.strip().upper()
    plate = re.sub(r'\s+', ' ', plate)
    plate = normalize_plate_format(plate)

    return bool(pattern_bucharest.match(plate) or pattern_regional.match(plate))
