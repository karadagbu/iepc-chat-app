# header_extractor.py

import re
import pdfplumber
import pycountry

# Build a lowercase set of all country names (official, common, and official variants)
ALL_COUNTRIES = set(
    c.name.lower() for c in pycountry.countries if hasattr(c, "name")
) | set(
    getattr(c, "common_name", "").lower() for c in pycountry.countries if hasattr(c, "common_name")
) | set(
    getattr(c, "official_name", "").lower() for c in pycountry.countries if hasattr(c, "official_name")
)

def find_countries_in_text(text: str) -> list[str]:
    """
    Look for any country names in `text` (case-insensitive).
    Returns a deduplicated list of normalized country names.
    """
    found = set()
    lower = text.lower()
    for country in ALL_COUNTRIES:
        # Whole-word match (handles multi-word names)
        if re.search(rf"\b{re.escape(country)}\b", lower):
            try:
                c_obj = pycountry.countries.lookup(country)
                found.add(c_obj.name)
            except (LookupError, KeyError):
                found.add(country.title())
    return list(found)


def split_authors_line(line: str) -> list[str]:
    """
    Heuristic to split an author line into individual names.
    Strips out footnote numbers like “¹” or “[2]” and splits on commas or " and ".
    E.g. "Jane Doe¹, John Roe² and Alice Smith³" → ["Jane Doe", "John Roe", "Alice Smith"]
    """
    # Remove digits or bracketed numbers
    cleaned = re.sub(r"(\d+|\[\d+\])", "", line)
    parts = re.split(r",\s*| and\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def extract_header_heuristic(pdf_path: str) -> dict:
    """
    Extract header info from page 1 of a PDF (no OCR, no translation):
      - title: first non-blank line
      - authors: first following line with ≥2 capitalized words
      - affiliations: subsequent lines containing Univ/Dept/Inst until "Abstract"/"Keywords"
      - countries: any country names found in those affiliation lines

    Returns a dict:
      {
        "title": str | None,
        "authors": [str,…],
        "affiliations": [str,…],
        "countries": [str,…]
      }
    """
    info = {"title": None, "authors": [], "affiliations": [], "countries": []}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            raw = first_page.extract_text() or ""
    except Exception:
        return info

    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return info

    # 1. Title = first non-blank line
    info["title"] = lines[0]

    # 2. Find the author line: the first line after title with ≥2 capitalized tokens
    name_pat = re.compile(r"\b[A-Z][a-zA-Z]+\b")
    author_idx = None
    for i in range(1, min(len(lines), 6)):
        candidates = name_pat.findall(lines[i])
        if len(candidates) >= 2:
            author_idx = i
            break

    if author_idx is None:
        return info

    info["authors"] = split_authors_line(lines[author_idx])

    # 3. Collect affiliation lines until we hit "Abstract", "Keywords", or similar
    affs = []
    for j in range(author_idx + 1, len(lines)):
        ln = lines[j]
        if re.match(r"^(abstract|keywords|introduction)\b", ln, re.IGNORECASE):
            break
        if re.search(r"\b(Univ|University|Dept|Department|Inst|Institute|College)\b", ln, re.IGNORECASE):
            affs.append(ln)
        else:
            # If the previous line was an affiliation and this line has a country, append it
            if affs and find_countries_in_text(ln):
                affs[-1] += ", " + ln

    info["affiliations"] = affs

    # 4. Extract countries from each affiliation
    countries = set()
    for a in affs:
        for c in find_countries_in_text(a):
            countries.add(c)
    info["countries"] = list(countries)

    return info
