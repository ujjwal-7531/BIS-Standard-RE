import argparse
import json
import re
import subprocess
from pathlib import Path


# ── Section map ────────────────────────────────────────────────────────────────
# Maps the SP-21 section headings to a short category tag that gets injected
# into every standard's text before embedding.  This gives the model free
# "what kind of thing is this" signal without any manual labelling.
SECTION_CATEGORIES = {
    "CEMENT AND CONCRETE": "cement concrete aggregates masonry",
    "BUILDING LIMES": "lime building mortar",
    "STONES": "stone masonry natural stone",
    "WOOD PRODUCTS FOR BUILDING": "wood plywood board building",
    "GYPSUM BUILDING MATERIALS": "gypsum plaster board",
    "TIMBER": "timber structural wood",
    "BITUMEN AND TAR PRODUCTS": "bitumen tar waterproofing",
    "FLOOR, WALL, ROOF COVERINGS AND FINISHES": "floor wall roof tile finish",
    "WATER PROOFING AND DAMP PROOFING MATERIALS": "waterproofing dampproofing membrane",
    "SANITARY APPLIANCES AND WATER FITTINGS": "sanitary appliances pipes fittings water",
    "BUILDER'S HARDWARE": "hardware fittings builder",
    "WOOD PRODUCTS": "wood products panel board",
    "DOORS, WINDOWS AND SHUTTERS": "door window shutter frame",
    "CONCRETE REINFORCEMENT": "reinforcement steel bar rebar concrete",
    "STRUCTURAL STEELS": "structural steel section",
    "LIGHT METAL AND THEIR ALLOYS": "aluminium alloy light metal",
    "STRUCTURAL SHAPES": "structural shapes section steel",
    "WELDING ELECTRODES AND WIRES": "welding electrode wire",
    "THREADED FASTENERS AND RIVETS": "bolt nut rivet fastener threaded",
    "WIRE ROPES AND WIRE PRODUCTS": "wire rope strand",
    "GLASS": "glass glazing",
    "FILLERS, STOPPERS AND PUTTIES": "filler stopper putty sealant",
    "THERMAL INSULATION MATERIALS": "insulation thermal",
    "PLASTICS": "plastic pipe tube pvc",
    "CONDUCTORS AND CABLES": "conductor cable electrical",
    "WIRING ACCESSORIES": "wiring accessories switch socket",
    "GENERAL": "general building material",
}

# ── Header regex ────────────────────────────────────────────────────────────────
# Handles every formatting variant found in SP-21:
#   IS 269 : 1989          (standard)
#   IS : 6411-1985         (colon after IS, dash separator)
#   IS 1489 (PART1) : 1991 (no space in PART)
#   IS 771 (PART 3/SEC 1)  (section sub-part)
#   IS 3513 (PART 1) 1989 :(year before colon)
#   IS : 2556 (PART 3) 1994(colon after IS, bare year)
STANDARD_HEADER_RE = re.compile(
    r"SUMMARY OF\s*\n+\s*"
    r"(IS\s*:?\s*\d+\s*"
    r"(?:\(\s*PART\s*[\dIVXivx]+(?:[/\s][A-Z0-9 ]*)?\s*(?:AND\s*\d+)?\s*\))?\s*"
    r"(?:[:\-]\s*\d{4}|\d{4}\s*:|\d{4}))",
    re.IGNORECASE,
)

SECTION_RE = re.compile(
    r"SECTION\s+\d+\s*\n+\s*([A-Z][A-Z,'\s/]+)",
    re.IGNORECASE,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_standard(standard: str) -> str:
    """Canonical form: IS 269 : 1989  /  IS 1489 (Part 1) : 1991"""
    s = normalize_whitespace(standard).upper()
    # fix IS:NNNN → IS NNNN
    s = re.sub(r"^IS\s*:\s*", "IS ", s)
    # normalise part notation
    s = re.sub(r"\(\s*PART\s*(\d+)\s*\)", r"(Part \1)", s, flags=re.I)
    # normalise separator
    s = re.sub(r"\s*[:\-]\s*(\d{4})", r" : \1", s)
    # tidy spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pdf_text(pdf_path: str) -> str:
    """Extract PDF text; prefers pdftotext (better layout), falls back to pypdf."""
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Install poppler-utils (pdftotext) or add pypdf to the environment."
        ) from exc

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_title(text_after_header: str) -> str:
    """Pull the human-readable title that follows the IS-number line."""
    text = normalize_whitespace(text_after_header)
    # Strip leading revision notes / part names
    text = re.sub(r"^\([^)]*(?:revision|part)[^)]*\)\s*", "", text, flags=re.I)
    # Cut at the first structural marker (scope, table, note …)
    title = re.split(
        r"\s+(?:\([^)]*Revision\)|1\.\s*Scope|Scope\s*[—\-]|1\.\s|2\.\s"
        r"|TABLE\s+|Note\s*[—\-]|PART\s+\d)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return normalize_whitespace(title).strip(" -—:")[:250]


def detect_section(position: int, section_positions: list) -> str:
    """Return the section category tag for a given character position."""
    category = ""
    for sec_pos, sec_cat in section_positions:
        if sec_pos <= position:
            category = sec_cat
        else:
            break
    return category


# ── Core extraction ─────────────────────────────────────────────────────────────

def build_section_index(full_text: str) -> list:
    """Return list of (char_position, category_string) sorted by position."""
    positions = []
    for match in SECTION_RE.finditer(full_text):
        heading = normalize_whitespace(match.group(1)).upper()
        for key, cat in SECTION_CATEGORIES.items():
            if key in heading:
                positions.append((match.start(), cat))
                break
    return sorted(positions, key=lambda x: x[0])


def split_into_standards(full_text: str) -> list:
    """
    Split SP-21 text into one record per standard summary.

    Each record:
        standard  – normalised IS identifier  e.g. "IS 269 : 1989"
        title     – human-readable title
        category  – section category keywords
        text      – full chunk text (used for BM25 + embedding)
    """
    section_positions = build_section_index(full_text)
    matches = list(STANDARD_HEADER_RE.finditer(full_text))
    standards = []
    seen: dict[str, int] = {}

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end]

        standard = normalize_standard(match.group(1))
        title = extract_title(chunk[match.end() - start:])
        category = detect_section(start, section_positions)

        # Build an enriched text field:
        # standard identifier + title + category keywords + raw chunk body
        enriched_text = normalize_whitespace(
            f"{standard} {title} {category} {chunk}"
        )

        record = {
            "standard": standard,
            "title": title,
            "category": category,
            "text": enriched_text,
        }

        if standard in seen:
            # Merge duplicate entries (some standards span multiple pages)
            existing = standards[seen[standard]]
            existing["text"] += " " + enriched_text
            if not existing["title"]:
                existing["title"] = title
            if not existing["category"]:
                existing["category"] = category
        else:
            seen[standard] = len(standards)
            standards.append(record)

    return standards


# ── Entry point ─────────────────────────────────────────────────────────────────

def save_processed_data(standards: list, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(standards, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BIS standards retrieval index.")
    parser.add_argument("--pdf", default="dataset.pdf", help="Path to SP-21 dataset PDF")
    parser.add_argument(
        "--output",
        default="data/processed_data.json",
        help="Output path for processed standards JSON",
    )
    args = parser.parse_args()

    print("Loading PDF …")
    text = load_pdf_text(args.pdf)

    print("Splitting into standards …")
    standards = split_into_standards(text)
    print(f"Extracted {len(standards)} standards")

    save_processed_data(standards, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()