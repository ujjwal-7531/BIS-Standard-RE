import argparse
import json
import re
import subprocess
from pathlib import Path


STANDARD_HEADER_RE = re.compile(
    r"SUMMARY OF\s*\n+\s*"
    r"(IS\s*:?\s*\d+\s*"                          # IS [optional colon] NUMBER
    r"(?:\(\s*PART\s*[\dIVXivx]+(?:[/\s][A-Z0-9 ]*)?\s*(?:AND\s*\d+)?\s*\))?\s*"  # optional (PART ...)
    r"[:\-]\s*\d{4})",                             # colon or dash + year
    re.IGNORECASE,
)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_standard(standard):
    standard = normalize_whitespace(standard).upper()
    standard = re.sub(r"\(\s*PART\s*(\d+)\s*\)", r"(Part \1)", standard, flags=re.I)
    standard = re.sub(r"\s*:\s*", ": ", standard)
    standard = re.sub(r"^IS\s*", "IS ", standard)
    return standard


def load_pdf_text(pdf_path):
    """Extract PDF text with layout order good enough for SP 21 summary headings."""
    pdf_path = str(pdf_path)
    try:
        completed = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Install poppler-utils for pdftotext or add pypdf to the environment."
        ) from exc

    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_title(text_after_standard):
    text = normalize_whitespace(text_after_standard)
    title = re.split(
        r"\s+(?:\([^)]*Revision\)|1\.\s*Scope|Scope\s*[—-]|1\.\s|2\.\s|TABLE\s+|Note\s*[—-])",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return normalize_whitespace(title).strip(" -—:")[:250]


def split_into_standards(full_text):
    matches = list(STANDARD_HEADER_RE.finditer(full_text))
    standards = []
    seen = {}

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end]
        standard = normalize_standard(match.group(1))
        title = extract_title(chunk[match.end() - start :])
        record = {
            "standard": standard,
            "title": title,
            "text": normalize_whitespace(chunk),
        }

        if standard in seen:
            standards[seen[standard]]["text"] += " " + record["text"]
            if not standards[seen[standard]].get("title"):
                standards[seen[standard]]["title"] = title
        else:
            seen[standard] = len(standards)
            standards.append(record)

    return standards


def save_processed_data(standards, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(standards, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Build the BIS standards retrieval index.")
    parser.add_argument("--pdf", default="dataset.pdf", help="Path to the SP 21 dataset PDF")
    parser.add_argument(
        "--output",
        default="data/processed_data.json",
        help="Path for the processed standards JSON",
    )
    args = parser.parse_args()

    print("Loading PDF...")
    text = load_pdf_text(args.pdf)

    print("Splitting into standards...")
    standards = split_into_standards(text)
    print(f"Extracted {len(standards)} standards")

    save_processed_data(standards, args.output)
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
