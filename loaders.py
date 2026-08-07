import re
import unicodedata
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
import docx  # python-docx
from langsmith import traceable


@traceable(run_type="tool", name="clean_text")
def clean_text(text: str) -> str:
    """General cleanup applied to every loader's output before it leaves this file."""
    text = unicodedata.normalize("NFKC", text)          # normalize unicode (smart quotes, etc.)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]  # strip trailing whitespace per line
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)                # collapse 3+ blank lines to 1
    text = re.sub(r"[ \t]{2,}", " ", text)                 # collapse repeated spaces/tabs
    return text.strip()


@traceable(run_type="tool", name="load_website")
def load_website(url: str) -> str:
    """Fetch a webpage and extract visible text."""
    headers = {
        # Many sites (Wikipedia included) return 403 for requests' default
        # "python-requests/x.x" User-Agent as basic bot protection. A normal
        # browser User-Agent avoids that without doing anything deceptive -
        # we're not spoofing headers to bypass paywalls or scraping restrictions,
        # just avoiding the default that flags us as an obvious script.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except requests.RequestException as e:
        raise ValueError(f"Could not fetch website: {e}")
    return text


def _extract_video_id(url: str) -> str:
    """Handles both youtube.com/watch?v=ID and youtu.be/ID links."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract a video ID from: {url}")
    return match.group(1)


@traceable(run_type="tool", name="load_youtube")
def load_youtube(url: str) -> str:
    """Extract transcript text from a YouTube video URL, stripping [Music]/[Applause]-style noise."""
    video_id = _extract_video_id(url)
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        text = " ".join(snippet.text for snippet in fetched)
    except Exception as e:
        raise ValueError(f"Could not fetch transcript for {video_id}: {e}")
    text = re.sub(r"\[.*?\]", "", text)   # remove [Music], [Applause], [inaudible], etc.
    return text


@traceable(run_type="tool", name="load_pdf")
def load_pdf(path: str) -> str:
    """Extract text from a PDF file. Skips individual pages that fail to parse."""
    try:
        reader = PdfReader(path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            print(f"Warning: skipped page {i + 1} of PDF ({e})")
    return "\n".join(pages)


@traceable(run_type="tool", name="load_docx")
def load_docx(path: str) -> str:
    """Extract text from a Word document (paragraphs only, no tables)."""
    try:
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise ValueError(f"Could not read DOCX: {e}")


@traceable(run_type="tool", name="load_markdown")
def load_markdown(path: str) -> str:
    """Read a markdown file as-is — we keep the ## headers, chunking uses them later."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not read markdown file: {e}")


@traceable(run_type="chain", name="load_document")
def load_document(source: str, source_type: str) -> str:
    """Single entry point — 'Loader Selection' step in the diagram.
    Runs the right loader, then applies general text cleaning, then checks
    the result isn't empty (fails loudly instead of passing empty text downstream)."""
    loaders = {
        "website": load_website,
        "youtube": load_youtube,
        "pdf": load_pdf,
        "docx": load_docx,
        "markdown": load_markdown,
    }
    if source_type not in loaders:
        raise ValueError(f"Unsupported source type: {source_type}")

    raw_text = loaders[source_type](source)
    text = clean_text(raw_text)

    if not text:
        raise ValueError(
            f"No text could be extracted from this {source_type} source. "
            f"(For PDFs, this usually means it's a scanned image with no text layer - OCR isn't supported yet.)"
        )
    return text