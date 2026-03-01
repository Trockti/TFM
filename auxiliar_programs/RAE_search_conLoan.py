"""
Search which words are included in the DLE (dle.rae.es) from a list of words fetched from a URL.
"""
import csv
import time
import io
import requests
from bs4 import BeautifulSoup


DLE_SEARCH_URL = "https://rae-api.com/api/words/{word}"


def fetch_tsv_from_github(url: str) -> list[str]:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    content = response.text
    column = "ale"

    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    if column not in reader.fieldnames:
        raise ValueError(f"Column '{column}' not found. Available columns: {reader.fieldnames}")
    
    rows = list(reader)  # consume the iterator once into a list
    print(f"Found column '{column}' with {len(rows)} entries.")
    return [row[column].strip() for row in rows]



def is_in_dle(word: str, session: requests.Session) -> bool:
    """
    Check if a word exists in the DLE (dle.rae.es).
    Returns True if the word has an entry, False otherwise.
    """
    url = DLE_SEARCH_URL.format(word=word.lower())
    response = session.get(url, timeout=10)
    print(f"Checked '{word}': HTTP {response.status_code}", end=" ")
    if response.status_code != 200:
        print(f"Word '{word}' not found (404).")
        return False
    return True


def search_words_in_dle(words: list[str], delay: float = 1.0) -> list[str]:
    """
    Search each word in the DLE.
    Returns a list of words that were found.
    """
    found = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; DLE-lookup-script/1.0)"
    })

    for i, word in enumerate(words, 1):
        print(f"[{i}/{len(words)}] Checking '{word}'...", end=" ")
        if is_in_dle(word, session):
            print("✓ Found")
            found.append(word)
        else:
            print("✗ Not found")
        # Be polite to the RAE server
        time.sleep(delay)

    return found


def main():
    url = "https://raw.githubusercontent.com/ZurichNLP/ConLoan/refs/heads/main/loanwords/Spanish_loanwords.tsv"

    # Load words
    print(f"Fetching TSV from: {url}")
    words = fetch_tsv_from_github(url)

    print(f"\nFound {len(words)} word(s) in the 'ale' column. Searching the DLE...\n")
    found_words = search_words_in_dle(words, delay=1.0)

    print(f"\n{'='*40}")
    print(f"Words found in the DLE ({len(found_words)}/{len(words)}):")
    print(found_words)
    return found_words


if __name__ == "__main__":
    main()