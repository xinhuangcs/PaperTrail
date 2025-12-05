import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from typing import List, Dict, Any, Callable
from functools import reduce
from multiprocessing import Pool, cpu_count
from pathlib import Path
import os
from pathlib import Path

# Initialize stemmer for word stemming
stemmer = PorterStemmer()


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    #Load entire JSON file content from input path.
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def to_lowercase(text: str) -> str:
    #Convert text to lowercase
    return text.lower()


def remove_special_characters(text: str) -> str:
    # Remove newlines, extra whitespace, and special punctuation
    cleaned = re.sub(r'[\n\r]+', ' ', text)  # Replace newlines with space
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)  # Remove punctuation, keep alphanumeric and spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with single space
    return cleaned.strip()


def stem_text(text: str) -> str:
    #Apply stemming to reduce words to their root form
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def combine_title_abstract(paper: Dict[str, Any]) -> str:
    #Combine title and abstract fields
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    return f"{title} {abstract}"


def process_paper_content(paper: Dict[str, Any]) -> str:
    # Combine title and abstract
    combined_text = combine_title_abstract(paper)

    # Apply processing pipeline
    pipeline = [
        to_lowercase,
        remove_special_characters,
        stem_text
    ]

    # Apply all functions in sequence using functional composition
    return reduce(lambda text, func: func(text), pipeline, combined_text)


def process_papers_parallel(papers: List[Dict[str, Any]], num_processes: int = 8) -> List[Dict[str, Any]]:
    #Process all papers through the content pipeline using parallel processing

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_paper_content, papers)

    # Add processed content back to original papers
    processed_papers = []
    for i, paper in enumerate(papers):
        paper_copy = paper.copy()
        paper_copy['processed_content'] = results[i]
        processed_papers.append(paper_copy)

    return processed_papers


def process_large_dataset_batch(input_file_path: str, batch_size: int = 1000) -> None:
    output_path = get_output_path(input_file_path)

    print(f"Processing large dataset in batches of {batch_size}...")

    def sanitize(s: str) -> str:
        return s.lstrip('\ufeff').replace('\x00', '').strip()

    with open(input_file_path, 'r', encoding='utf-8-sig') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        batch = []
        total_processed = 0
        line_no = 0

        for raw_line in infile:
            line_no += 1
            line = sanitize(raw_line)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode failed at line {line_no}: {e}")
                print(f"       Line preview (first 120 chars): {raw_line[:120]!r}")
                continue

            batch.append(obj)

            if len(batch) >= batch_size:
                processed_batch = process_papers_parallel(batch, num_processes=min(cpu_count(), 8))
                for paper in processed_batch:
                    json.dump(paper, outfile, ensure_ascii=False)
                    outfile.write('\n')
                total_processed += len(batch)
                print(f"Processed {total_processed} papers...")
                batch = []

        if batch:
            processed_batch = process_papers_parallel(batch, num_processes=min(cpu_count(), 8))
            for paper in processed_batch:
                json.dump(paper, outfile, ensure_ascii=False)
                outfile.write('\n')
            total_processed += len(batch)

        print(f"Completed! Total processed: {total_processed} papers")
        print(f"Saved to: {output_path}")


def save_preprocessed_data(papers: List[Dict[str, Any]], output_path: str) -> None:
    #Save preprocessed papers to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as file:
        for paper in papers:
            json.dump(paper, file, ensure_ascii=False)
            file.write('\n')
    print(f"Saved {len(papers)} preprocessed papers to {output_path}")


def get_output_path(input_path: str) -> str:
    #Generate output path with _preprocessed suffix
    path = Path(input_path)
    output_path = path.parent / f"{path.stem}_preprocessed{path.suffix}"
    return str(output_path)


def main(input_file_path: str) -> None:
    #Main function to load and process papers
    # 1. Load JSON file content
    print("Loading papers...")
    papers = load_json_file(input_file_path)
    print(f"Loaded {len(papers)} papers")

    # 2-5. Process all papers through the pipeline with parallel processing
    processed_papers = process_papers_parallel(papers)

    # 6. Save preprocessed data to new file
    output_path = get_output_path(input_file_path)
    save_preprocessed_data(processed_papers, output_path)

    # Display first few processed documents as examples
    print("\nFirst 3 processed documents:")
    for i, paper in enumerate(processed_papers[:3]):
        content = paper['processed_content']
        print(f"Document {i}: {content[:200]}...")


# Example usage
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[2]
    input_path = ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations-final-dataset.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    print("Using batch processing for large dataset...")
    process_large_dataset_batch(input_path, batch_size=1000)
