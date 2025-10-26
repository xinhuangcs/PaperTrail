""""

template:
python /Users/jasonh/Desktop/02807/PaperTrail/src/search/similarity_search.py --query "give me a paper about machine learning deep learning" --method tfidf --top_k 10
python /Users/jasonh/Desktop/02807/PaperTrail/src/search/similarity_search.py --query "neural networks" --method lsa --top_k 5
"""

import argparse
import json
import os
import re
import time
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy import sparse
import joblib


try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except ImportError:
    STEMMER = None
    print("[warning] NLTK not installed, will skip stemming step")

# config paths
BASE_DIR = "/Users/jasonh/Desktop/02807/PaperTrail"
TFIDF_DIR = os.path.join(BASE_DIR, "data/tf_idf")
LSA_DIR = os.path.join(BASE_DIR, "data/lsa")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/similarity_results")

# File paths
TFIDF_MATRIX_PATH = os.path.join(TFIDF_DIR, "tfidf_matrix.npz")
TFIDF_VECTORIZER_PATH = os.path.join(TFIDF_DIR, "tfidf_vectorizer.joblib")
DOC_IDS_PATH = os.path.join(TFIDF_DIR, "doc_ids.npy")
DOC_TITLES_PATH = os.path.join(TFIDF_DIR, "doc_titles.npy")
LSA_MATRIX_PATH = os.path.join(LSA_DIR, "lsa_reduced.npz")
LSA_MODEL_PATH = os.path.join(LSA_DIR, "lsa_model.joblib")


class SimilaritySearch:
    """Class for similar paper retrieval."""
    
    def __init__(self, method: str = "tfidf"):
        """
        Initializes the searcher.
        
        Args:
            method: Search method ("tfidf" or "lsa").
        """
        self.method = method.lower()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.lsa_matrix = None
        self.lsa_model = None
        self.doc_ids = None
        self.doc_titles = None
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Loads the required models and data."""
        print(f"[INFO] Loading data for {self.method.upper()} search...")
        
        # Load document metadata (needed for both TF-IDF and LSA)
        self.doc_ids = np.load(DOC_IDS_PATH, allow_pickle=True)
        self.doc_titles = np.load(DOC_TITLES_PATH, allow_pickle=True)
        print(f"[info] loaded {len(self.doc_ids)} papers' metadata")
        
        if self.method == "tfidf":
            self.tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
            self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            print(f"[INFO] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            
        elif self.method == "lsa":

            lsa_data = np.load(LSA_MATRIX_PATH)
            self.lsa_matrix = lsa_data['X_reduced']

            self.lsa_model = joblib.load(LSA_MODEL_PATH)

            self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            print(f"[INFO] LSA matrix shape: {self.lsa_matrix.shape}")
            print(f"[INFO] LSA model loaded, number of components: {self.lsa_model.n_components}")
            
        else:
            raise ValueError(f"Unsupported method: {self.method}. Please choose 'tfidf' or 'lsa'")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocesses the query text to match the processing of the training corpus.
        
        Args:
            query: The raw query text.
            
        Returns:
            The preprocessed query text.
        """

        query = query.lower()
        
   
        query = re.sub(r"[^a-z0-9\s]+", " ", query)
        

        query = re.sub(r"\s+", " ", query).strip()
        
        if STEMMER:
            tokens = query.split()
            tokens = [STEMMER.stem(token) for token in tokens]
            query = " ".join(tokens)
        
        return query
    
    def search_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Performs search using TF-IDF + Cosine Similarity.
        
        Args:
            query: The query text.
            top_k: The number of top results to return.
            
        Returns:
            A list of results, where each element is a tuple of (doc_id, title, score).
        """

        processed_query = self.preprocess_query(query)
        

        query_vector = self.vectorizer.transform([processed_query])

        similarity_scores = (query_vector @ self.tfidf_matrix.T).toarray().ravel()
        

        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0:  
                results.append((
                    str(self.doc_ids[idx]),
                    str(self.doc_titles[idx]),
                    float(similarity_scores[idx])
                ))
        
        return results
    
    def search_lsa(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Performs search using LSA + Cosine Similarity.
        
        Args:
            query: The query text.
            top_k: The number of top results to return.
            
        Returns:
            A list of results, where each element is a tuple of (doc_id, title, score).
        """

        processed_query = self.preprocess_query(query)
        

        query_tfidf = self.vectorizer.transform([processed_query])
        
        query_lsa = self.lsa_model.transform(query_tfidf)
        

        from sklearn.preprocessing import normalize
        query_lsa_norm = normalize(query_lsa)
        lsa_matrix_norm = normalize(self.lsa_matrix)
        
        similarity_scores = np.dot(query_lsa_norm, lsa_matrix_norm.T).ravel()
        

        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0: 
                results.append((
                    str(self.doc_ids[idx]),
                    str(self.doc_titles[idx]),
                    float(similarity_scores[idx])
                ))
        
        return results
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Executes the similar paper search.
        
        Args:
            query: The query text.
            top_k: The number of top results to return.
            
        Returns:
            A list of results, where each element is a tuple of (doc_id, title, score).
        """
        print(f"[INFO] Searching with {self.method.upper()} method: '{query}'")
        
        start_time = time.time()
        
        if self.method == "tfidf":
            results = self.search_tfidf(query, top_k)
        elif self.method == "lsa":
            results = self.search_lsa(query, top_k)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        elapsed_time = time.time() - start_time
        print(f"[INFO] Search completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def save_results(self, query: str, results: List[Tuple[str, str, float]], 
                    output_file: str = None) -> str:
        """
        Saves the search results to a file.
        
        Args:
            query: The query text.
            results: The search results.
            output_file: The output filename (optional).
            
        Returns:
            The path to the output file.
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"similarity_results_{self.method}_{timestamp}.json"
        
        output_path = os.path.join(OUTPUT_DIR, output_file)
        
       
        output_data = {
            "query": query,
            "method": self.method,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_results": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "paper_id": result[0],
                    "title": result[1],
                    "score": result[2]
                }
                for i, result in enumerate(results)
            ]
        }
        
      
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] Results saved to: {output_path}")
        return output_path
    
    def save_results_for_recommend(self, query: str, results: List[Tuple[str, str, float]], 
                                 output_file: str = None) -> str:
        """
        Saves the search results in JSONL format compatible with recommend.py.
        
        Args:
            query: The query text.
            results: The search results.
            output_file: The output filename (optional).
            
        Returns:
            The path to the output file.
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"similarity_for_recommend_{self.method}_{timestamp}.jsonl"
        
        output_path = os.path.join(OUTPUT_DIR, output_file)
        
        # Create JSONL format (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                paper_id, title, score = result
                
                # Create paper object compatible with recommend.py
                paper_obj = {
                    "id": paper_id,
                    "title": title,
                    "sim_score": float(score),  # Similarity score for recommend.py
                    "score": float(score),      # Alternative field name
                    "similarity": float(score), # Another alternative field name
                    "rank": i + 1,
                    "query": query,
                    "method": self.method,
                    # Add placeholder fields that recommend.py might expect
                    "citation_count": 0,       
                    "update_date": "",         
                    "abstract": "",            
                    "processed_content": ""     
                }
                
                # Write as single line JSON
                f.write(json.dumps(paper_obj, ensure_ascii=False) + "\n")
        
        print(f"[INFO] Results for recommend.py saved to: {output_path}")
        return output_path
    
    def print_results(self, query: str, results: List[Tuple[str, str, float]]):
        """
        Prints the search results to the console.
        
        Args:
            query: The query text.
            results: The search results.
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Method: {self.method.upper()}")
        print(f"Found {len(results)} similar papers:")
        print(f"{'='*80}")
        
        for i, (doc_id, title, score) in enumerate(results, 1):
            print(f"{i:2d}. [{doc_id}] {title}")
            print(f"    Similarity Score: {score:.4f}")
            print()
        
        if not results:
            print("No relevant papers found. Please try other query terms.")


def main():
    """Main function - Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Similar Paper Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python similarity_search.py --query "machine learning deep learning" --method tfidf --top_k 10
  python similarity_search.py --query "neural networks" --method lsa --top_k 5
  python similarity_search.py --query "computer vision" --method tfidf --top_k 20 --output results.jsonl
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The query text (supports free-text description)"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["tfidf", "lsa"],
        default="tfidf",
        help="Similarity calculation method: tfidf (TF-IDF + Cosine Similarity) or lsa (LSA + Cosine Similarity)"
    )
    
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=10,
        help="Number of top-k results to return (default: 10)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output filename (optional, auto-generated by default)"
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save results to a file, only display on the console"
    )
    
    
    args = parser.parse_args()
    
    try:
        # Create the searcher
        searcher = SimilaritySearch(method=args.method)
        
        # Execute the search
        results = searcher.search(args.query, args.top_k)
        
        # Display the results
        searcher.print_results(args.query, results)
        
        # Save results (unless specified not to)
        if not args.no_save:
            searcher.save_results_for_recommend(args.query, results, args.output)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())