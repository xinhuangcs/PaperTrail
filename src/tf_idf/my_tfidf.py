import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import re

class MyTfidfVectorizer:
    """
    
    A from-scratch implementation of TF-IDF Vectorizer,
    now updated to support max_df, min_df, max_features, ngram_range, sublinear_tf, norm, and dtype.
    """
    
    def __init__(self, 
                 lowercase=True, 
                 stop_words=None, 
                 ngram_range=(1, 1),
                 max_df=1.0, 
                 min_df=1, 
                 max_features=None,
                 norm='l2', 
                 sublinear_tf=False,
                 dtype=np.float64
                ):
        """
        Initializes the vectorizer and stores all parameters.
        """
        self.vocabulary_ = None
        self.idf_ = None
        self.feature_names_ = None 

        self.lowercase = lowercase
        self.stop_words = stop_words 
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.dtype = dtype
        
       
        self.token_pattern_ = re.compile(r"(?u)\b\w\w+\b")


    def _tokenize(self, doc):
        """
        A helper function to turn a single doc into tokens and generate n-grams.
        """
        
        if self.lowercase:
            doc = doc.lower()
        
        # (scikit-learn 默认: r"(?u)\b\w\w+\b")
        words = self.token_pattern_.findall(doc)
        
        min_n, max_n = self.ngram_range
        if min_n < 1:
            raise ValueError("ngram_range的最小值必须 >= 1")
            
        ngrams = []
        
        for n in range(min_n, max_n + 1):

            for i in range(len(words) - n + 1):

                ngrams.append(" ".join(words[i:i+n]))
                
        return ngrams


    def fit(self, raw_documents):
        """
        Learn vocabulary and IDF from a collection of raw documents.
        Now supports min_df, max_df, and max_features.
        """
        
        df = defaultdict(int)
        total_tf = defaultdict(int)
        n_docs = 0

        for doc in raw_documents:
            n_docs += 1
            tokens = self._tokenize(doc)
            
            for token in tokens:
                total_tf[token] += 1
            unique_tokens_in_doc = set(tokens)
            for token in unique_tokens_in_doc:
                df[token] += 1

        if n_docs == 0:
            raise ValueError("fit被调用时传入了空语料库。")


        if isinstance(self.min_df, int):
            min_doc_count = self.min_df
        else: # float
            min_doc_count = self.min_df * n_docs

        if isinstance(self.max_df, int):
            max_doc_count = self.max_df
        else: # float
            max_doc_count = self.max_df * n_docs

        pruned_vocab_set = set()
        for term, doc_count in df.items():
            if min_doc_count <= doc_count <= max_doc_count:
                pruned_vocab_set.add(term)
        
        
        if self.max_features is not None and len(pruned_vocab_set) > self.max_features:

            filtered_tf = {term: count for term, count in total_tf.items() if term in pruned_vocab_set}
            

            sorted_terms = sorted(
                filtered_tf.items(), 
                key=lambda item: (-item[1], item[0])
            )
            
            final_vocab_set = {term for term, count in sorted_terms[:self.max_features]}
        else:
            final_vocab_set = pruned_vocab_set

        sorted_vocab_list = sorted(list(final_vocab_set))
        
        self.vocabulary_ = {term: i for i, term in enumerate(sorted_vocab_list)}
        self.feature_names_ = sorted_vocab_list 

        df_array = np.array([df[term] for term in sorted_vocab_list])
        
        self.idf_ = np.log((1 + n_docs) / (1 + df_array)) + 1
        
        return self

    def fit_transform(self, raw_documents):
        """
        Learn vocabulary and IDF, then return document-term matrix.
        This is a convenience method that combines fit() and transform().
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
       
        if self.vocabulary_ is None or self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted yet. Call 'fit' first.")

        rows, cols, data = [], [], []
        n_docs_transform = 0
        
        for i, doc in enumerate(raw_documents):
            n_docs_transform += 1
            tokens = self._tokenize(doc)
            
            term_counts = defaultdict(int)
            for token in tokens:
                if token in self.vocabulary_:
                    term_counts[token] += 1

            for term, count in term_counts.items():
                rows.append(i)
                cols.append(self.vocabulary_[term])
                data.append(count)
        
        n_features = len(self.vocabulary_)
        
        tf_matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(n_docs_transform, n_features),
            dtype=self.dtype
        )

        if self.sublinear_tf:
            tf_matrix.data = 1 + np.log(tf_matrix.data)

        idf_diag = sp.diags(self.idf_, dtype=self.dtype)
        tfidf_matrix = tf_matrix @ idf_diag

        if self.norm == 'l2':
            tfidf_matrix = tfidf_matrix.tocsr()
            
            norms = np.sqrt(tfidf_matrix.power(2).sum(axis=1))
            
            norms[norms == 0] = 1e-9
            
            inv_norms = sp.diags(1.0 / np.array(norms).flatten(), dtype=self.dtype)
            
            normalized_tfidf_matrix = inv_norms @ tfidf_matrix
            
            return normalized_tfidf_matrix
        
        elif self.norm is None:
            return tfidf_matrix.tocsr()
            
        else:
            raise ValueError(f"norm='{self.norm}' is not supported. Only 'l2' or None.")

    def get_feature_names_out(self):
        """
        Returns a list of feature names (tokens) sorted by index.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Vectorizer has not been fitted yet.")
        return self.feature_names_

