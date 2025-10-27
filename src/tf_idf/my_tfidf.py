"""
my_tfidf.py 
manual implementation of TfidfTransformer


"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError


from sklearn.preprocessing import normalize

class MyTfidfVectorizer:
    

    def __init__(
        self,
        *,
       
        input="content", encoding="utf-8", decode_error="strict",
        strip_accents=None, lowercase=True, preprocessor=None,
        tokenizer=None, analyzer="word", stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
        max_df=1.0, min_df=1, max_features=None,
        vocabulary=None, binary=False,
        
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        
       
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.dtype = dtype 


        self.cv_params_ = dict(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer,
            analyzer=analyzer, stop_words=stop_words,
            token_pattern=token_pattern, ngram_range=ngram_range,
            max_df=max_df, min_df=min_df, max_features=max_features,
            vocabulary=vocabulary, binary=binary,
            dtype=np.int64 
        )

        self.cv_ = None

        self.vocabulary_ = None
        self.idf_ = None
        self.feature_names_out_ = None


    def _calculate_idf(self, X_counts):


        
        # 1. calculate document frequency (df) from term frequency matrix
        df = np.asarray((X_counts > 0).sum(axis=0)).ravel()
        
        # 2. get total number of documents (n_docs)
        n_docs = X_counts.shape[0]
        
        # 3. calculate IDF
        if self.smooth_idf:
            # smooth formula: log((n_docs + 1) / (df + 1)) + 1
            n_docs_smooth = n_docs + 1
            df_smooth = df + 1
            idf = np.log(n_docs_smooth / df_smooth) + 1
        else:
            idf = np.log(n_docs / df) + 1
            

        return idf.astype(np.float64) 


    def _apply_tfidf_transform(self, X_counts):
        X_tfidf = X_counts.astype(self.dtype)

        if self.sublinear_tf:
            X_tfidf.data = 1 + np.log(X_tfidf.data)

        if self.use_idf:
            if self.idf_ is None:
                raise NotFittedError("IDF has not been computed. Call 'fit' first.")
            
            X_tfidf = X_tfidf.multiply(self.idf_).tocsr()

        if self.norm == 'l2':
            X_tfidf = normalize(X_tfidf, norm='l2', axis=1, copy=False)
        elif self.norm == 'l1':
            X_tfidf = normalize(X_tfidf, norm='l1', axis=1, copy=False)
        
        return X_tfidf


    def fit(self, raw_documents, y=None):

        self.cv_ = CountVectorizer(**self.cv_params_)
        X_counts = self.cv_.fit_transform(raw_documents)
        

        self.vocabulary_ = self.cv_.vocabulary_
        self.feature_names_out_ = self.cv_.get_feature_names_out()

        if self.use_idf:
            self.idf_ = self._calculate_idf(X_counts)
        else:
            self.idf_ = None 
        return self


    def transform(self, raw_documents):
 
        if self.cv_ is None:
            raise NotFittedError("Vectorizer has not been fitted yet. Call 'fit' first.")
            
        X_counts = self.cv_.transform(raw_documents)
        
        return self._apply_tfidf_transform(X_counts)


    def fit_transform(self, raw_documents, y=None):

        self.cv_ = CountVectorizer(**self.cv_params_)
        X_counts = self.cv_.fit_transform(raw_documents)
        
        self.vocabulary_ = self.cv_.vocabulary_
        self.feature_names_out_ = self.cv_.get_feature_names_out()
        
        if self.use_idf:
            self.idf_ = self._calculate_idf(X_counts)
        else:
            self.idf_ = None

        return self._apply_tfidf_transform(X_counts)


    def get_feature_names_out(self):

        if self.feature_names_out_ is None:
            raise NotFittedError("Vectorizer has not been fitted yet.")
        return self.feature_names_out_