
<div align="center">
    <img src="./website/logo.png" alt="PaperTrail Logo" width="300"/>
    <br/>
    <h3>Intelligent Literature Exploration & Learning Plan Generator</h3>
    <p>
        <a href="https://xinhuangcs.github.io/PaperTrail/"><strong>🌐 website</strong></a>  | 
        <a href="https://github.com/xinhuangcs/PaperTrail/issues"><strong>🐛 Report Bug</strong></a>
    </p>
    <p>
        <em>Final Project for DTU 02807 Computational Tools for Data Science</em>
    </p>
</div>

---

## 📖 Table of Contents
* [1. Introduction](#1-introduction)
* [2. Features](#2-features)
* [3. Data Source](#3-data-source)
* [4. Technical Architecture ](#4-technical-architecture)
* [5. Version History & Iteration](#5-version-history--iteration)
    * [5.1 Data Preprocessing & Features](#51-data-preprocessing--features)
    * [5.2 Search Module)](#52-search-module)
    * [5.3 Recommend Module](#53-recommend-module)
    * [5.4 AI Advice Module](#54-ai-advice-module)
* [6. Usage Guide 🚀](#6-usage-guide-)
    * [6.1 User Mode](#61-user-mode-web-interface)
    * [6.2 Developer Mode: Start directly using our pre-calculated results](#62-developer-mode-start-directly-using-our-pre-calculated-results)
    * [6.3 Developer Mode: Starting with initial dataset filtering](#63-developer-mode-starting-with-initial-dataset-filtering)

---
## 1. Introduction
**PaperTrail** is an intelligent literature exploration tool designed to help researchers navigate the vast sea of computer science publications. It goes beyond being a mere search and recommendation engine by generating **structured learning plans** tailored to your specific objectives (e.g., “learning graph neural networks”).

This project integrates technologies like **TF-IDF**, **LSA (Latent Semantic Analysis)**, and **LLM (Large Language Model)** to retrieve relevant papers, rank them by learning value, and ultimately synthesize a step-by-step reading guide.

---
## 2. Features
📚 **Large-Scale Corpus**: Indexing over 700,000+ CS papers from arXiv.  
⚡  **Hybrid Search**: Explored multiple approaches including keyword matching (TF-IDF) and semantic understanding (LSA & SBERT), with LSH acceleration applied to portions of the code.  
🧠 **Smart Recommendation**: Uses weak supervision and MMR (Maximal Marginal Relevance) to rank papers by pedagogical value.  
🎯 **LLM-Powered Plans**: Generates JSON-structured study plans including reading order, key questions, and timelines using GPT-4/5.  
🔧 **Automated Workflow**: Fully integrated with GitHub Actions—request a plan via the Github pages, and get the result delivered to a GitHub Issue.  

---
## 3. Data Source
The project uses the [arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv). We also provide the [processed dataset](https://github.com/xinhuangcs/PaperTrail/releases) for direct use.
* **Original Size**: ~2.84M papers.
* **Filtering**: We filter for categories starting with `cs.` (Computer Science), `stat.` (Statistics), and `eess.` (Electrical Engineering and Systems Science), etc.
* **Add citation data**: Use the OpenAlex API to add citation data to papers.
* **Final Dataset**: Contains ~730,000 papers used for the search index.
* **Preprocessing**: Text cleaning, stop word removal, and stemming are applied before indexing.
---
## 4. Technical Architecture
The project structure follows the data processing pipeline:
```text
PaperTrail/
├── data/                   # Data storage (Artifacts like matrices, models)
├── scripts/
│   └── run_pipeline.py     # Script: Search -> Recommend -> Plan
├── src/
│   ├── preprocess_data/    # 1. Data Cleaning & Filtering
│   ├── tf_idf/             # 2. Feature Extraction (TF-IDF)
│   ├── lsa_and_clustering/ # 3. Dimensionality Reduction & Clustering
│   ├── search/             # 4. Similarity Search Engines
│   ├── recommend/          # 5. Re-ranking Logic/Recommadation Algorithm
│   └── ai_advice/          # 6. LLM Generation
├── website/                # Frontend
└── .github/workflows/      # CI/CD
````

---
## 5. Version History & Iteration 
The codebase reflects an iterative development process aimed at handling scale and improving quality.
### 5.1 Data Preprocessing & Features 
Located in `src/preprocess_data/`, `src/tf_idf/`, `src/lsa_and_clustering/`.
- **Preprocessing**:
    - `v1 (0_1_reduce_categories)`:
        - Reduce 2.8M papers to a manageable subset relevant to CS.
    - `v1 (1_0_add_incite_num)`:
        - Add citation counts (OpenAlex) as a quality signal for ranking.
    - `v1 (1_2_retry_record_of_negone)`:
        - Fix missing data from failures.
- **TF-IDF**:
    - `v1 (build_tfidf.py)`:
        - Implementation using Python libraries
    - `v2 (build_tfidf_manual.py)`:
        - Attempt to implement manually
- **Clustering**:
    - `v1 (lsa_cluster.py)`:
        - Slow convergence on large datasets.
    - `v2 (lsa_cluster_v2.py)`:
        - Switched to `MiniBatchKMeans` for faster execution.
    - `v3 (sbert_hdbscan_cluster_lite.py)`:
        - Replaced LSA with **SBERT** embeddings and **HDBSCAN** for density-based clustering, yielding more coherent topics.
### 5.2 Search Module 
- **v1 (`similarity_search.py`)**:
    - **Motivation**: Baseline retrieval.
    - **Result**: Real-time calculation of cosine similarity was too slow (O(N)).
- **v2 (`similarity_search_v2.py`)**:
    - **Motivation**: Optimize latency.
    - **Changes**: Pre-computed L2 norms (`row_l2_norms.npy`) to speed up dot products.
- **v3 (`similarity_serach_v3.py`)**:
    - **Motivation**: Contextualize results.
    - **Changes**: Integrated clustering info to return "Topics" alongside papers
- **v4 (`similarity_search_v4.py`)**:
    - **Implemented **LSH (Locality Sensitive Hashing)** with random hyperplanes.
### 5.3 Recommend Module
Located in `src/recommend/`.
- **v1 (`recommend.py`)**:
    - **Motivation**: Re-rank search results.
    - **Changes**: Simple heuristic weights (e.g., `0.3*sim + 0.4*citations`).
    - **Result**: Rigid and hard to tune for different user intents.
- **v2 (`recommend_v2.py`)**:
    - **Motivation**: Pipeline integration.
    - **Changes**: Auto-loading of latest search artifacts for automation.
- **v3 (`recommend_v3.py`)**:
    - **Motivation**: Adaptive and diverse ranking.
    - **Changes**:
        1. **Weak Supervision**: Generated pseudo-labels to train weights using Pairwise Logistic Loss.
        2. **MMR (Maximal Marginal Relevance)**: Added to penalize redundancy and ensure diversity.
        3. Dynamic weights for different views (Trending/Review/Theory) and diverse result sets.
### 5.4 AI Advice Module 
Located in `src/ai_advice/`.
- **v1 (`v1/`)**:
    - **Motivation**: Validating LLM capability.
    - **Result**: Free-form text output that was unstable for frontend rendering.
- **v2 (`v2/`)**:
    - **Motivation**: Structured, reliable output.
    - **Changes**: Implemented **JSON Schema** enforcement and retry logic.
    - **Result**: Returns strict JSON with "Timeline", "Actions", and "Reading Order" for UI display.
---
## 6. Usage Guide 🚀

### 6.1 User Mode (Web Interface)
You don't need to install any code.  
You don't need to pay for AI API fee (We have integrated OpenAI's API key into the system for your free use).
1. Go to the [PaperTrail Website](https://xinhuangcs.github.io/PaperTrail/).
2. Enter your learning objectives (e.g., “I want to learn about diffusion models”) and paper preferences.
3. Click **Generate**.
4. This will redirect you to open a **GitHub Issue**. Submit it to trigger the backend pipeline.
5. Wait ~1-3 mins for the bot to comment with your generated plan.

### 6.2 Developer Mode: Start directly using our pre-calculated results
If you have the processed data artifacts (in data/), you can run the search and generation pipeline locally. 
These data can be found in the [Github Release](https://github.com/xinhuangcs/PaperTrail/releases).
We recommend using data-action-v2, which offers better search performance but requires more storage space and memory. 
Due to Github Action server limitations, our website utilizes the lightweight version of data-action-v3.

**Prerequisites:**
- Python 3.12+
- OpenAI API Key (`export OPENAI_API_KEY='sk-...'`)

Bash
```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
# This runs: Search (v4) -> Recommend (v3) -> Standardize -> Generate Plan (v2)
python scripts/run_pipeline.py \
  --goal "Graph Neural Networks" \
  --top_k 10 \
  --method lsa_lsh \
  --mode review \
  --issue "local_test_run"
```

### 6.3 Developer Mode: Starting with initial dataset filtering
To rebuild the index from the raw arXiv dataset: (Note: Using a personal computer may take more than a day)
**Prerequisites:**
- Python 3.12+
- OpenAI API Key (`export OPENAI_API_KEY='sk-...'`)

1. **Download Data**: Place `arxiv-metadata-oai-snapshot.json` in `data/preprocess/`.
2. **Filter Categories**:
    ```
	pip install -r requirements.txt
	python src/preprocess_data/0_1_reduce_categories.py
    ```
3. **Text Cleaning**:
    ```
    python src/preprocess_data/2_data_filtering.py
    ```
4. **Build Search Index (TF-IDF & LSA)**:
    ```
    python src/tf_idf/build_tfidf_manual.py  # Generates sparse matrix
    python src/lsa_and_clustering/build_lsa.py  # Generates LSA model
    ```
5. **Export Artifacts for Search Engine**:
    ```
    python src/search/export_artifacts_v2.py  # Prepares fast-loading artifacts
    ```
6. **Run using the calculated results**:
    ```
    # This runs: Search (v4) -> Recommend (v3) -> Standardize -> Generate Plan (v2)
    python scripts/run_pipeline.py \
      --goal "Graph Neural Networks" \
      --top_k 10 \
      --method lsa_lsh \
      --mode review \
      --issue "local_test_run"
    ```

---
## 🙏 Acknowledgements
- We gratefully acknowledge **[arXiv](https://arxiv.org/)** for providing the open-access metadata.
- Citation data is powered by **[OpenAlex](https://openalex.org/)**. We verify our data enrichment process complies with their usage policy (Polite Pool).
## 📄 License
- **Code**: Licensed under the **Apache License 2.0**.
- **Data**: The enriched dataset is released under **CC0 1.0 Universal**, consistent with its sources (arXiv & OpenAlex).
