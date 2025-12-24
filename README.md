
# An LLM-Enhanced Data Pipeline for High-Quality Scientific Paper Processing

This README provides detailed instructions for reproducing the experiments in the project **"An LLM-Enhanced Data Pipeline for High-Quality Scientific Paper Processing."** We recommend creating a Python 3.10 environment using `conda` and installing all required packages via:



```bash
pip install -r requirements.txt
```
---



Our system is designed with relatively low computational requirements. All components of the data processing pipeline can be executed efficiently on CPU-only machines. GPU acceleration is optional and mainly provides benefits for large-scale embedding generation or LLM inference.

> **Note:** The data collection stage relies on time-filtered crawling (e.g., papers from the last three months). As a result, the exact set of retrieved papers may vary depending on the execution date. Minor differences in dataset statistics and downstream results are expected. Nevertheless, the overall pipeline structure, processing logic, and experimental conclusions can be consistently reproduced.

The project is organized into two main subsystems:

1. **LLM-Enhanced Data Processing Pipeline** – Responsible for collecting, cleaning, enhancing, and evaluating scientific papers to produce a high-quality dataset.
2. **RAG System for Academic Q&A** – Enables retrieval-augmented question answering using the processed dataset and LLMs.

---

## LLM-Enhanced Data Processing Pipeline

The pipeline is structured into four stages: **Data Collection, Data Cleaning, Data Enhancement, and Evaluation & Analysis**. All code and outputs are organized in their respective directories.

### Setup

```bash
cd src
```

### 1. Data Collection

```bash
cd Data_Collection
```

* Scrape papers from arXiv:

```bash
python arxiv_scraper.py
```

* Scrape papers from Semantic Scholar:

```bash
python semantic_scholar_scraper.py
```

* Scrape papers from OpenAlex:

```bash
python openalex_scraper.py
```

* Merge all collected papers into a unified dataset:

```bash
python merge_jsonl.py
```

> For demonstration purposes, all Data Collection steps are integrated and executed sequentially in `paper_collection.ipynb`, allowing users to reproduce the workflow in a single notebook.

---

### 2. Data Cleaning

```bash
cd Data_Cleaning
```

* Perform strict deduplication of collected papers:

```bash
python strict_deduplication.py
```

* Conduct text cleaning to remove noise and normalize content:

```bash
python text_cleaning.py
```

* Filter out articles with exceptionally low citations (optional; not used in this project):

```bash
python citation_filter.py
```

* Clean and standardize `field_of_study` annotations:

```bash
python fields_of_study_clean.py
```

* Align formats across different data sources to ensure consistency:

```bash
python format_alignment.py
```

> For demonstration purposes, all Data Cleaning steps are integrated and executed sequentially in `data_cleaning.ipynb` to facilitate understanding of the complete cleaning workflow and its intermediate outputs.

---

### 3. Data Enhancement

```bash
cd Data_Enhancement
```

* Complete and refine fields of study for better semantic coverage:

```bash
python enhance_fields_of_study.py
```

* Generate keywords for data augmentation to enrich metadata:

```bash
python enhance_keywords.py
```

* Score articles based on the title and abstract to evaluate quality:

```bash
python enhance_scoring.py
```

* Generate structured research contribution summaries for each paper:

```bash
python enhance_summary.py
```

* Optionally, merge all generated JSONL files without filtering:

```bash
python build_simple_dataset.py
```

* Merge all generated JSONL files with filtering to create the final high-quality dataset:

```bash
python build_final_dataset.py
```

> For demonstration purposes, all Data Enhancement steps are integrated and executed sequentially in `data_enhancement.ipynb`, providing a clear and continuous view of the enhancement pipeline and its outputs.

---

### 4. Evaluation and Analysis

```bash
cd Data_Analysis/evaluation
```

* Compare dataset completeness and quality across different stages:

```bash
python data_quality_comparison.py
```

* Visualize dataset statistics and improvements:

```bash
python visualize_dataset_quality.py
python visualize_improvement.py
```

* Apply ruler-based multi-dimensional quality scoring:

```bash
python quality_scoring.py
python ruler_score_plot.py
```

* Evaluate multi-dimensional quality scoring using LLM-based methods:

```bash
python deepseek_scoring.py
python llm_score_plot.py
```

> For demonstration purposes, all Evaluation and Analysis steps are integrated and executed sequentially in `evaluation.ipynb`, enabling users to observe the full evaluation workflow along with all intermediate analyses and visualizations.

---

## RAG System for Academic Q&A

```bash
cd RAG
```

* Create a new conda environment with Python 3.10 and install all required dependencies:

```bash
pip install -r rag_requirements.txt
```

* All RAG-related code is contained and executed in `rag.ipynb`. To run the notebook and reproduce the entire RAG workflow:

```bash
jupyter nbconvert --execute rag.ipynb
```

> This setup allows for seamless reproduction of the retrieval-augmented question answering system using the processed scientific paper dataset.

```
```
