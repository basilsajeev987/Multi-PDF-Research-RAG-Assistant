# Multi-PDF-Research-RAG-Assistant
# 📚 Multi-PDF Research RAG Assistant

An advanced **Retrieval-Augmented Generation (RAG)** system for asking questions across a large collection of PDF documents.

The system is designed for **research-scale multi-document question answering**, allowing users to search across **228+ PDF documents**, retrieve relevant evidence, rerank the retrieved information, and generate grounded answers using a locally hosted **Qwen 2.5 14B Instruct** language model.

The system provides document-level and chunk-level retrieval, semantic and lexical search, reciprocal rank fusion, cross-encoder reranking, comprehensive multi-document retrieval, Map-Reduce synthesis, conversational query rewriting, and a Gradio-based user interface.

---

## 🚀 Key Features

* 📚 Search across **228+ PDF documents**
* 🔎 Semantic vector search using **BGE-M3**
* 🔤 Keyword search using **BM25**
* ⚡ GPU-accelerated **FAISS** vector search
* 🔀 **Reciprocal Rank Fusion (RRF)** for hybrid retrieval
* 🎯 Cross-encoder reranking using **BGE Reranker v2 M3**
* 🧠 Local **Qwen 2.5 14B Instruct** LLM
* 🎮 NVIDIA **A100 40 GB GPU**
* 🗂️ Document-level retrieval
* 📄 Chunk-level retrieval
* 🔬 Exhaustive multi-PDF retrieval
* 📖 Comprehensive research mode
* 🧩 Map-Reduce document synthesis
* 💬 Conversational RAG
* 🔄 Follow-up question/query rewriting
* 📌 PDF and page-level citations
* 🛡️ Grounded generation to reduce hallucinations
* 🧹 Duplicate/overlapping evidence control
* 📊 Retrieval statistics
* 🖥️ Gradio web interface
* 💾 Google Drive document storage
* 📦 Parquet-based document/chunk storage
* 🚀 CUDA and TensorFloat-32 optimization
* 🔢 Token-aware context management
* 🔍 Separate Normal and Comprehensive retrieval modes

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │     PDF Collection   │
                         │      228+ PDFs        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   PyMuPDF / Fitz     │
                         │    PDF Extraction    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Text Cleaning &    │
                         │      Chunking        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       BGE-M3         │
                         │      Embeddings      │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
              ┌──────────────┐             ┌──────────────┐
              │    FAISS     │             │     BM25     │
              │ Vector Search│             │Keyword Search│
              └──────┬───────┘             └──────┬───────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Reciprocal Rank     │
                         │       Fusion         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Candidate Evidence │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ BGE Reranker v2 M3   │
                         │   Cross Encoder      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         │                      │
                         ▼                      ▼
                  Normal RAG              Comprehensive RAG
                         │                      │
                         │               ┌──────▼──────┐
                         │               │ Map Stage   │
                         │               │ Per-PDF     │
                         │               └──────┬──────┘
                         │                      │
                         │               ┌──────▼──────┐
                         │               │ Reduce      │
                         │               │ Synthesis   │
                         │               └──────┬──────┘
                         │                      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Qwen 2.5 14B       │
                         │      Instruct        │
                         │    Local LLM         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Final Answer     │
                         │  + PDF/Page Sources  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Gradio UI       │
                         └──────────────────────┘
```

---

# 🎮 Hardware

## GPU

The project was developed and tested using:

| Component  | Specification                           |
| ---------- | --------------------------------------- |
| GPU        | **NVIDIA A100**                         |
| GPU Memory | **40 GB VRAM**                          |
| CUDA       | CUDA-enabled PyTorch environment        |
| CPU RAM    | ~83.5 GB available in Colab environment |
| Platform   | Google Colab                            |
| Storage    | Google Drive                            |

The A100 GPU is used for the computationally intensive components of the RAG pipeline.

### GPU workloads

The GPU is used for:

* BGE-M3 embedding generation
* FAISS GPU vector search
* BGE cross-encoder reranking
* Qwen 2.5 14B inference

---

# ☁️ Google Colab Environment

The project was developed using a paid Google Colab environment with access to an NVIDIA A100 GPU.

Example environment:

```text
GPU:
NVIDIA A100 40GB

System RAM:
~83.5 GB

GPU VRAM:
40 GB
```

The project takes advantage of GPU acceleration rather than relying entirely on CPU processing.

---

# 📂 Project Storage

PDF documents are stored in Google Drive.

```text
/content/drive/MyDrive/Rag_Project/
│
├── Documents/
│   ├── paper_01.pdf
│   ├── paper_02.pdf
│   ├── paper_03.pdf
│   ├── ...
│   └── paper_228.pdf
│
├── index/
│   ├── pages.parquet
│   ├── chunks.parquet
│   ├── embeddings.npy
│   ├── documents.parquet
│   └── document_embeddings.npy
│
├── models/
│
└── outputs/
```

---

# 📄 PDF Processing

PDF extraction is performed using **PyMuPDF (`fitz`)**.

The system processes PDFs page by page and preserves:

* Document name
* PDF path
* Page number
* Extracted text

Example metadata:

```python
{
    "document": "research_paper.pdf",
    "path": "/Documents/research_paper.pdf",
    "page": 12,
    "text": "Extracted document text..."
}
```

This metadata is retained throughout the retrieval pipeline so that final answers can provide page-level citations.

---

# 🧹 Text Cleaning

Extracted PDF text is cleaned before indexing.

The cleaning process handles:

* Null characters
* Excess whitespace
* Unnecessary line breaks
* Basic text normalization

Example:

```python
def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

# ✂️ Text Chunking

The extracted document text is divided into smaller chunks for retrieval.

The initial implementation uses overlapping chunks.

Current configuration:

```text
Chunk size: ~600 words
Overlap: ~100 words
```

Each chunk maintains:

* Chunk ID
* Document name
* Page number
* Chunk number
* Original PDF path
* Text

Example:

```text
Document: paper.pdf
Page: 17
Chunk: 3
```

This allows the system to locate the original evidence after retrieval.

---

# 🧠 Embedding Model

## BGE-M3

The project uses:

```text
BAAI/bge-m3
```

BGE-M3 converts document chunks and queries into dense vector representations.

The embeddings are normalized:

```python
normalize_embeddings=True
```

This allows cosine similarity to be calculated efficiently using inner-product FAISS search.

---

# ⚡ Embedding Performance

The BGE-M3 embedding batch size was benchmarked on the A100.

Benchmark on 1,000 chunks:

| Batch Size |         Time |            Throughput |
| ---------: | -----------: | --------------------: |
|         32 |     9.71 sec |     102.94 chunks/sec |
|     **64** | **9.46 sec** | **105.74 chunks/sec** |
|        128 |    10.55 sec |      94.81 chunks/sec |
|        256 |    11.37 sec |      87.98 chunks/sec |

The selected batch size is:

```python
batch_size=64
```

because it provided the highest measured throughput in the benchmark.

---

# 🔎 Vector Database / Search

## FAISS

The project uses FAISS for high-speed vector similarity search.

Index type:

```text
IndexFlatIP
```

The embeddings are normalized, so inner-product similarity corresponds to cosine similarity.

The FAISS index is transferred to the NVIDIA A100 GPU.

Architecture:

```text
BGE-M3 embeddings
        ↓
Normalized vectors
        ↓
FAISS IndexFlatIP
        ↓
GPU
        ↓
Fast similarity search
```

---

# 🔤 BM25 Keyword Retrieval

Dense semantic retrieval is combined with lexical retrieval.

The project uses:

```text
rank_bm25
```

BM25 is useful for:

* Exact terminology
* Names
* Technical terms
* Numbers
* Acronyms
* Specific phrases
* Rare keywords

Example:

```text
User Query
     │
     ├──────────────► BGE-M3 / FAISS
     │
     └──────────────► BM25
```

This hybrid approach helps retrieve both semantically similar and keyword-specific evidence.

---

# 🔀 Reciprocal Rank Fusion

Dense and BM25 results are combined using **Reciprocal Rank Fusion (RRF)**.

The RRF score is calculated as:

```text
RRF(d) = Σ 1 / (k + rank)
```

where:

```text
k = 60
```

The system combines:

```text
FAISS ranking
      +
BM25 ranking
      ↓
RRF
      ↓
Hybrid ranking
```

---

# 🎯 Cross-Encoder Reranking

After hybrid retrieval, candidates are reranked using:

```text
BAAI/bge-reranker-v2-m3
```

The reranker receives pairs:

```text
[query, document_chunk]
```

and produces a relevance score.

Pipeline:

```text
FAISS
  +
BM25
  ↓
RRF
  ↓
Candidate chunks
  ↓
BGE Reranker v2 M3
  ↓
Final relevance ranking
```

The reranker is also executed on the NVIDIA A100.

---

# 📚 Multi-PDF Retrieval

A major feature of the project is **multi-document retrieval**.

Instead of treating the collection as a single document, the system maintains document-level metadata.

For example:

```text
228 PDFs
   │
   ├── Paper 1
   ├── Paper 2
   ├── Paper 3
   ├── ...
   └── Paper 228
```

When a question is asked, the system can identify evidence across multiple PDFs.

---

# 🌎 Comprehensive Retrieval Mode

The project provides two retrieval modes.

## Normal Mode

Normal mode is optimized for focused questions.

Example:

```text
"What is the accuracy of the Random Forest model?"
```

Pipeline:

```text
Query
 ↓
FAISS
 ↓
BM25
 ↓
RRF
 ↓
Reranker
 ↓
Top relevant chunks
 ↓
Qwen
```

---

## Comprehensive Mode

Comprehensive mode is designed for questions such as:

```text
"Give me all information about galaxy morphology
across the research papers."
```

Instead of only taking a small global top-K result, the system evaluates the document collection more broadly.

Conceptually:

```text
228 PDFs
    ↓
Search all documents
    ↓
Search evidence across documents
    ↓
Rerank evidence
    ↓
Group evidence by PDF
    ↓
Summarize relevant documents
    ↓
Global synthesis
```

This reduces the chance that a single document dominates the final answer.

---

# 🔬 Exhaustive Multi-PDF Retrieval

The comprehensive retrieval system evaluates chunks across the document collection.

For each PDF, candidate evidence can be selected using:

* Semantic similarity
* BM25 relevance
* Hybrid relevance

The candidate evidence is then reranked.

This provides document coverage such as:

```text
228 PDFs searched
       ↓
45 PDFs contain relevant evidence
       ↓
Relevant chunks extracted
       ↓
Reranker
       ↓
Evidence synthesis
```

The goal is not to force all 228 PDFs into the answer.

Instead:

> **All PDFs are searched, while only PDFs containing relevant evidence contribute to the answer.**

---

# 🧩 Map-Reduce RAG

For very large multi-document questions, directly passing all retrieved text to the LLM would exceed practical context limits.

The project therefore supports a Map-Reduce approach.

## Map Stage

Each relevant PDF is analyzed separately.

```text
PDF 1 → Qwen → Summary 1
PDF 2 → Qwen → Summary 2
PDF 3 → Qwen → Summary 3
...
PDF N → Qwen → Summary N
```

Each summary preserves source information and page references.

---

## Reduce Stage

The individual document summaries are combined.

```text
Summary 1
Summary 2
Summary 3
...
Summary N
       ↓
     Qwen
       ↓
Final comprehensive synthesis
```

This allows the system to work with significantly more source documents than would fit into a single raw context window.

---

# 🧠 LLM

## Qwen 2.5 14B Instruct

The generation model is:

```text
Qwen/Qwen2.5-14B-Instruct
```

The model is loaded locally using Hugging Face Transformers.

---

# 📦 Quantization

Qwen 14B is loaded using 4-bit quantization.

Configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

This significantly reduces GPU memory requirements while allowing the 14B model to run on the 40 GB A100.

---

# ⚙️ Qwen Compute Configuration

The model uses:

```text
4-bit NF4 quantization
BF16 compute
Double quantization
CUDA
```

Model loading:

```python
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
```

---

# 🛡️ Grounded Generation

The system uses a strict RAG prompt to reduce hallucination.

The model is instructed to:

* Use supplied document evidence
* Avoid unsupported facts
* Avoid relying on pretrained knowledge
* Cite relevant documents
* Cite page numbers
* Avoid fabricating page numbers
* Identify disagreements between sources
* State when evidence is insufficient

Example citation:

```text
[research_paper.pdf, Page 17]
```

---

# 📌 Citation System

Each retrieved chunk contains:

```text
Document
Page
Chunk
Text
```

The model is instructed to preserve these references.

Example final response:

```text
Galaxy morphology can be classified using several
structural characteristics [paper_01.pdf, Page 12].

Another study reports a different classification approach
based on machine-learning features [paper_04.pdf, Page 27].
```

This allows the user to trace information back to the original PDF.

---

# 💬 Conversational RAG

The system also supports follow-up questions.

Example:

```text
User:
What methods were used for galaxy classification?

AI:
Several machine-learning methods were discussed...

User:
Which one performed better?

AI:
...
```

The second question is rewritten into a standalone retrieval query.

---

# 🔄 Query Rewriting

The query rewriting system uses the conversation history to resolve references such as:

```text
it
they
this method
the first one
the second model
that approach
its accuracy
```

Example:

```text
Previous:
"What methods were used for galaxy classification?"

Follow-up:
"Which one performed better?"

                ↓

Rewritten retrieval query:

"Which galaxy classification method
performed better?"
```

The rewritten query is then sent to the document retrieval system.

---

# 🖥️ Gradio Interface

The project includes a Gradio web interface.

The interface provides:

* Chat interface
* Retrieval mode selection
* Normal mode
* Comprehensive mode
* Answer panel
* Source panel
* Retrieval details
* Conversation history
* Clear button

Conceptually:

```text
┌───────────────────────────────────────────────┐
│       📚 Multi-PDF Research Assistant         │
├───────────────────────────────────────────────┤
│ Retrieval Mode:                              │
│ ○ Normal       ● Comprehensive               │
│                                               │
│ ┌───────────────────────────────────────────┐ │
│ │ Conversation                              │ │
│ │                                           │ │
│ │ User: What does the research say about X? │ │
│ │ AI: ...                                   │ │
│ └───────────────────────────────────────────┘ │
│                                               │
│ Ask your documents...                         │
│                                               │
│ [ 🔎 Search & Answer ] [ 🗑️ Clear ]          │
│                                               │
│ 🧠 Answer                                     │
│                                               │
│ 📚 Documents Used                             │
│                                               │
│ 🔍 Retrieval Details                          │
└───────────────────────────────────────────────┘
```

---

# 📊 Retrieval Statistics

The interface can display:

```text
PDFs searched
PDFs contributing evidence
PDFs summarized
Evidence chunks
Retrieved pages
Reranker scores
```

For example:

```text
PDFs searched: 228

PDFs contributing evidence: 31

PDFs summarized: 31

Evidence chunks: 87
```

---

# 🧹 Duplicate Evidence Handling

The system prevents one document or repeated chunks from unnecessarily dominating the context.

Evidence can be limited using:

```python
max_chunks_per_pdf
```

and:

```python
max_total_evidence
```

For example:

```python
max_chunks_per_pdf=4
max_total_evidence=100
```

This provides a balance between:

* Evidence coverage
* Document diversity
* Context size
* Generation quality

---

# 🔢 Token-Aware Context Management

Because LLM context is measured in tokens rather than characters, the project includes token-aware context management.

The Qwen tokenizer is used to:

* Count tokens
* Limit evidence
* Trim oversized evidence packages
* Control Map-stage input
* Control Reduce-stage input

Example:

```python
count_tokens(text)
```

This helps prevent oversized prompts.

---

# 🚀 CUDA Optimization

The project enables GPU optimizations:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

These settings are intended to improve performance on supported NVIDIA hardware such as the A100.

---

# 🧰 Technologies Used

## Programming Language

* Python

## Machine Learning

* PyTorch
* Hugging Face Transformers
* Sentence Transformers

## Embedding

* BAAI BGE-M3

## Reranking

* BAAI BGE Reranker v2 M3

## Large Language Model

* Qwen 2.5 14B Instruct

## Quantization

* BitsAndBytes
* 4-bit NF4
* BF16 computation

## Vector Search

* FAISS
* FAISS GPU

## Lexical Search

* BM25
* rank_bm25

## PDF Processing

* PyMuPDF / Fitz

## Data Storage

* Pandas
* Parquet
* NumPy

## Interface

* Gradio

## Hardware

* NVIDIA A100 40 GB

## Cloud Environment

* Google Colab
* Google Drive

---

# 📦 Main Python Packages

Typical dependencies include:

```text
torch
transformers
accelerate
bitsandbytes
sentence-transformers
faiss-gpu-cu12
rank-bm25
pymupdf
pandas
numpy
gradio
pyarrow
```

---

# 🔧 Installation

Install the main dependencies:

```bash
pip install -q \
    transformers \
    accelerate \
    bitsandbytes \
    sentence-transformers \
    rank-bm25 \
    pymupdf \
    pandas \
    numpy \
    gradio \
    pyarrow
```

For the GPU FAISS installation:

```bash
pip install -q faiss-gpu-cu12
```

---

# 📁 Directory Configuration

Set the project paths:

```python
PROJECT_DIR = "/content/drive/MyDrive/Rag_Project"

PDF_DIR = f"{PROJECT_DIR}/Documents"

INDEX_DIR = f"{PROJECT_DIR}/index"

MODEL_DIR = f"{PROJECT_DIR}/models"

OUTPUT_DIR = f"{PROJECT_DIR}/outputs"
```

Mount Google Drive:

```python
from google.colab import drive

drive.mount(
    "/content/drive"
)
```

---

# ▶️ Running the System

## 1. Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

## 2. Set the document directory

```python
PDF_DIR = (
    "/content/drive/MyDrive/"
    "Rag_Project/Documents"
)
```

## 3. Extract PDFs

Use PyMuPDF to extract page-level text.

## 4. Create chunks

Generate chunk metadata and save:

```text
chunks.parquet
```

## 5. Generate BGE-M3 embeddings

Recommended tested batch size:

```python
batch_size=64
```

Save:

```text
embeddings.npy
```

## 6. Build FAISS

Create the vector index and transfer it to GPU.

## 7. Build BM25

Create the lexical retrieval index.

## 8. Load the reranker

```text
BAAI/bge-reranker-v2-m3
```

## 9. Load Qwen

```text
Qwen/Qwen2.5-14B-Instruct
```

using 4-bit NF4 quantization.

## 10. Launch Gradio

```python
demo.launch(
    share=True,
    debug=True
)
```

---

# 🔎 Example Questions

### Basic question

```text
What is galaxy morphology?
```

### Technical question

```text
What machine-learning methods are used for galaxy classification?
```

### Cross-document question

```text
Compare the galaxy classification methods discussed
across the research papers.
```

### Comprehensive question

```text
Give me all information about galaxy morphology
across the relevant research papers.
```

### Follow-up question

```text
What was the best performing method?
```

### Evidence question

```text
Which papers report results for Random Forest?
```

### Source question

```text
Which document discusses this method on page 20?
```

---

# 🔬 RAG Pipeline Details

The retrieval pipeline can be summarized as:

```text
                    USER QUERY
                        │
                        ▼
                Query Rewriting
                 (if follow-up)
                        │
                        ▼
             ┌────────────────────┐
             │   Semantic Search  │
             │      BGE-M3        │
             └─────────┬──────────┘
                       │
             ┌─────────▼──────────┐
             │      FAISS GPU     │
             └─────────┬──────────┘
                       │
                       ├──────────────┐
                       │              │
                       ▼              ▼
                  Dense Rank      BM25 Rank
                       │              │
                       └──────┬───────┘
                              ▼
                       RRF Fusion
                              │
                              ▼
                        Candidates
                              │
                              ▼
                     Cross Encoder
                         Reranker
                              │
                              ▼
                    Evidence Selection
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
                  Normal          Comprehensive
                    │                   │
                    │             Group by PDF
                    │                   │
                    │              Map Stage
                    │                   │
                    │             Reduce Stage
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         Qwen 14B
                              │
                              ▼
                       Grounded Answer
                              │
                              ▼
                    PDF/Page Citations
```

---

# 🧠 Why Hybrid Retrieval?

Using only vector search can miss exact technical terminology.

Using only BM25 can miss semantically related language.

The project therefore combines both:

```text
Semantic Search
      +
Keyword Search
      ↓
Hybrid Retrieval
```

This is particularly useful for technical and scientific documents containing:

* Acronyms
* Equations
* Model names
* Dataset names
* Scientific terminology
* Exact numerical values
* Domain-specific vocabulary

---

# 🎯 Why Reranking?

Initial retrieval is designed for high recall.

The reranker then improves precision.

```text
Large candidate pool
        ↓
Cross Encoder
        ↓
More relevant evidence
```

This two-stage retrieval approach is more suitable for a large document collection than relying entirely on a single retrieval method.

---

# 📚 Why Map-Reduce?

A collection containing hundreds of PDFs can contain far more relevant text than can fit into a single LLM prompt.

Map-Reduce addresses this:

```text
Many PDFs
    ↓
Individual document analysis
    ↓
Compact summaries
    ↓
Global synthesis
```

This allows the system to answer broad research questions while keeping the final LLM context manageable.

---

# 🛡️ Hallucination Control

The system uses several mechanisms to reduce unsupported answers:

### 1. Retrieval grounding

The model receives retrieved document evidence.

### 2. Strict prompting

The model is instructed not to use unsupported information.

### 3. Source citations

Factual claims are expected to include:

```text
[Document, Page X]
```

### 4. Insufficient evidence response

If the available evidence cannot establish an answer, the model is instructed to say:

```text
I couldn't find enough information in the provided documents.
```

### 5. Deterministic generation

The current generation configuration uses:

```python
do_sample=False
```

for more deterministic responses.

---

# ⚠️ Current Limitations

The current pipeline is primarily **text-based PDF RAG**.

Information may be missed when it exists primarily in:

* Scanned PDF pages
* Images
* Complex tables
* Charts
* Diagrams
* Figures
* Mathematical notation that extracts poorly

A PDF with no usable text layer requires OCR before it can be reliably searched.

---

# 🔮 Future Improvements

Potential next improvements include:

## OCR

Add OCR for scanned PDFs.

```text
Scanned PDF
     ↓
OCR
     ↓
Text
     ↓
RAG
```

## Table Extraction

Extract tables separately and preserve their structure.

## Figure Understanding

Add multimodal processing for:

* Figures
* Charts
* Diagrams
* Captions

## Structure-Aware Chunking

Improve the current chunking strategy by preserving:

* Headings
* Sections
* Paragraph boundaries
* Tables
* Academic document structure

## Hybrid Multimodal RAG

Future architecture:

```text
PDF
 │
 ├── Text
 │
 ├── Tables
 │
 ├── Images
 │
 └── Figures
       │
       ▼
Multimodal Retrieval
       │
       ▼
Multimodal LLM
```

## Better Context Compression

Use relevance-based compression before sending evidence to Qwen.

## Evaluation Framework

Add automated evaluation metrics such as:

* Recall@K
* Precision@K
* MRR
* NDCG
* Faithfulness
* Answer relevance
* Citation accuracy
* Context recall

## Retrieval Benchmarking

Evaluate retrieval quality using a curated question-answer dataset.

---

# 📈 Scalability

The system is designed around a modular architecture.

The number of documents can increase without fundamentally changing the pipeline:

```text
228 PDFs
   ↓
500 PDFs
   ↓
1,000 PDFs
   ↓
10,000+ PDFs
```

At larger scales, the FAISS index can eventually be replaced or supplemented with a distributed/vector database architecture.

Potential future technologies include:

* FAISS IVF
* FAISS HNSW
* FAISS PQ
* Qdrant
* Milvus
* Weaviate
* Elasticsearch
* OpenSearch

---

# 🗃️ Saved Index Files

The system stores processed data so PDFs do not need to be reprocessed every time the notebook restarts.

Example:

```text
index/
│
├── pages.parquet
├── chunks.parquet
├── embeddings.npy
├── documents.parquet
└── document_embeddings.npy
```

### `pages.parquet`

Contains page-level extracted text.

### `chunks.parquet`

Contains chunk-level retrieval data.

### `embeddings.npy`

Contains BGE-M3 chunk embeddings.

### `documents.parquet`

Contains document-level metadata.

### `document_embeddings.npy`

Contains aggregated document embeddings.

---

# 🔄 Runtime Recovery

Because Google Colab sessions can restart, the project saves the processed indexes to Google Drive.

After a runtime restart, the system can reload:

```text
chunks.parquet
embeddings.npy
documents.parquet
document_embeddings.npy
```

and rebuild the GPU FAISS indexes without reprocessing every PDF.

---

# 🔐 API Keys

The current architecture is designed to run the retrieval and LLM components locally within the Colab environment.

No external LLM API is required for Qwen inference.

If external services are added later, API keys should **never be committed to GitHub**.

Use environment variables or a secure secret manager instead.

Example:

```python
import os

API_KEY = os.environ.get(
    "API_KEY"
)
```

Never place secrets directly inside source code.

---

# 🧪 Example End-to-End Workflow

```text
1. Upload PDFs to Google Drive
          ↓
2. Mount Google Drive
          ↓
3. Extract PDF text
          ↓
4. Clean text
          ↓
5. Create chunks
          ↓
6. Generate BGE-M3 embeddings
          ↓
7. Build FAISS GPU index
          ↓
8. Build BM25 index
          ↓
9. Load BGE reranker
          ↓
10. Load Qwen 14B
          ↓
11. Launch Gradio
          ↓
12. Ask question
          ↓
13. Search documents
          ↓
14. Hybrid retrieval
          ↓
15. Reranking
          ↓
16. Evidence selection
          ↓
17. Map-Reduce if comprehensive
          ↓
18. Qwen synthesis
          ↓
19. Grounded answer
          ↓
20. PDF + page citations
```

---

# 📌 Project Goals

The primary goal of this project is to create a research assistant capable of answering questions over a large collection of technical and scientific PDFs while maintaining source traceability.

The system focuses on:

* High retrieval coverage
* Multi-document reasoning
* Evidence-grounded generation
* Source attribution
* GPU acceleration
* Scalable document processing
* Conversational research
* Comprehensive document synthesis

---

# 🏆 Current Configuration

```text
Documents:
228+ PDFs

PDF Processing:
PyMuPDF

Chunking:
~600 words
~100 word overlap

Embedding:
BAAI/bge-m3

Embedding Batch:
64

Vector Search:
FAISS IndexFlatIP

Vector Search:
GPU accelerated

Keyword Search:
BM25

Hybrid Retrieval:
RRF

Reranker:
BAAI/bge-reranker-v2-m3

Reranker Device:
NVIDIA A100

LLM:
Qwen/Qwen2.5-14B-Instruct

LLM Quantization:
4-bit NF4

LLM Compute:
BF16

GPU:
NVIDIA A100 40 GB

Environment:
Google Colab

Storage:
Google Drive

Interface:
Gradio

RAG Modes:
Normal + Comprehensive

Advanced Retrieval:
Exhaustive Multi-PDF Retrieval

Long-Document Strategy:
Map-Reduce

Conversation:
Query Rewriting / Conversational RAG

Citation:
Document + Page
```

---

# 📜 License

Add the appropriate license for your project and verify the licenses of all third-party models and libraries before distributing the system.

---

# 👨‍💻 Author

**Basil Sajeev**

MSc Computer Science
Ulster University, Manchester

---

# ⭐ Summary

This project implements a GPU-accelerated, multi-stage RAG architecture for research over a large collection of PDFs.

The system combines:

```text
PyMuPDF
    +
BGE-M3
    +
FAISS GPU
    +
BM25
    +
RRF
    +
BGE Reranker
    +
Qwen 2.5 14B
    +
Map-Reduce
    +
Conversational Query Rewriting
    +
Gradio
```

running on:

```text
NVIDIA A100 40 GB
```

The resulting system is designed to search across **228+ PDF documents**, retrieve relevant evidence from multiple documents, synthesize information across sources, and provide grounded answers with **document and page-level citations**.
