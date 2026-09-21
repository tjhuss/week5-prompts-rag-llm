# Prompt Engineering, Local LLM, Fine-Tuning, RAG, AI Agents, and Final Capstone

Week 5 of an AI/ML/DL internship prep program. Final week --
moves from Week 4's transformer/LLM architecture theory into actually
building with LLMs: prompt engineering, running a local/API LLM,
fine-tuning concepts (LoRA/QLoRA), RAG, AI agent concepts, and the final
capstone project.

## Setup

```
pip install -r requirements.txt
```

## Project structure

| Folder | Contents |
| --- | --- |
| `day1/` | Prompt engineering notebook -- zero-shot/few-shot prompting, role prompting, system prompts, prompt templates, structured output, temperature/top-k/top-p, hallucination |
| `day2/` | Local LLM notebook -- CPU vs GPU, quantization, Ollama, Hugging Face model loading, tokenizer, pipeline, chat template |
| `day3/` | Fine-tuning dataset (JSONL) and workflow document -- prompting vs fine-tuning, LoRA, QLoRA, PEFT, hyperparameters |
| `day4/` | RAG notebook and ChromaDB index -- document loading, chunking, chunk overlap, embeddings, vector search |
| `day5/` | RAG chatbot notebook -- retriever, context injection, source citations, hallucination check, query rewriting |

## Day 1: Prompt Engineering and Structured Output

**Key learning areas:** zero-shot prompting, few-shot prompting, role
prompting, system prompts, prompt templates, structured output,
temperature, top-k, top-p, hallucination.

**Practical task:** create prompts for classification, summarization,
extraction, and Q&A -- comparing outputs across prompting strategies on
the same task, natural follow-up to Day 6's observation that a base
model doesn't follow instructions on its own.

**Deliverable:** `day1/day1_prompt_engineering.ipynb`.

Used `google/flan-t5-base` instead of Day 6's base GPT-2, since it's
instruction-tuned and can actually follow a task instead of just
continuing text. Compared prompting strategies across all four task
types: zero-shot vs. few-shot classification, plain vs. role-prompted
summarization, unstructured vs. templated extraction, and
zero-shot vs. context-grounded Q&A. The pattern that held across all
four: prompting changes how a model says something far more reliably
than it changes what it actually gets right. Context-grounded Q&A was
the one case where prompting genuinely fixed the problem, which is
basically a small preview of RAG. Also caught a real hallucination
early on, a confidently invented stock number that wasn't in the input
at all.

## Day 2: Local LLM, GPU Usage, and Hugging Face Workflow

**Key learning areas:** CPU vs GPU, VRAM, CUDA basics, quantization,
4-bit, 8-bit, Ollama, LM Studio, llama.cpp, vLLM overview, Hugging Face
model loading, tokenizer, pipeline, chat template.

**Practical task:** run a local (or API) LLM and test inference.

**Deliverable:** `day2/day2_local_llm.ipynb`.

Installed Ollama via Homebrew, pulled `llama3.2:3b`, and ran real local
inference through Ollama's local API, no API key, nothing leaving the
machine. Asked it about Tesla's current lineup and caught a real
knowledge-cutoff limitation, a confident but outdated answer. Second
half loads `TinyLlama-1.1B-Chat` directly through Hugging Face instead,
contrasting a service that runs a model for you against loading and
controlling it yourself. A plain prompt with no chat template applied
didn't get answered at all, the model just generated more questions
instead of responding, same base-model behavior as Day 6's GPT-2.
Applying the real chat template fixed that immediately. CPU vs. GPU,
VRAM, CUDA, quantization, and the local-LLM tooling landscape (Ollama,
LM Studio, llama.cpp, vLLM) are covered conceptually in the notebook,
this machine has no NVIDIA GPU.

## Day 3: Fine-Tuning Concepts and LoRA/QLoRA Workflow

**Key learning areas:** prompting vs. fine-tuning, instruction dataset,
JSONL format, chat template, LoRA, QLoRA, PEFT, learning rate, batch
size, epochs, adapter saving, overfitting.

**Practical task:** prepare a fine-tuning dataset and document the
workflow -- the curriculum scopes today's deliverable as the dataset
plus a written workflow, not a full training run.

**Deliverable:** `day3/day3_finetune_prep.ipynb`, plus the dataset
files it produces (`finetune_dataset.jsonl`,
`finetune_dataset_chat.jsonl`, `finetune_train.jsonl`,
`finetune_val.jsonl`).

Grew the dataset from 278 to 843 rows first to fix a real category
imbalance (Politics and Energy were both too thin), adding apnews.com
and oilprice.com as new sources. Final breakdown: Business 363,
Markets 205, Technology 119, Politics 56, Energy 56, Health 44. All
843 headlines converted into an Alpaca-style instruction dataset plus
a chat-template version, split 758 train / 85 validation. LoRA, QLoRA,
and PEFT are documented in the notebook: LoRA freezes the base model
and trains small adapter layers instead of every weight, QLoRA adds
4-bit quantization on top of that, PEFT is the umbrella term for the
whole family of techniques.

## Day 4: RAG Fundamentals and Vector Databases

**Key learning areas:** RAG concept, fine-tuning vs. RAG, document
loading, text extraction, chunking, chunk overlap, metadata,
embeddings, FAISS, ChromaDB, Qdrant, vector search.

**Practical task:** load documents, chunk text, create embeddings,
store in a vector database.

**Deliverable:** `day4/day4_rag_vectordb.ipynb`, plus the ChromaDB
index it builds (`day4/chroma_db/`).

Headlines are too short to chunk, so fetched full article text for 30
fool.com URLs instead. Split into 282 chunks with overlap, embedded
with `all-MiniLM-L6-v2`, stored in ChromaDB along with each chunk's
URL, title, and category as metadata. Ran two search queries, both
matched their source article by meaning instead of keywords, the
Nvidia query never says AMD or GPU but still pulled the right
AMD-vs-Nvidia article.

## Day 5: Building RAG Chatbot and AI Agent Concepts

**Key learning areas:** retriever, prompt template, context injection,
source-grounded answers, citations, hallucination reduction, hybrid
search, reranking, query rewriting, tool calling, AI agents.

**Practical task:** build and improve a RAG chatbot.

**Deliverable:** `day5/day5_rag_chatbot.ipynb`.

Built on top of Day 4's ChromaDB collection. A retriever pulls the
closest chunks for a question, a prompt template injects them as
labeled sources, and the model cites which source number(s) it used.
An off-topic question ("what's the weather in Tokyo") correctly got
"not enough information" instead of a made up answer, though the
sources list still showed three unrelated articles, retrieval always
returns its top matches regardless of relevance. Query rewriting
swapped one of three retrieved articles after rephrasing a vague
question, two stayed the same. Hybrid search, reranking, tool calling,
and AI agents are covered conceptually in the notebook.

## Week 5 roadmap

| Day | Topic | Deliverable |
| --- | --- | --- |
| 1 | Prompt Engineering and Structured Output | Prompt engineering notebook |
| 2 | Local LLM, GPU Usage, and Hugging Face Workflow | Local/API LLM demo |
| 3 | Fine-Tuning Concepts and LoRA/QLoRA Workflow | Fine-tuning dataset + workflow doc |
| 4 | RAG Fundamentals and Vector Databases | Vector database index |
| 5 | Building RAG Chatbot and AI Agent Concepts | Working RAG assistant |
| 6 | Final Capstone Presentation and Evaluation | Final report, presentation, GitHub repo, demo |

**Week 5 deliverable:** final capstone project with ML/NLP model, LLM
integration, RAG chatbot, GitHub repository, report, and presentation --
likely built on top of the existing scraped/cleaned news dataset from
this project (AI News Intelligence Assistant style: classify, summarize,
and RAG-chat over the news content already collected).
