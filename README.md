# Prompt Engineering, Local LLM, Fine-Tuning, RAG, AI Agents, and Final Capstone

Week 5 of a self-directed AI/ML/DL internship prep program. Final week --
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

## Day 1: Prompt Engineering and Structured Output

**Key learning areas:** zero-shot prompting, few-shot prompting, role
prompting, system prompts, prompt templates, structured output,
temperature, top-k, top-p, hallucination.

**Practical task:** create prompts for classification, summarization,
extraction, and Q&A -- comparing outputs across prompting strategies on
the same task, natural follow-up to Day 6's observation that a base
model doesn't follow instructions on its own.

**Deliverable:** Prompt engineering notebook (`day1/`).

Planned notebook outline:
1. zero-shot vs few-shot prompting on the same task, side by side
2. role/system prompting -- does framing the model's "role" change output quality
3. prompt templates for structured output (e.g. forcing JSON-style responses)
4. temperature/top-k/top-p -- sampling params and how they trade off
   determinism vs variety (direct extension of Day 6's greedy decoding
   looping problem)
5. a short section on hallucination -- what it looks like, why it
   happens, and how prompting choices can reduce (not eliminate) it
6. four small demos: classification, summarization, extraction, Q&A --
   each run with at least two different prompting strategies for comparison

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
