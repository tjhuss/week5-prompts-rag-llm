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

## Day 1: Prompt Engineering and Structured Output

**Key learning areas:** zero-shot prompting, few-shot prompting, role
prompting, system prompts, prompt templates, structured output,
temperature, top-k, top-p, hallucination.

**Practical task:** create prompts for classification, summarization,
extraction, and Q&A -- comparing outputs across prompting strategies on
the same task, natural follow-up to Day 6's observation that a base
model doesn't follow instructions on its own.

**Deliverable:** `day1/day1_prompt_engineering.ipynb`.

Used `google/flan-t5-base` instead of Day 6's base GPT-2 -- it's
instruction-tuned, so it can actually follow a task instead of just
continuing text, which every comparison below depends on.

The pattern that showed up across all four task types: prompting
reliably changes *how* a model says something (tone, format attempt,
style) far more than it changes *what* it actually gets right.

- **Zero-shot vs. few-shot classification** -- an easy headline (Nvidia
  chip news) got "Technology" both ways, proving nothing. Swapped to a
  genuinely ambiguous one (a Tesla stock move caused by a tech product)
  and both zero-shot and few-shot landed on the generic "Business"
  catch-all instead of committing to Markets or Technology -- the exact
  same overlap Week 2 Day 4's clustering already found in this dataset.
  Few-shot examples didn't sharpen the answer at all.
- **Role prompting for summarization** -- framing the model as "a
  financial editor writing for busy investors" did shift tone (terser,
  more headline-like), but neither summary mentioned the 4% after-hours
  stock move, arguably the most investor-relevant fact in the article.
  Style changed, priority didn't.
- **Prompt templates for structured extraction** -- describing an exact
  output format ("Company: / Change:") didn't work, and neither did
  adding one worked example on top of it. Both attempts either dropped
  half the requested info or ignored the format completely. A genuine
  negative result: structured field extraction is a real limitation of
  a model this size, not just a prompt-wording problem.
- **Temperature / top-k / top-p** -- greedy decoding gave an identical
  output on repeat runs (fully deterministic). Low temperature (0.3)
  mostly matched greedy but not always. High temperature (1.5) produced
  genuinely incoherent text on one run -- real evidence that too much
  randomness breaks output, it doesn't just make it "creative." top_p=0.5
  stayed readable across runs while still varying.
- **Q&A, zero-shot vs. context-grounded** -- without context, the model
  hallucinated a garbled non-answer ("a computer chip chip"). Given the
  source text directly in the prompt, it correctly answered "a new AI
  chip," pulled straight from the given content. This is the cleanest
  result in the notebook, and it's basically a tiny preview of RAG,
  which Days 4-5 build for real.

Also worth flagging: the very first output in the notebook (before any
of the above) was a hallucinated stock number invented out of nowhere --
a real reminder that ungrounded generation defaults to confidently
making things up, not just occasionally.

## Day 2: Local LLM, GPU Usage, and Hugging Face Workflow

**Key learning areas:** CPU vs GPU, VRAM, CUDA basics, quantization,
4-bit, 8-bit, Ollama, LM Studio, llama.cpp, vLLM overview, Hugging Face
model loading, tokenizer, pipeline, chat template.

**Practical task:** run a local (or API) LLM and test inference.

**Deliverable:** `day2/day2_local_llm.ipynb`.

Installed Ollama via Homebrew, pulled `llama3.2:3b` (~2GB), and ran real
local inference against it through Ollama's local REST API
(`localhost:11434`) -- no API key, nothing leaving the machine after the
model finished downloading.

- **Knowledge cutoff, caught live** -- asked the local model about
  Tesla's current vehicle lineup and got an answer accurate as of a
  while ago, not now, stated with full confidence. Same root cause as
  Day 1's hallucinated Dow Jones number: the model has no live
  connection to the world, only whatever got frozen into it during
  training. Directly motivates RAG later this week.
- **Ollama vs. Hugging Face** -- loaded `TinyLlama-1.1B-Chat` directly
  through `transformers.pipeline` to contrast a service that runs a
  model for you (Ollama, one API call) against loading and controlling
  the model yourself (Hugging Face, own tokenizer/generation settings).
- **Chat templates, the hard way** -- a plain string prompt sent to
  TinyLlama-Chat (a chat-tuned model) with no chat template applied
  didn't get answered at all -- the model just generated more
  similar-looking questions instead of responding, the same base-model
  continuation behavior as Day 6's GPT-2. Switching to a proper
  role/content message list (letting the pipeline apply the model's
  real chat template) fixed it immediately -- same model, same
  question, real answers both times.
- **A real hallucination, caught mid-comparison** -- the raw (broken)
  Overwatch answer described "orcs, trolls" as being in the game, which
  is actually Warcraft, a different Blizzard title entirely. The chat
  template version fixed the *structural* problem (answering at all)
  but not this -- a model can follow format perfectly and still blend
  facts from somewhere else with total confidence.
- **CPU vs. GPU, VRAM, CUDA, quantization, and the local-LLM tooling
  landscape** (Ollama, LM Studio, llama.cpp, vLLM) -- covered
  conceptually in the notebook itself. This machine is a MacBook Air
  with no NVIDIA GPU, so these are understood rather than benchmarked
  directly here.

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

Grew the underlying news dataset in three passes, chasing category
imbalance that would've been a real problem for fine-tuning. First 278
to 446 rows by re-running Week 3's scraper -- it broke on
marketscreener.com, which had added a promo ad element with two `<b>`
tags inside one link, violating the scraper's assumption of exactly
one match, fixed with `.first` on that locator. Politics still sat at
only 12 examples after that, so added apnews.com's politics section as
a new source and expanded the Politics keyword list to catch general
political language ("Trump," "Democrats," "primaries") instead of just
finance-adjacent policy terms -- needed the same bot-detection fix as
marketscreener.com (a real user-agent), and its markup nests the `<a>`
inside the `<h3>` instead of wrapping it, the opposite structure from
every other scraper here. That pass alone took Politics from 12 to 49,
446 to 660 rows total. Energy was still the smallest category at 30
after that, so added oilprice.com as a dedicated energy source, no
bot-detection workaround needed there. Final: 843 rows. Category
breakdown: Business 363, Markets 205, Technology 119, Politics 56,
Energy 56, Health 44.

- **Instruction dataset** -- all 843 headlines converted into Alpaca-
  style instruction/input/output examples (classify the headline into
  one of the six categories), saved as JSONL.
- **Chat-template version** -- the same examples reshaped into a
  `messages` (system/user/assistant) format, the shape a chat-tuned
  model's fine-tuning job would expect instead of Alpaca's flat fields.
- **Train/validation split** -- 758 train / 85 validation (90/10),
  mirroring the same train/test split concept used since Week 1, now
  applied to fine-tuning data specifically.
- **LoRA, QLoRA, and PEFT** -- documented in the notebook: LoRA freezes
  the base model and trains small adapter layers instead of every
  weight; QLoRA adds 4-bit quantization on top of that same idea; PEFT
  is the umbrella term for this family of techniques.

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
