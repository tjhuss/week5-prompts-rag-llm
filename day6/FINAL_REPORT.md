# AI News Intelligence Assistant — Final Capstone Report

I built this over five weeks around one dataset, scraped in Week 1 and reused and grown every week after. Week 1 I scraped financial headlines from three sites, cleaned the data, and stored it in a SQLite database. A baseline classifier got me 61% on 6-way category classification, which became the number every later model had to beat.

Week 2 I moved into classical ML, comparing regression and classification models on the news data and two other datasets. A Decision Tree hit 70%, later matched by Gradient Boosting once I wrapped everything in a proper pipeline with feature engineering. Cross-validation work that week taught me a single test-set score can lie to you on small data.

Week 3 I built deep learning from the ground up, a neuron by hand, then activation functions and optimizers, then a full ANN and an LSTM, all tested against the same growing dataset. None of it beat the classical ML results. The dataset was too small for a model to learn English and the task at the same time from nothing.

Week 4 I moved into NLP and transformers, TF-IDF, word embeddings, transformer architecture, then fine-tuned DistilBERT for classification and got 69.6%, the first deep learning model in the whole project to compete with classical ML. I also ran base GPT-2 to see what a model with no instruction tuning actually does when you ask it something directly.

Week 5 I put it all together. Prompt engineering across four task types, running local LLMs through Ollama and Hugging Face, growing the dataset from 278 to 843 rows to fix a category imbalance, building a fine-tuning dataset, and building a RAG pipeline, chunking article text, embedding it, storing it in ChromaDB, and building a chatbot on top that retrieves and answers with citations.

For the capstone demo I wrote one script that takes a headline, classifies it, then asks the chatbot a question about the same topic. Building it turned up a reliability problem I hadn't caught before, the chatbot's generation step wasn't deterministic and would sometimes refuse to answer even when it had retrieved the right source. I fixed that by lowering the temperature and loosening the refusal instruction, and confirmed it held across three separate 10-run tests with zero refusals each time.

I also tested classification accuracy properly instead of on one example, running it against 50 random headlines across five batches: 44 percent correct. That's well above random guessing on a 6-category task, but far below the 69.6 percent the fine-tuned DistilBERT model hit back in Week 4 on the same categories, the same lesson this project keeps landing on, a model fine-tuned on a specific task beats a general model just being asked to do it.

Expanding the RAG chatbot's coverage from 30 to 103 fool.com articles also made a scope limitation obvious: the classifier works across the entire 843-headline dataset, but the RAG chatbot only ever has source material for fool.com, about 12 percent of the total. Testing across a wider set of headlines also surfaced citation problems I hadn't seen on a single example, the model inventing a source number that didn't exist in the given context, and answering questions about a topic that was close to what got retrieved but not actually the same story.

## Repos

- day1-news-scraper (Week 1: scraping, database, baseline model)
- day4-ml-prep (Week 2 Day 1, regression models)
- week3-deep-learning
- week4-nlp-transformers
- week5-prompts-rag-llm (Week 5, this capstone)
