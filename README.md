# Generative-AI-with-LLMs
The Lab assignment of the Generative AI with LLMs course by DeepLearning.Ai


# Transformer Architecture
Attention is All You Need
https://arxiv.org/pdf/1706.03762
 - This paper introduced the Transformer architecture, with the core “self-attention” mechanism. This article was the foundation for LLMs.

BLOOM: BigScience 176B Model  
https://arxiv.org/abs/2211.05100
 - BLOOM is a open-source LLM with 176B parameters (similar to GPT-4) trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model. You can also see a high-level overview of the model 
here
.

Vector Space Models
https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3
 - Series of lessons from DeepLearning.AI's Natural Language Processing specialization discussing the basics of vector space models and their use in language modeling.

# Pre-training and scaling laws
Scaling Laws for Neural Language Models
https://arxiv.org/abs/2001.08361
 - empirical study by researchers at OpenAI exploring the scaling laws for large language models.

# Model architectures and pre-training objectives
What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?
https://arxiv.org/pdf/2204.05832.pdf
 - The paper examines modeling choices in large pre-trained language models and identifies the optimal approach for zero-shot generalization.

HuggingFace Tasks
https://huggingface.co/tasks
 and 
Model Hub
https://arxiv.org/pdf/2204.05832.pdf
 - Collection of resources to tackle varying machine learning tasks using the HuggingFace library.

LLaMA: Open and Efficient Foundation Language Models
https://arxiv.org/pdf/2302.13971.pdf
 - Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)

# Scaling laws and compute-optimal models
Language Models are Few-Shot Learners
https://arxiv.org/pdf/2005.14165.pdf
 - This paper investigates the potential of few-shot learning in Large Language Models.

Training Compute-Optimal Large Language Models
https://arxiv.org/pdf/2203.15556.pdf
 - Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.

BloombergGPT: A Large Language Model for Finance
https://arxiv.org/pdf/2303.17564.pdf
 - LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.


FLAN paper: Scaling Instruction-Finetuned Language Models
https://arxiv.org/abs/2210.11416

How Is ChatGPT’s Behavior Changing over Time?
https://arxiv.org/pdf/2307.09009.pdf



# Model Evaluation Metrics
HELM - Holistic Evaluation of Language Models
https://crfm.stanford.edu/helm/latest/
 - HELM is a living benchmark to evaluate Language Models more transparently. 

General Language Understanding Evaluation (GLUE) benchmark
https://openreview.net/pdf?id=rJ4km2R5t7
 - This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.

SuperGLUE
https://super.gluebenchmark.com/
 - This paper introduces SuperGLUE, a benchmark designed to evaluate the performance of various NLP models on a range of challenging language understanding tasks.

ROUGE: A Package for Automatic Evaluation of Summaries
https://aclanthology.org/W04-1013.pdf
 - This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.

Measuring Massive Multitask Language Understanding (MMLU)
https://arxiv.org/pdf/2009.03300.pdf
 - This paper presents a new test to measure multitask accuracy in text models, highlighting the need for substantial improvements in achieving expert-level accuracy and addressing lopsided performance and low accuracy on socially important subjects.

BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models
https://arxiv.org/pdf/2206.04615.pdf
 - The paper introduces BIG-bench, a benchmark for evaluating language models on challenging tasks, providing insights on scale, calibration, and social bias.
