# A survey of Large Language Models

##  four major development stages

### Statistical language models(SLM)

- based on the Markov assumption(predicting the next word based on the most recent context)

- with a fixed context length n, called the n-gram language model

  #### n-gram language model

  gram: calculating window, bigram, or trigram language models

  the Markov chain, with two tasks:

  1. calculating the joint possibility(unigram model/bag-of words)
  2. calculating the conditional possibility(bigram/trigram/n-gram)

  ##### Unigram language modeling

  every word are independent, with no grammar between each other
  
  $$
  P(W1,W2,...,Wn)=P(W1)P(W2)...P(Wn)
  $$
  
  Parameter type: The possibility of a word

  Parameter instances: |V|

   **Out-of-vocabulary**: the words that are not shown in the training set

  ​	if the word is not shown in the training set,  the model cannot give the possibility of the sentence that contains this word in the testing set

  **how to solve**: smoothing

  ​	add-one smoothing/add-α smoothing: let every word have a nonzero possibility
    ![image](https://github.com/xinyun2000/PaperNotes/assets/130521370/fe87cb35-2a62-4fab-b51f-6acf460a6fe5)
  
  #####  Bigram language modeling
  calculating method： Chain rule + Independent assumptions
  
  **Sparsity**: the Binary phrase that doesn't show in the training set (much higher than the possibility of the word doesn't show in the training set of Unigram)

  **how to solve**: 1. **back-off estimation** 2. **Good-Turing estimation**
  ![image](https://github.com/xinyun2000/PaperNotes/assets/130521370/f80aa6ba-7618-4fdf-8ef1-0a3bf67bc024)

  ##### Trigram language modeling

  Modeling target: P(s)
  
  **Sparsity**: much higher than the Bigram language model

  **Back-off estimation**: 
  
  ![image](https://github.com/xinyun2000/PaperNotes/assets/130521370/1571e75b-6508-4fb3-89a6-d4de8cb04beb)

  ##### Methods to address sparsity:

  ![image](https://github.com/xinyun2000/PaperNotes/assets/130521370/1d513584-c0ed-4a7b-ab7a-207c7b962fc2)

   ##### Log probability model

  the number of occurrences is very small, the probability is also very small, not good for calculation, we can use the logarithmic method to transform the floating point number

  

problem of SLM: **Curse of Dimensionality**

- **Data sparsity problem:** have no true neighbor

  e.g. in the K-nearest neighbor algorithm, when the number of variables to be considered becomes sufficiently large, there are almost no true neighbors. When a binary variable becomes sufficiently large, the unique combination also becomes sufficiently large. Every unique combination with 10 samples can require an exponential level of the number of samples.

-  **increased computational complexity**: exponential level of calculation

- **greater data requirements for effective modeling**



### Neural language models(NLM)

(A Neural Probabilistic Language Model)まだ読めない

characterize the probability of word sequences by neural networks(RNN)

**Distributed representation of words:** build the word prediction function conditioned on aggregated context features(the distributed word vector)

Reverse: one-hot vector

Constructing a predictive function(also the probability function) under the condition of aggregating contextual features

**word2vec**: The process of mapping a sparse word vector of one-hot form into a K-dimensional dense vector by a one-layer neural network[K can be set by user]

- **CBOW**: analyzes the probability of intermediate words based on n words before and after

- **SKIP-GRAM**: predicts the probability of before and after words based on intermediate words

initiated the use of language models for representation learning



### Pre-trained language models(PLM)

ELMo: pre-training a biLSTM + fine-tuning the biLSTM network according to specific downstream tasks

Transformer architecture with self-attention mechanisms

BERT: pre-training bidirectional language models with specially designed pre-training tasks on large-scale unlabeled corpora

learning paradigm: pre-training+ fine-tuning

pre-training model: GPT-2/BART



### Large language models(LLM)

leads to an improved model capacity on downstream tasks (following *the scaling law*)

***emergent abilities*** (show surprising abilities in solving a series of complex tasks)

Difference between LLM and PLM:

- LLMs display some surprising emergent abilities
- LLMs revolutionize the way that humans develop and use AI algorithms
- the development of LLMs no longer draws a clear distinction between research and engineering

**Artificial general intelligence(AGI)**

(planning for AGI and beyond) OpenAI まだ読めない

discusses the short-term and long-term plans to approach AGI, GPT4 may be an early version of an AGI system 

GPT4 supports multi-model input by integrating the visual information

The fundamentals of LLM have not been explained:

- Why emergent capabilities appear in the LLM
- It is difficult for the research community to train competent LLMs
- Possibility of toxic, fictitious, and harmful content

________________________________________________________________________________________
## Overview

### 2.1 Background for LLMs

large language models refer to **Transformer** Language models(GPT3, PaLM, Galactica, **LLaMA**)

basic background for LLMs: scaling laws, emergent abilities, and key techniques

#### Scaling Laws for LLMs

LLMs significantly extend the model size, data size, and total compute.

need to establish a quantitative approach to characterizing the **scaling effect**

##### KM scaling law

- model performance respective to three major factors: model size(N), dataset size(D), the amount of  training compute(C)
- **a larger budget allocation in model size than the data size**(model size↑ > data size↑）

##### Chinchilla scaling law

- compute budget depends on model size and data size
- **model size and data size should be increased in equal scales**

#### Emergent Abilities of LLMs

has close connections with the phenomenon of *phase transition* in physics

three typical emergent abilities: **in-context learning, instruction following, step-by-step reasoning**

##### In-context learning

- appears in LLMs but does not appear in PLMs
- depends on the specific downstream task

##### Instruction following

- perform well on unseen tasks that are also described in the form of instructions
- have an improved generalization ability
- when the model size reaches quite an amount, it will show good generalization ability

##### Step-by-step reasoning

**Chain-of-thought prompting strategy (CoT)**

> CoT & ICL
>
> Difference between CoT & ICL
>
> CoT focuses on teaching and ICL focuses on  training tasks from simple to difficult

#### Key Techniques for LLMs

##### Scaling

although big data can improve the ability of LLM, it also should be controlled in a reasonable amount

**How to control the calculation:**

1. Keep model size, data size, and total calculation amount at a reasonable ratio
2. use cleaning data

##### Training

The amount of training was too large, so the distributed parallel training algorithm was born.
GPT4 directly implements new optimization solutions. The fundamental purpose is to increase the amount of data while controlling the amount of calculations.

##### Ability eliciting

In terms of generalization ability, general tasks may not reflect it, so professional tasks are needed to activate it and let it reflect such abilities.

##### Alignment tuning

reduce the harmful output

##### Tools manipulation

Limited by the expression format and the amount of pre-trained data, Chat GPT uses plug-ins to help synchronize update iterations

### 2.2 Technical Evolution of GPT-series Models

with two key points to the success:

1. training **decoder-only** Transformer language models that can accurately predict the next word
2. scaling up the size of language models

when there is just RNN, have an idea. when the transformer comes out, OpenAI builds GPT-1

##### GPT1&GPT2

GPT1 is actually PLM

GPT2 starts to perform tasks via unsupervised language modeling, without explicit fine-tuning using labeled data(have the idea of zero-shot/few-shot)

##### GPT3

formally introduced the concept of in-context learning, which utilizes LLMs in a few-shot or zero-shot

become the LLMs

**two major approaches** to further improving the GPT-3 model: training on code data & human alignment

##### training on code data

Codex: trained by Github code

GPT3.5 models are developed based on a code-based model (code-davinci-002)

##### human alignment

using human preference to update the model

##### (milestone)ChatGPT

human-generated conversation

##### (milestone)GPT4

- extended the text input to **multimodal signals**
- introduce a new mechanism called **predictable scaling** that can accurately predict the final performance with a small proportion of computing during model training

__________________________________________________________________
## Resources of LLMs

### 3.1 model checkpoint or APIs

categorize the public models into two scale levels:  *tens of billions of parameters* and *hundreds of billions of parameters*

#### tens of billions of parameters

- FlanT5: explores the instruction tuning from three aspects: ***increasing the number of tasks***, ***scaling the model size*** and ***fine-tuning with chain-of-thought prompting data***.
- CodeGen: autoregressive language model
- mT0: fine-tuned on **multilingual tasks** with **multilingual prompts**
- PanGu-α: good performance in **Chinese downstream tasks** in zero-shot or few-shot
- Falcon: featured by a more careful data cleaning process to prepare the pre-training data

![image-20231012160115525](https://github.com/xinyun2000/PaperNotes/assets/130521370/ab755bb9-db30-4dc4-87dd-7fc3b9941093)

- LLaMA & LLaMA2 & LLaMA-chat

#### Hundreds of Billions of Parameters

only a handful of models have be publicly released

#### LLaMA Model Family

1. effectively adapt LLaMA models in non-English languages
2. often needs to **extend the original vocabulary**(trained mainly on English corpus) or f**ine-tune it with instructions or data in the target language**
3. self-instruct using **text-davinci-003**

#### public API of LLMs

- OpenAI
- two APIs related to Codex: **code-cushman-001 **& **code-davinci-002**

### 3.2 Commonly used Corpora

- books

  > **BookCorpus** (GPT/GPT-2) 
  >
  > **Project Gutenberg**(MT-NLG LLaMA)

- CommonCrawl

  > crawling databases(noisy and low-quality information in web data/perform data preprocessing before usage)
  >
  > **C4**
  >
  > **CC-Stories**
  >
  > **CC-News**
  >
  > **RealNews**

- Reddit

  > GPT2 uses WebText which relay on Reddit, much more accurate than GPT1
  >
  > **WebText** is not publicly available, so **OpenWebText** exists
  >
  > **PushShift.io**: real-time updated dataset

- Wikipedia

  >  the English-only filtered versions of Wikipedia are widely used in most LLMs (e.g., GPT-3, LaMDA, and LLaMA

- Code

  > Github & StackOverflow

- others

LLMs always trains on various sources

### 3.3 Library Resource

- Transformers

  > an open-source Python library for **building models using the Transformer architecture**

- DeepSpeed

  > a deep learning optimization library(compatible with **PyTorch)**

- Megatorn-LM 

  > a deep learning library developed by NVIDIA for training large-scale language models.  provides **rich optimization techniques for distributed training.**

- JAX

  > a Python library for high-performance machine learning algorithms developed by Google, allowing users to **easily perform computations on arrays with hardware acceleration**(GPU & TPU)

- Colossal-AI

  > a deep learning library for training large-scale AI models. based on PyTorch and **supports a rich collection of parallel training strategies.**

- BMTrain

  > an efficient library for training models with large-scale parameters in **a distributed manner**, which **emphasizes code simplicity, low resource, and high availability**

- FastMoE

  > a specialized training library for MoE (i.e., mixture-of-experts) models.
