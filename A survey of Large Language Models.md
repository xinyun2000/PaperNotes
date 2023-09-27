# A survey of Large Language Models

##  four major development stages

### Statistical language models(SLM)

- based on the Markov assumption(predicting the next word based on the most recent context)

- with a fixed context length n, called n-gram language model

  #### n-gram language model

  gram: calculating window, bigram or trigram language models

  the Markov chain, with two tasks:

  1. calculating the joint possibility(unigram model/bag-of words)
  2. calculating the conditional possibility(bigram/trigram/n-gram)

  ##### Unigram language modelling

  every words are independent, with no grammar between each other
  $$
  P(W1,W2,...,Wn)=P(W1)P(W2)...P(Wn)
  $$
  Parameter type: The possibility of a word

  Parameter instances: |V|

   **Out-of-vocabulary**: the words which are not show in the training set

  ​	if the word is not show in the training set,  the model cannot give the possibility of the sentence which contains this word in the testing set

  **how to solve**: smoothing

  ​	add-one smoothing/add-α smoothing: let every word has a nonzero possibility

  ​	![image-20230926170217757](C:\Users\QUAN\AppData\Roaming\Typora\typora-user-images\image-20230926170217757.png)

  #####  Bigram language modelling

  condition on the previous word
  $$
  P(Wn|W2,...,Wn-1)=P(Wn|Wn-1)
  $$
  Using two consecutive words calculates the conditional possibility
  $$
  S=<S>W1W2...Wn</s>
  $$

  $$
  P(S)=P(W1W2..Wn</s>|<s>)=P(W1|<s>)P(W2|W1)...P(</s>|Wn)
  $$

  calculating method： Chain rule + Independent assumptions

  **Sparsity**: the Binary phrase which doesn't show in the training set (much higher than possibility of the word doesn't show in the training set of Unigram)

  **how to solve**: 1. **back-off estimation** 2. **Good-Turing estimation** 

  ![image-20230926171113254](C:\Users\QUAN\AppData\Roaming\Typora\typora-user-images\image-20230926171113254.png)

  ##### Trigram language modelling

  Modeling target: P(s)

  Parameterized model form:
  $$
  P(s)=P(W1|<s><s>)...P(</s>|Wn-1Wn)
  $$
  **Sparsity**: much higher than Bigram language model

  **Back-off estimation**: 

  ![image-20230926172307782](C:\Users\QUAN\AppData\Roaming\Typora\typora-user-images\image-20230926172307782.png)

  ##### Methods to address sparsity:

  ![image-20230926172346145](C:\Users\QUAN\AppData\Roaming\Typora\typora-user-images\image-20230926172346145.png)

  ##### Log probability model

  the number of occurrences is very small, the probability is also very small, not good for calculation, we can use the logarithmic method to transform the floating point number

  

problem of SLM: **Curse of Dimensionality**

- **Data sparsity problem:** have no true neighbor

  e.g. in the K-nearest neighbor algorithm, when the number of variables to be considered becomes sufficiently large, there are almost no true neighbors. When binary variable becomes sufficiently large, the unique combination also becomes sufficiently large. Every unique combination with 10 samples can require an exponential level of the amount of samples.

-  **increased computational complexity**: exponential level of calculation

- **greater data requirements for effective modeling**



### Neural language models(NLM)

<A Neural Probabilistic Language Model>まだ読めない

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

<planning for AGI and beyond> OpenAI まだ読めない

discusses the short-term and long-term plans to approach AGI, GPT4 may be an early version of an AGI system 

GPT4 supports multi-model input by integrating the visual information

The fundamentals of LLM have not been explained:

- Why emergent capabilities appears in the LLM
- It is difficult for the research community to train competent LLMs
- Possibility of toxic, fictitious and harmful content

