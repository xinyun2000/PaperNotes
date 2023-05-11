# GPT2：Language Models are Unsupervised Multitask Learners 
## Abstract
also use the decoder of transformer, but trained on a new dataset of millions of webpages called WebText.<br>

_Zero-shot_:
* The current trend is to collect one data set for each task, because the model is not very generic, and the trained model is difficult to apply directly to the next model<br>
* multitask learning: train model with several database, but not very popular in NLP<br>
* in this paper, still work on language models, but when comes to the down-stream tasks, use a zero-shot setting
_______________________________________________________
what is called _zero shot_?<br>
when doing the down-stream tasks, don't need any labeled data, of course don't need to train or fine tune any model.
_______________________________________________________


## introduction
* the limiations of current machine learning systems: highly skilled in narrow tasks, but not generalists, need to train before every tasks.
* proposes multitask learning, improve general performance
* big potential of language models to perform a wide range of tasks in a zero-shot setting


## Approach
* The core of the whole model is the language model, which is a group of Windows composed of symbol sequences of different lengths. Because language has its natural word order, it is common to factor the joint probability of symbols into the product of conditional probabilities<br>
![image](https://github.com/xinyun2000/papernotes/assets/130521370/9d061e1a-de52-4d75-a9a5-d05140409854)
* to perform a single task is, for instance in judging the input text emotion classification, at the time of output the correct judgment, can use p (output | input) the conditional probability formula to calculate the output probability of correct judgment, but not only use a child tasks and classification task, So we should add for the task to distinguish his p (output | input, task)<br>

<u>_view gpt model:_</u>
* The model is constructed in natural language
* but when fine tuning, use start, extract and delim token on the natural language, it haven't been learn by model, it will have big puzzled in zero-shot setting<br>

_so we use prompt_
for example: 
+ a translation training example can be written as the sequence(translate to French, english text, french text)
+ a reading comprehension training example can be written as(answer the question, document, question, answer)
+ the first three word is the hint of model, telling model what need to do<br>
not created by GPT

___________________________________________________________
_one puzzled part:_<br>
Since the supervised objective is the the same as the unsupervised objective but only evaluated on a subset of the sequence, the global minimum of the unsupervised objective is also the global minimum of the supervised objective.<br>
+ talking about why zero-shot setting can use prompt to let computer know what to do
+ The basic principle here is that if f is a function whose domain is D, and S is a subset of D, then d will also maximize f on S if D maximizes f on d, and d also happens to be a member of S.<br>

What is a global maximum and a local maximum for GPT?
> step1: For unsupervised data, using a lot of learning to predict the next word, for example, given hello worl, he can predict that the next letter is d<br>

> step2: We applied the pre-training model to the actual subtask and asked him to answer, "Who wrote the book the origin of species?" "It answers "Charles Darwin", seemingly answering a different question than the first one, but actually a prediction of the natural language of English

> The first step is also the prediction of English natural language, so in fact their underlying logic is the same, and the optimization function used can be the same, that is, the global maximum is also the local maximum
______________________________________________________________
The second paragraph discusses the feasibility of using an unsupervised pre-training model to implement zero-shot, and how powerful the pre-training model would be if it were possible

### 2.1 Training Database
+ (diss) Common Crawl, two much useless form of data, needs to much data processing part
+ (support) reddit, scraped all outbund links, which received at least 3 karma

### 2.2 Input Representation
1. The large language model (LM) should be able to calculate (and generate) the probability of an arbitrary string, but it requires characters to satisfy certain formats (so there are preprocessing steps, such as converting to lower case, word segmentation, and tokens for handling unknown words), so in practice the range of strings the model can handle is limited.
2. can model using Unicode strings, but with many different forms of the same state, such as dog.dog! dog?
3. (_highlight_) stop bound dog with marks, add an exception for spaces which significantly improves the compression efficiency<br>
Combine the advantages of LM with the advantages of Unicode to set a probability for each Unicode code

### 2.3 Model
Most of it is the same as the GPT. But the move to layer normalization has moved to the input of each subblock, similar to the pre-activated residual network, with additional layer normalization following the final self-attention blocks


## Experiment
The model was trained to be two times larger than the GPT

### 3.1 language model
+ When using the model for zero-shot task generalization, the performance of WebText language model in language modeling task is evaluated to understand its performance in generalization. Because Unicode is used, there is no need to preprocess or tokenize the data
+ These data sets differ significantly from the distribution of their training data, including the over-standardized text, problems caused by tokenization, and a variety of other variations. In order to obtain more accurate evaluation results, we use a reversible de-labeling method to treat these markers and preprocessing features.
+ get good result on different database

### 3.2 Children's Book Test
_task: Predict which of the 10 possible choices of ellipses is correct._<br>
Rather than reporting perplexity as an evaluation metric, CBT reports accuracy on an automatically constructed cloze test where the task is to predict which of 10 possible choices for an omitted word is correct.  

### 3.3 Lambada
_task: tests the ability of systems to model long-range dependencies in text._<br>
to predict the final word of sentences which require at least 50 tokens of context for a human to successfully predict. 

### 3.4 Winograd Scheme Challenge
_task:to measure the capability of a system to perform commonsense reasoning by measuring its ability to resolve ambiguities in text._

### 3.5 Reading Comprehension
1. name: Conversation Question Answering dataset 
2. what: consists of documents from 7 different domains paired with natural language dialogues between a question asker and a question answerer about the document.
3. test what: tests reading comprehension capabilities and also the ability of models to answer questions that depend on conversation history (such as “Why?”).
4. problems show from this test: While GPT-2’s performance is exciting for a system without any supervised training, some inspection of its answers and errors suggests GPT-2 often uses simple retrieval based heuristics such as answer with a name from the document in response to a who question.<br>
    e.g. answer with a name from the document in response to a who question.

### 3.6 Summarization
perform summarization on the CNN and Daily Mail dataset (Nallapati et al., 2016). 

### 3.7 Translation & 3.8 Question Answering


## Generalization vs Memorization
+ This section focuses on the data duplication rate in the data set
+ An 8-gram Bloom filter containing WebText training set tags was created to analyze the repetition rate of test data in training data. For the most part, it's unexplained, but it could be a direction of research. It is recommended to use n-gram to recreate the processed data set
+ The performance of GPT in the training set and the test set is similar, but it improves with the increase of the model scale, indicating that there are still underfitting conditions in many places
+ GPT2 can write its own news articles (fake news generator lol)


## related
The relevant research on the performance of language models and pre-training methods is discussed. It also mentioned the training of larger language models on larger data sets, the application of extended language models on different tasks, the method of constructing large text corpus, and the application of LM pre-training on various generation tasks.
  
  
## discussion
+ Unsupervised task learning is a promising research area.
+ Performance is still very basic
+ When the language models have sufficient capacity, they begin to outperform the baseline models.
  

_________________________________________________________________________________
# What did I learn
+ openai is taking a longer view and wants to generlize to multiple subtasks in a zero-shot approach without subsequent model tweaks and retraining. From a subsequent perspective, this was achieved on chatgpt.
+ In the process of training, the convenience and use of larger data sets, here the original simple prediction of natural language words, into the prediction of Unicode code, so that can solve the problem of having too many words in the dictionary. However, it is important to note that when the same word is combined in different ways, it appears repeatedly in the dictionary  
+ In general, GPT2 puts forward the hypothesis of zero-shot, but in terms of the overall model building, it relies on a larger amount of data, and there are not many innovations
  
  
  
  
