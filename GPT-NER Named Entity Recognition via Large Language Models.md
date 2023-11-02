# GPT-NER: Named Entity Recognition via Large Language Models

**main issue to be solved:**

GPT perform not quite good on NER(Named Entity Recognition) task.

**reason for the issue:**

GPT is a text-generation model, but not a multi-classify model, NER task is a sequence labeling task.

**to solve it:** 

change the classification task to text generation task.

## Introduction

what is NER task: divide sentence in tokens and label each token.

_________________________________________________________________________________________________________________________________________________

LLM's biggest problem: hallucination issue

to solve it: add self-verification strategy right after the entity extraction stage, let LLM ask itself whether the word belongs to the classification which we ask it to output



## background

**normal NER task solution:** 

1. Representation Extraction

   - embedding each sentence and send them to the model, model output a much higher dimensional vector with the representation of each words

   - e.g., each token[50 dimension], each sentence has 5 token[5,50dimension], BERT parameter is 1024, output[5,1024]

2. Classification

   - use SOFTMAX to classify(本质上先用训练集调整每个label的参数，后续参数进来对比该参数和label参数的相似度，得出概率)

## Main Body

**GPT-NER**

three steps:

1. Prompt Construction
2. generate text sequence
3. transform sequence to get final result

### Prompt Construction

1. #### Task Description

   - task description 

     **“I am an excellent linguist”**

   - variable sentence to help iterate over all entity labels (transform N binary classification to N个 binary classification)

     **"The task is to label [Entity Type] entities in the given sentence"**

   - instruction for LLM to check the *"few-shot"* below

     **“Below are some examples”**

2. #### Few-shot Demonstration (provide in a list of example)

   why:

   -  for format output

     if does not contain any entity, copy the sentence and output

     if contains, use special tokens "@@"&"##" to surround the word

   - help to predict

   how to retrieve Demonstration:

   - Random Retrieval

     may have different semantical relationship with the sentence which need to be labelled.

   - kNN-based retrieval

     sentence-level Embedding: 是对比句子之间的关系

     ***token-level Embedding***：对比词之间的关系

3. #### Self-verification

   - Ask it self whether the word it labels belong to the classification which we ask it to output

   - prompt:

     **(1) “The input sentence: Only France and Britain backed Fischler’s proposal”,**
     **(2) “Is the word "France" in the input sentence a location entity? Please answer with yes or no”.** 
     **(3) Yes.**

   - referenced demonstration Selection

     Random Retrieval

     kNN-based retrieval

![1698909539036](C:\Users\QUAN\Documents\WeChat Files\wxid_huobapv11fgp21\FileStorage\Temp\1698909539036.png)

![1698910374651](C:\Users\QUAN\Documents\WeChat Files\wxid_huobapv11fgp21\FileStorage\Temp\1698910374651.png)



