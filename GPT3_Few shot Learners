# GPT3:Language Models are Few-shot Learners
## few-shot:
the same as human, need samples to help model to learn the model. But different as the front one which needs a quite huge amount of samples, this one only need a few
## Abstract
train GPT-3 with 175 billion parameters, 10x more than any previouse nnon-sparse language model, without any gradient update or fine-tuning<br>
can generate samples of news articles

## Introduction
+ recent years have featured a trend towards pre-trained language representations in NLP systems
+ have problem:
++ a need for task-specific database and task-specific fine-tuning
++ when models are designed to be large to absorb information during pre-training, but are then fine-tuned on very narrow task distribution, the fine-tuned model maybe not satisfied to the data which not appears in the database<br>
big model is easy to overfitting on the small database, so a good fine-tuning result does not mean a good generalization.
+ how to solve it:
++ figure 1.1<br>
language model meta-learning(few shot/sero shot)
++ figure 1.2<br>
meta-learning（in-context learning）,don't need to do any refresh
model is a technic model which contains 10-100 samples
++ few-shot learning, provide maybe 10-100 training samples
++ one-shot learning, provide only one sample(like show how helloworld translates to French and then let you translate to French)
++ zero-shot learning, don't give you any sample

## Approach
+ few shot<br>
  more extension on zero-shot
+ one-shot<br>
  in-content learning,but only focus on in-content
+ zero-shot<br>
only a natural language description of the task
+ fine-tuning<br>
Train pre-training during fine tuning, provide some samples on each task, assuming using batch size 1 for training, give me one sample at a time, because there are labels to calculate losses and weights to update<br>

Fine tuning requires less data than training from zero because the start and end of the fine tuning are close

## Model and Architecture
+ use the same model and architecture as GPT2
+ have little difference: Sparse transformer
+ 8 different sizes of models

especially: when the batch-size becoming smaller, the learning rate also come to smaller

## Training Dataset
Common Crawl(two changes were made)
1. filered a version of Common Crawl<br>
  (download the database which used by gpt2, as a positive example. Download the data of commoncrawl, do the dichotomy, the data which like gpt2's data will be collected and the data which is useless will be delete)
2. lsh algorithm(local sensitive hash)<br>
  (lsh algorithm can judge the similarity between two sets, so as to determine whether two sets coincide. This completes the process of removing repetition.)

## Training Process

## Evaluation
evaluate the model directly after the pre-training<br>

the way of evaluating the model: doing the in-content tasks<br>
in the training set of the subtasks, select k samples, the k can be 0,1 or 10 to 100<br>
prompt=Answer: / A: (when doing category, select few sample to show what do I want to do)<br>
if dichotomy, answer will show true or false<br>
if the answer is much free, when doing Q&A, use beam search to find a best answer

## Result
1. As the number of calculations increases, the linear loss decreases(but the calculator also linear increases)
2. open domain: Q&A

## Limitation
1. Although much better than gpt2, it is easy to repeat several paragraphs after generating several paragraphs
2. Structural and algorithmic limitations<br>
3. unable to teach model which token is important and which one is not important
4. sample size is quite large
5. uncertain: cannot know is it really learn from the beginning, or just finding out the similar part from the training memory.

## Conclusion
use zero shot/one shot/few shot at the same time


  
  
 
