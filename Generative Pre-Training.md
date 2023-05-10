# Improving Language Understanding by Generative Pre-Training
GPT
## Abstract
*problem we should solve*
+ we should solve diverse tasks in NLP, although we have large unlabeled text, we have little labeled text. It makes us hard to train model on these little labeled text.

*how to improve*
+ train a generative pre-training model(language model) on unlabeled text, do discriminative fine-tuning on labeled text

*differece from front works*
+ task aware input minimal changes to the model architecture
+ the used NLP model needs to build new model each time when we solving the different sub-tasks when gpt only need to fine-tune the model

## Instruction
how to use unsupervised model

*the problem when we use unlabeled model now*
+ don't know to use what optimization objective, different objective functions have different degrees of goodness for different problems
+ How to effectively transfer learned text representations to subtasks

1. a semi-supervised model be raised, train a pre-training model on unlabeled text and fine-tune the model on labeled text.
2. use transformer model, because it will be more stable when doing transfor learning
3. use task-specific input

## Related workre
1. the Semi-supervised learning for NLP(how to train pre-training model on unlabeled model)
2. unsupervised pre-training(how to do fine-tuning on labeled text)
3. What happens when you're training with multiple target functions(use two traget function when doing fine-tuning)

## Framework
### 3.1 Unsupervised pre-training
![image](https://github.com/xinyun2000/papernotes/assets/130521370/892a02aa-1e45-4743-b8ad-64e6b54bd3ba)

This means that each word in a sentence is marked by a serial number, which does not change the position of the word in the sentence
![image](https://github.com/xinyun2000/papernotes/assets/130521370/932ca3bd-5e5b-426a-9e74-db78e75cdb97)

The language model also uses the preceding word to predict the occurrence probability of the i-th word

Specifically: k words in front of ui, k is the window size (the length of the input sequence), take k words at a time to predict who will be the word after k words
+ Model theta, given k words, given model, predicts the word after k words, gets the first objective function, denoted L1 ()
+ Why add: Because it takes the log, takes the exponent and puts it back to multiply the probability of all of these words, the joint probability of this text appearing (Markov chain (?). , Bayes formula (?) )

Only transformer decoder can be used, not its encoder, because unsupervised pre-training can only predict the probability of next words by previous words (a standard language modeling)

Compared with bert, bert predicts the information in the middle through the information in the front and the back, which is equivalent to cloze, with much higher accuracy. However, if gpt can predict the future, it will be more powerful than bert
### 3.2 supervised fine-tuning
+ When we get the unsupervised pre-training model, we apply its value directly to the supervised task. For a labeled data set C, each instance has m input tokens: {x1,... xm}, which is composed of the tag y. First, these tokens are input into the trained pre-training model to obtain the final feature vector hml. Then, through a fully connected layer, the predicted result y is obtained:
+ Wy is the parameter of the fully connected layer.
![image](https://github.com/xinyun2000/papernotes/assets/130521370/1f1f9d36-de6e-4dca-9f28-4d07a3c3baa0)

finally, just like we did before how do we maximize all the probabilities
![image](https://github.com/xinyun2000/papernotes/assets/130521370/9c81021b-4969-4901-95fa-fb67f750226a)

This is a standard classification task
+ Of course, in the process of optimization, we will only pay attention to the above objective function and how to maximize the probability of obtaining labely with a word sequence. But at the same time, what if the previously given objective function to predict the probability of n+1 words with the first n words is also included
![image](https://github.com/xinyun2000/papernotes/assets/130521370/0592c82d-df3c-477a-aa41-633539eef4eb)

λ is a hyperparameter that can be adjusted (the parameter of λ is generally 0.5
### Task-specific input transformations
How to represent some typical small tasks of NLP in the form we need (a sequence, and a label for y)

![image](https://github.com/xinyun2000/papernotes/assets/130521370/9dd546f1-d7b1-47bc-a360-cfbed90b1631)
1. *Classification*
- e.g. (Classification tasks, give a piece of text, judge the label corresponding to this text, such as whether the customer's evaluation is positive or negative)
- The text that needs to be classified is made into a sequence by placing a starting token in front and an ending token in the back. The features extracted by the last word are put into the linear layer (newly added in the fine-tuning of GPT) by the model and projected into the space of the label (the text becomes a sequence, and the label is placed behind for training).
2. *Entailment*
- e.g. (giving you a paragraph, and then I'm going to give you a hypothetical paragraph, and I'm going to show you if it contains anything hypothetical.)
- Give a paragraph: a gives b a bunch of roses - Suppose: a likes b, which is supported by the previous paragraph<br>
It's kind of a triad problem, and this passage supports or does not support or is neutral about the assumptions made about this<br>
premise是这段话，Hypothesis是假设，start是开始的token，delim是中间连接的token，extract是最后的结束token，start,delim和extract三个token是不在字典里的，否则会混淆
3. *Similarity*
- e.g. ((Similarity, because similarity is mutual relation, two pieces are made to compare a and b, and b and a are similar. Finally, they can be added together, or concat function can be used.)
- The result is a binary problem of whether similar or not similar
4. *multiple choices*
- e.g. (Multiple choice, give several answers, and finally select the most correct one among the answers. If n answers are generated to generate a question, calculate such a scalar for each answer, and finally do a softmax, you can know the confidence degree of this text to each answer)

we can find that no matter how change the sub-task, no change to the model

## Experiments
### 4.1 setup
Supervised pre-training (BooksCorpus data set), a corpus that uses 7,000 unpublished books and contains a large amount of continuous information, generative model to learn to condition on long-range information) Generative model to learn to condition on long-range information)
________________________________________________________________________________________
*what is called perplexity:*<br>
In short, it is the evaluation index of language model.

The common evaluation index for the effectiveness of language models is the perplexity. The lower the perplexity of a test set, the better the effect of modeling. The formula for calculating perplexity is as follows:
![image](https://github.com/xinyun2000/papernotes/assets/130521370/21a0267c-ca0c-432f-9441-e36d5c04c890)

In simple terms, perplexity describes the ability of language models to predict a language sample, for example, what has been known (w1,w2,w3,... ,wm)(w_1,w_2,w_3,... ,w_m) will appear in the corpus, so the higher the probability of calculating this sentence through the language model, the better the language model fits this corpus.
________________________________________________________________________________________
model specifications:<br>
(model use 12 transformer decode layer, with 768 dimension)<br>
Using the adam optimizer, spaCy word segmentation structure

fine-tuning details
add dropout to the classifier with a rate of 0.1
we useL3(C) = L2(C) + λ ∗ L1(C) ,  λ regularly set to 0.5

### 4.2 Supervised fine-tuning
Conduct experiments on various supervisory tasks, including text classification, implication, question-answering, and semantic similarity analysis<br>
Different corpus data sets are used for different small supervisory subtasks<br>
In the end, state-of-the-art results were obtained in 9 of the 12 data sets

## Analysis
Impact of number of layers transferred

*_Zero-shot Behaviors_*<br>
openai wants to migrate the model so that it can be trained unsupervised and then applied directly to multiple different downstream subtasks<br>
This is where the concept of zero-shot comes in
________________________________________________________________________________________
*what is zero-shot:*
- The training set data is used to train the model, so that the model can classify the objects of the test set, but there is no intersection between the training set category and the test set category. During this period, it is necessary to establish the relationship between the training set and the test set with the help of the description of the categories, so as to make the model effective. (This will be applied to GPT2 to enable better migration of the entire model)
- one example of zero-shot
- Assuming we know the morphological characteristics of donkeys and horses, that tigers and hyenas are striped animals, that pandas and penguins are black and white animals, again, we define zebras as black and white striped equine animals. Without looking at any pictures of zebras, just by reasoning, we can find zebras among other animals in zoos.

The example above involves a process of reasoning that uses past knowledge (descriptions of known animals) to mentally reason about the specific shape of the new object so that the new object can be identified. ZSL hopes to emulate this reasoning process, giving computers the ability to recognise new things.

![image](https://github.com/xinyun2000/papernotes/assets/130521370/dc262514-900a-4bed-858a-7ac1d1294be5)

However, the current pure supervision model often needs enough samples to train a good enough model, and the classifier trained with panda can only classify pandas, but cannot identify other species, and can not conduct comprehensive reasoning of features at the same time
______________________________________________________________________________________
Here, GPT compares transformer and LSTM for zero-times learning by reasoning with similar features, which has a better effect, and also shows that transformer is suitable for zero-times learning

## Conclusion
A framework was developed to allow him to generate the model in the unsupervised state and then fine-tune it in the supervised state, so that he could give a good prediction effect on different tasks with less dependence on the types of tasks and facilitate migration.

For the future, greater use of unsupervised learning could lead to new research into natural language understanding and other areas of unsupervised learning, further improving our understanding of how and when unsupervised learning works.

# What did I learn
- The subsequent GPT2 and GPT3 were obtained by iterating on the model based on GPT. 
- From the GPT paper, we can already see that apenai hopes to make the whole model more generality, puts more focus on the pre-training model with unlabeled data, and minimizes the model adjustment for the downstream subtasks as possible. Less dependence on labeled data, let him conduct unsupervised or even self-supervised model training, which is conducive to the use of more data to train a larger model (do not need labels can save a lot of data preprocessing links), and can be well transferred to other subtasks. 
- It is worth noting that the zero-shot behavior proposed in the GPT paper will be an inspiration for the subsequent use of prompt, which is more similar to human natural language input. 
- At the same time, the GPT model uses a lot of data to train the model, which may become a big problem that is difficult to reproduce.


