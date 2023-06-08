# Language models can explain neurons in language models
## Introduction
+ Why doing this: <br>
Language models have tens of thousands of neurons so that there is no way to analyze what the neurons are doing one by one (manually inspect the data to determine what features they represent). Want to make AI interpretable
+ What did this research do:<br>
Solved the problem of scaling a parseable technique to large language models (i.e., using the language model to parse a variety of neurons, explaining all of GPT2's more than 300,000 neural units)
+ By doing this, we can?
Review models for security prior to deployment. (Improves LLM performance and reduces AI bias and harmful outputs)

>Our technique aims to explain what patterns (ways) cause the activation of neurons. It consists of three steps:
> + Explanation: Show GPT4 the activation of neurons (Give GPT4 a piece of text and show GPT2 what neurons fire when they understand the text)<br>
> ![image](https://github.com/xinyun2000/papernotes/assets/130521370/e6428adc-d1a5-49ff-baf5-da0d84108b30)
> let GPT4 to guess what this passage about
> + Simulation: GPT4 was used to generate texts according to the previous analysis (references to movies, charactors and entertainment), and GPT4 guessed which tokens would have higher weights in order to generate articles with such topics
> ![image](https://github.com/xinyun2000/papernotes/assets/130521370/d3dbc38f-4ae4-4b91-b5a6-2d02288096a8)
> + Scoring: GPT4 is the weight of the preset token to obtain the article, GPT2 is to obtain the weight of each token through the article, and compare the difference between the two models in the token<br>
> GPT4 is used to quantify and define a concept called "explanation score" to automatically measure its interpretability. Explanation score is the ability of a language model to compress and reconstruct neurons in natural language. With this quantification, we can now the extent to which this model evolves towards a human interpretable perspective.
<br>
____________________________________________________________________________________<br>
- Explanation Score：<br>
Higher interpretation scores, more like the way humans understand natural language<br>
<br>
- How to increase?<br>
Iteration of interpretation. gpt4 is asked to give possible counterexamples, rerevising the interpretation based on their activation<br>
use stronger explainer model and simulator model model <br>
_____________________________________________________________________________________<br>
<br>
But when it comes to looking at absolute scores, it's still far from human interpretation. In terms of typical neurons, like Kats, there are a lot of implications, suggesting that we should change what we interpret.

## Method
contain three model:
+ subject model:<br>
The model we are trying to explain
+ explainer model<br>
Some hypotheses about the behavior of the topic model are proposed
+ simulator model<br>
Hypothesis models need to make predictions based on hypotheses. Based on how well the prediction overlaps with reality, we can evaluate how well the prediction was made. This hypothetical model needs to interpret the hypothesis in the same way as a real person

### STEPS
+ step1:<br>
Write a prompt, send it to the interpreter model, and ask it to generate an explanation of the neuron's behavior. The researcher will give it a few examples
+ step2:<br>
The resulting explanation is used to model the behavior of the neuron (the most accurate activation it should have)
+ step3:<br>
Comparison of simulated and actual neuronal behavior to explain the scoring<br>
(A score of 1 is given if the simulated neuron behaves the same as the real neuron. If the behavior of the simulated neuron is random, for example, if the explanation is independent of the behavior of the neuron, then the score will tend to be around zero.)

## Result
### "top-and-random scoring"
+ meaning:<br>
The snippet of text selected at the top of the explanation generation is used to assess the neuron's ability to represent a particular feature (in the case of some polysemy, it helps the model to pinpoint which meaning it is).
+ score:<br>
The average score is 0.151: This means that the interpretation score using the top text sequence and the random text sequence is 0.151 on average. This score reflects the ability of the explanation to capture the specific features most strongly represented by the neuron as well as to avoid overly broad explanations. A higher score indicates that the explanation better captures the specific features represented by the neuron.

### "random-only scoring"
+ meaning:<br>
Only randomly selected text sequences are used for scoring
+ the score is 0.037, A lower score may mean that the random sample does not capture the specific characteristics of the neuron well.

### these scores generally decrease with deeper layers of the neural network,  which means that interpreting neuronal behavior becomes more difficult as depth increases.

## Some Limitation and caveats
### Neurons may not be explainable
1. A neuron can represent many features(There are polysemous words, and although the explanation can generate an explanation like "it's x, and sometimes y," it still doesn't work.)
2. Alien feature(Might represent alien concepts that humans can't express, things that humans don't even notice (humans don't care or discover natural abstractions that humans haven't yet discovered, e.g., some similar concept family in a different domain).)
### We explain correlations, not mechanisms
- It explains the relationship between the input of the paragraph and each neuron (i.e., how the neurons are activated to obtain the reference to the paragraph, the correlation between them), but it doesn't explain at what level of mechanism these neurons are selected to obtain what the paragraph is about
### Context length
1. Current approaches require that the interpreter model's context be at least twice the length of the text excerpt passed to the topic model.
2. If the interpreter model and the topic model have the same context length, we will only be able to explain the behavior of the topic model in at most half of the context length, and thus may not be able to capture some behaviors that are only shown in the later markup.
3. Subject model: the model which need to be explain(GPT2 in this page)
4. explainer model: the model which generate the exlpanation(GPT4 in this paper)
### Tokenization problem
subject model and explainer model use different tokenization way
### (BIGGEST)The simulation may not reflect human understanding
We used GPT4 to simulate the activation of human beings in response to each word in order to obtain the content of the passage, but it is possible that human beings do not understand this way, they get the reference of the passage in terms of other words
<br>
## My Thought after reading 
1. In fact, while I was reading it, I started thinking about whether GPT4 could really simulate the process of how a human would understand a piece of text when they saw it. In the following limitations, the author also raised this issue. Essentially, it's still using one black box to explain another, and we still don't have control over, or fully understand, the behavior of each neuron.
2. Secondly, in this paper, after giving the tokens that the model focuses on, the weight assigned to each token is predicted and scored. A human might see a keyword, rate this keyword out of 10, and then immediately make a judgment on the central theme of this content, but the LLM model needs to constantly assign different weight coefficients to the token to adjust, and here the test is the correlation between different weight coefficient coefficients and human judgment. Disappointingly, he doesn't explain how the language model chose these words, which means that there is no explanation mechanism, only correlation.
3. But what's interesting and new is that openai is ambitious, and it's a good attempt to solve a problem that everyone agrees cannot be solved any time soon, just as it did with zero-shot.




















