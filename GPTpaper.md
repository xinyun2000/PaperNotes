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
