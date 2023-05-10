# Attention is all you need
transformer
## Abstract\Conclusion\Introduction\Background
+ Do only use attention without RNN or CNN
+ Makes generation less time-sensitive. don't need to input all the previous ht-1 before it can calculate t, it can be parallelized.
+ would't lose previous information (maybe also can be done by LSTM?)
+ also use coder and decoder, have multi-head attention mechanism.
## model architecture
![image](https://user-images.githubusercontent.com/130521370/234434994-943afafd-5d8f-4107-95ba-3f27edf60483.png)
+ left is encoder and right is discoder
+ use 自回归（auto-regressive）
### encoder and decoder stack
### encoder
> 6 identical layers, each layer will have two sub-layers, first a multi-head self-attention mechanism, then a simple position-wise fully connected feed-forward The network is actually an MLP multilayer perceptron, using a [residual connection](https://zhuanlan.zhihu.com/p/46393518) (residual connection resnet) for each sub-layer, and finally using something called layer normalization
#### layer norm\batch norm
+ batch norm: 
> For the batch norm is for a feature to count, so it will involve a lot of different length samples, here although they will be complemented by 0, but the length of the effective data is a large difference, so that the calculation of the variance will be more complex
+ layer norm: 
> This problem does not exist for the layer norm, because the layer norm is to find the variance and mean for each sample by itself.
No matter the sample is long or short, the mean value will be calculated more conveniently.
### decoder
+ the decoder is also composed of a stack of N=6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.
+ same as encoder but with a modify the self-attention masking
### attention(Scaled Dot-product attention)
+ Use query and key to compare(Similarity) and finally multiply its own value
+ what transformer use is called Scaled Dot-product attention
+ use this to collect the information from whole sequence, have more flexibility
### how to use attentiom mechanism in transformer
1. Encoder input (what the encoder input is doing)
> If the sentence length is n, the input is n vectors of length d. Assuming that the value of pn is 1, it is equivalent to a sentence in which there are n words each of which is a vector of length d.
The input multi-head attention is three inputs, and the input is used as key and value as well as query, this is called self-attention, self-attention mechanism (QKV same)
2. Decoder input(first Attention in decoder)
> In fact, for the decoder, except for the input, which is different from the encoder, to input the content of all sequences before t, but to mask the content of t+1, so it needs to masked after
3. The decoder's second Attention
> No longer a self-attentive mechanism, the key and value come from the output of the encoder, and the query comes from the previous t-1 part, that is, the part of the input below
The final output of the encoder is a vector of m vectors of length d

_________________________________________________________________________
## My thought after reading
It can be found that in machine learning and subsequent natural language processing, more and more emphasis is placed on unstructured and more flexible processing of the input information, and Transformer is the pioneer of the subsequent new models including Bert and GPT. The transformer uses fewer parameters to be trained and allows the model to be used for different tasks without the need to build a new model structure for each task. It is not necessary to build a new model structure for each task. However, the disadvantage is that a large amount of data training is needed to help train a single parameter to meet the x-training set results, and this disadvantage seems to persist in GPT.
