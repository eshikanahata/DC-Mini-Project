# Task 2: The Transformer Architecture

## Part A: Context Dependency

"Bank of the river" vs. "Bank of America"

Short answer is that embedding depends on the context of the sentence.

Now for the long answer, first the input is taken and broken down into tokens. These tokens are then vectorized and stored. 
Let's say that "bank" is a token, initially both of them have the same but as we build up our model, the embedding changes. As we go through more and more layers, it becomes more context dependent. 

## Part B: Attention

*"He told him that he wants to go with his boyfriend, and his boyfriend loves him"*

Now in this sentence, there are multiple entities and a whole lot of pronouns. Quite an ambigous sentence, now lets see how the computer deciphers this. 
 Now we have an embedding matrix with the index as the token id and each row is a vector. 

Now we start with the actual attention process, we put our input into the layers of the transformer. 

In each layer, first a vector is created, this vector(h) is a sum of the vector in the embedding matrix and the positional value of the token in the sentece. 

`h = EM[token id] + Positional vector`

where EM is the Embedding matrix.


Then for each of these vectors a corresponding query (Q), key(K) and value(V) vector is created.

`X = h * Wx`

where x is either of Q,K or V and W is the weight matrix for each of Q,K or V.

The query vector of a token essentially tells us about what the token is looking for, the key vector tells us what information the vector has and the value vector tells us what information we should take from the token. 

The amount of information we take from this vector is decided by a term whcih is called attention.

To actually find what attention is, first we need to find a parameter alpha which is the attention weights for each vector. 

`alpha ij = softmax(Qi.Kj/root(d))`

where 
alpha ij is the attention weight of that vector pair
Qi is the query vector of a token
Kj is the key vector of any token
d is the dimentionsality of the vector

To explain the actual formula, softmax is the way of making all the values positive and make sure that they sum upto 1. Qi is the query vector of a token and Kj is the key vector and their dot product is taken to see how compatable that query and key are, the dot product shows us how much they align with each other. d is the dimensionality of the vector, it helps in normalising and making sure that the values don't blow up.

`Attention = sum of (alpha ij * Vj)`

where 
Vj is the value of that vector
and j changes from 0 to the the (size of the vector-1)

now our embedded vector has this attention added to it and it moves on to the other layers or to predict an output.

Now that we're done with the process, let's use our example.


*"He told him that he wants to go with his boyfriend, and his boyfriend loves him.*

Lets take two words which repeat in this sentence, *him* and *he*, how does the model treat them?

Initially both repetitions of the words have the same vector but as we go layer by layer eventually one him starts pointing towards the person being spoken to and the other him points towards the speaker. He, however, on the other hand always refers to the speaker, how does this happen?

As we go layer by layer, slowly more context is developed and the model learns the two hims refer to different people based on the surrounding words. One is more towards words like told and the other one towards loves and boyfriend. However he comes in similar roles, hence it remains consistent throughout. 

Summing it up, the tokens start with the same values but as it goes through more layers, the model learns more about context and how it affects each word. Ensuring that the model can identify various entities and their relationships effectively.










