# Task 1 : Tokens and Vectors

## Part A

### How can we train a model to process text?

Computers run on numbers, and directly processing text character by character would be so incredibly computationally inefficient, that's why we need to have a better way. 

This is where tokens come in, we can represent a character, a part of a word, a word or even a series of words as a token. These tokens are then mapped to a unique numeric token id.

Vectors are objects which are used to represent these token ids.

### Comparison of Different Phrases

Using TikTokenizer
The respective tokens for 

`The 5th place result : 200264, 17360, 200266, 976, 220, 20, 404, 2475, 1534, 200265, 200264, 1428, 200266, 200265, 200264, 173781, 200266`

`The fifth place result : 200264, 17360, 200266, 976, 29598, 2475, 1534, 200265, 200264, 1428, 200266, 200265, 200264, 173781, 200266`

We can see that all the other tokens are the same except for the ones representing "5th" and "fifth". 
"5th" is represented by two tokens 20 for "5" and 404 for "th" while "fifth" is represented by 29598.
But there is another difference, the space between "The" and "5th" is taken as an individual token.
This happens because "fifth" is a term used commonly but "5th" is rarer. It is based on statistical frequency while training.

