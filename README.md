# Baby Name Generator using Language Model (Bigram)

## Overview

This project leverages advanced natural language processing (NLP) techniques to create a baby name generation model. Harnessing the power of language models, we developed an innovative solution for generating unique and meaningful name suggestions.

## Dataset

The model was trained on a comprehensive database containing:
- Total of 32,023 names
- Diversity of origins and cultures
- Male and female names


## 1- First version

In this first version of the project, we created a simpler model, with only a one-hot encoding layer for text representation and an activation function, without embeddings, without MLP... List the "withouts."
 - Without embeddings
 - Without MLP (Multi-Layer Perceptron)
 - Without advanced representations
 - Without complex layers (such as RNNs, LSTMs, or Transformers)
 - Without attention mechanisms
 - Without deep learning with multiple layers

  ![image](https://github.com/user-attachments/assets/230e104b-ae7d-4fd8-9fdb-99c7a4494823)
  

  1.1 -Visualizing the bigram dataset.
 
  ![image](https://github.com/user-attachments/assets/3dec20ce-2653-4c32-803b-ff9d6f67e6e7)
  

   1.2 - Generate names after training
   
  ``` Python 

      g=torch.Generator().manual_seed(2147483647)
      for i in range(5):
         out=[]
         ix=0
         while True:
             p=P[ix]
             ixs=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
             out.append(itos[ix])
             if ix==0:
                 break
         print('.join(out)')
```

``` 1.3 - Names generated

         junide
         janasah
         p
         copy
         a
``` 
  ##  2 - Second Version
  In the second version of the project, I implemented embedding layers, which enabled more efficient training.

  ## 2.1 - 









