## Baby Name Generator using Language Model (Bigram)

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

   
   ``` Python

       emb.view(emb.shape[0],emb.shape[1]*emb.shape[2])[:5]
   ```

   2.1 - Summary of the full network
   
   ``` Python
      g=torch.Generator().manual_seed(2134563788) # for reproducibility
      C=torch.randn([27,2],requires_grad=True)
      w1=torch.randn([6,100],requires_grad=True)
      b1=torch.randn(100,requires_grad=True)
      w2=torch.randn([100,27],requires_grad=True)
      b2=torch.randn(27,requires_grad=True)
      parameters=[C,w1,b1,w2,b2]
```
2.2 - Visualizing the loss after 40,000 iterations

   ![image](https://github.com/user-attachments/assets/54ffbcbd-3785-49d8-8a37-bc6f94e93d3a)

2.3 -  Visualizing the embeddings after training

   ![image](https://github.com/user-attachments/assets/27474c09-db54-41df-92e9-665cc2d4ef4d)

2.4 - Generate names after training

   ``` Python
     g=torch.Generator().manual_seed(2147483647+10)
     for _  in range(20):
      out = []
    
      context=[0] * block_size
    
      while True:
        emb=C[torch.tensor([context])]
        h=torch.tanh(emb.view(1,-1) @ w1+b1)
        logits=h @ w2+b2
        y_pred=F.softmax(logits,dim=1)
        ix=torch.multinomial(y_pred,num_samples=1,replacement=True,generator=g).item()
        context=context[1:] + [ix]
        out.append(ix)
        if ix ==0:
          break
      print(''.join(itos[i] for i in out))
```

```
   bria.
   mmyan.
   dee.
   ved.
   ryah.
   bethan.
   bri.
   bri.
   aden.
   daelii.
   brizaen.
   ddennesonnar.
   vayzimtokelin.
   shuber.
   dhi.
   bel.
   vin.
   rweel.
   pxnt.
   porou.
```
2.4 - Results 

   ![image](https://github.com/user-attachments/assets/eca3f3b0-f561-453c-bb8e-87310682e864)





# 3 - Third Version

  In version 3 of the project, I started by performing a diagnosis to better understand the model's behavior. I noticed that the activation functions were saturated, so I adopted some 
  initialization strategies and added Batch Normalization to improve the model's training.

  3.1 - Visualizing  activation functions  saturated

   ![image](https://github.com/user-attachments/assets/88d886b5-10ba-472f-acbd-b4b35f3ecf3e)

  3.2 - Visualizing activation distributions without initiation strategies

   ![image](https://github.com/user-attachments/assets/e7711f4a-5f92-4d3d-a510-701f84635404)
 

  3.3 - Visualizing activation distribuitions  with initiation strategies
 
   ![image](https://github.com/user-attachments/assets/1f700bcd-3b94-41e8-9aab-3c7af8167f50)


   ![image](https://github.com/user-attachments/assets/8c5f345b-a9f9-49d8-a935-2fb38b64b19c)

 3.4 - Batch Normalization + Training Loop

   ``` Python

  batch_size=32
  max_steps=10000
  
  for  i in range(max_steps):
  
    # minibatch construct
    ix=torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    Xb,Yb=Xtr[ix],Ytr[ix] # batch X,Y
  
    # forward pass
    emb=C[Xb] # embed the characters into vectors
    embcat=emb.view(emb.shape[0],emb.shape[1]*emb.shape[2]) # concatenate the vectors
    hpreact=embcat @ W1+b1 #  hidden layer pre - activation
    # BatchNorm layer
    # ------------------------------------------------------
    bnmeani=hpreact.mean(axis=0,keepdim=True)
    bnstdi=hpreact.std(axis=0,keepdim=True)
    hpreact=bngain*(hpreact-bnmeani) / bnstdi + bnbias
  
    with torch.no_grad():
      bnmean_running=0.999 * bnmean_running + 0.001*bnmeani
      bnstd_running=0.999 * bnstd_running + 0.001*bnstdi
  
  
    h=torch.tanh(hpreact) # hidden layer
    logits=h @ W2+b2 # output layer
    loss=F.cross_entropy(logits,Ytr[Yb]) # loss_funtion
    #print(f'loss:{loss.item()},epochs:{i+1}')

  for  p in parameters:
    p.grad=None

  # backward pass
  loss.backward()

  # update
  lr=0 if i  < 10000 else 0.01
  for p  in parameters:
    p.data+=-lr*p.grad

  # strack stats
  if i % 1000==0:# print every once in while
    print(f'{i:7d}/{max_steps:7d}:{loss.item():.4f}')
  lossi.append(loss.log10().item())

```
3.5 - Evaluating Training and Testing Data

  ``` Python
    @torch.no_grad()
    def  split_loss(split):
      x,y={
          'train':(Xtr,Ytr),
          'val':(Xdev,Ydev),
          'test':(Xte,yte),
      }[split]
    
      emb=C[x]
      embcat=emb.view(emb.shape[0],emb.shape[1]*emb.shape[2])
      hpreact=embcat @ W1 + b1
      hpreact=bngain*(hpreact-bnmean_running) / bnstd_running + bnbias
      h=torch.tanh(hpreact)
      logits=h @ W2+ b2
      loss=F.cross_entropy(logits,y)
      print(split,loss.item())
    
    split_loss('train')
    split_loss('val')
```

```

train 3.299492120742798
val 3.2995262145996094

```

