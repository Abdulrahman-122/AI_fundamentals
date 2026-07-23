# Convolutional neural network(Cnn)
  - this type excells at  image processing that includes(convolutional layer,pooling layer,fully-connected layer)
  - used in pattern  classification
  - this type of  models uses  kernel when processing images(uses filter matrix)
  - notes:
    - CNN use local receptive field-> needs to know local images ->then  combined these local  images to  form global information(far images)
    - CNN  uses parameter sharing(all elements of the images share the same  weights(parameters) so the cores of  the models when they are scanning the image ->their parameters are  fixed.
- Architecture of CNN:
  - <img width="629" height="374" alt="image" src="https://github.com/user-attachments/assets/e6d2ad58-0879-4e85-ab90-bd850017695d" />
    - you have two layers that forms the CNN:
      - Convolution layers.
      - Pooling layers
        - they both forming the fully-connected layer.
      - what is the convolution layers?
          - it's contain several layers that are composed
          - each layer has parameters that obtain from  backward propagation algorithm.
          - 1st convolution layer extract small features
          - multi-leyer neetwork ->can extract more complex features
      - what is the Relu layer?
        - Relu(Rectified linear unit) is a  function may be used as activation function in this layer.
      - What is the Pooling layer?
        - this layer take partition layers that came from convolution layer and try to extract smaller features again.
      - What is the Fully connected leyer?
        - this  layer  combine both local features with each other to form global features in order to define the scores of each layer.
      - What is the output layer?
          - this layers output the final result.
          - <img width="561" height="249" alt="image" src="https://github.com/user-attachments/assets/e2cce59b-42e9-451f-80b2-60c46b94302f" />
              - this is the result image (3*3) as you enter 3*3 filter.
           

    - What is the Convolution layer in Detail?
        - it'a multi layers
        - output of the previous layer -> input the next layer
        - the output of  each layer ->convolved with convolution kernel
          - this convolution of each layer is a weight to learn
          - then this convolution should be biased through an activation function before going to the next layer
          - <img width="578" height="227" alt="image" src="https://github.com/user-attachments/assets/e09df0b9-b9ab-478a-a93d-b25f2a43d3c5" />

    - What is the Pooling layer?
     - layer combines nearby units -> in order to reduce size of  input layer.
     - it's uses pooling methods (max pooling,average pooling ) to do that task
     - max pooling -> choose max value from the square that represent the area of the input
     - average pooling -> choose  average value from the square that represent the area of the input
     - <img width="581" height="269" alt="image" src="https://github.com/user-attachments/assets/0bbf1cff-d414-4bc4-913e-a3b5129dd646" />
       
        so to rap up this layer:
         - Pooling layer contain ->
           - invarience(max pooling)
           - Reducing input size (including number of parameters)
           - obtain fixed lenght data.
           - increase the scale
           - prevent overfitting.
          
 - What is the fully-connected Layer?
     - it's a layer that used as a classifier for the output of both(convolution,pooling)layer.
     - Softmax function used with it -> it combine all local featues into global ones + compute score of each class.
     - this is the equation of softmax: <img width="246" height="77" alt="image" src="https://github.com/user-attachments/assets/fe7ee3b4-f9e1-42a3-b981-a3cca088ae58" />

--------
# Recurrent Neural Network????
  - capture dynamic information in sequential data.
    - through periodical connection of hidden layers.
  - it can store ,learn ,express related info of any length
  - unlike CNN -> it's not restricted by a space boundry instead->it can extend time sequences.
  - it's used in tasks related to sequences like; videos that contain image frames,clips,sentence consisting of words.

- RNN Architecture:
  - <img width="501" height="204" alt="image" src="https://github.com/user-attachments/assets/90c7a352-2976-4879-9a50-e694409b3885" />
    - Xt -> input sequence at time t
    - St -> memory cell of the sequence at time t (contain previous info(cells)
      - equation :  <img width="156" height="37" alt="image" src="https://github.com/user-attachments/assets/9a041152-ba14-4fe3-b05b-a0374422a48a" />
    - St contain multi hidden layers
    - output of Rnn is Ot -><img width="158" height="37" alt="image" src="https://github.com/user-attachments/assets/49663159-eb5e-476d-9bb0-a38b26e9f484" />

  - Back-propagation through Time?
    - convolution backward propagation -> extension of time sequence.
    - errors in memory cells are -> derivative of Ct with respect  with  t , t+1
    - the longer the time sequence -> vanishing,exploding gradient problem will happen.
    - Steps of Back-propagatino Through time:
      - compute output of each neuron.
      - compute error of each neuron
      - compute the gradient of each weight( using SGD Algorithm)
     
  - What are the problems of  RNN model?
    - <img width="658" height="84" alt="image" src="https://github.com/user-attachments/assets/5b6f411f-6c3d-4982-a09f-78252cf855d1" />
    - this model solves the problem of information memory however this solution not for a long term
    - so the long term memory attanuates at this .
    - so it has limited memory capacity .
    - To solve this problem in RNN -> we moved forward to LSTM model.
  - what are the types of RNN?
    - <img width="905" height="366" alt="image" src="https://github.com/user-attachments/assets/de084140-7788-4708-a4e5-0ffbbc883c3d" />
------------------
# LSTM 

-<img width="817" height="334" alt="image" src="https://github.com/user-attachments/assets/40ee1a2a-c188-4b1d-82ba-8627ba54f9d9" />

- LSTM Architecture:
  1.Forget Gate of LSTM:
    -  <img width="341" height="245" alt="image" src="https://github.com/user-attachments/assets/07015cb2-9aa2-4143-a4f8-af6169b839cf" />

    - this gate is used to decide whetther or not we will discard the information that enter the gate.
    - gate reads ht-1,Xt -> output either 1 or 0 (1=>info is retained now,0 -> info isn't retained yet (discarded) ) based on Ct-1
    - it did that for each digit that enter the gate.
    - note;
      - 𝜎  -> this is an activation function that 
    - <img width="314" height="63" alt="image" src="https://github.com/user-attachments/assets/fdad337c-e59a-4f50-be1e-59a6c4e1e969" />
         - this is the equation for that gate
     
  2. input Gate of LSTM:
      -<img width="344" height="270" alt="image" src="https://github.com/user-attachments/assets/93562dca-e3e2-4445-a3ec-af145ca38872" />
      - it has two parts-> 
          - input layer due to sigmoid function output It-><img width="304" height="44" alt="image" src="https://github.com/user-attachments/assets/4017bc5a-3ccb-4034-ac78-5032d2386e31" />
          - another candidate vector that generated by tanch function(Ct):<img width="357" height="48" alt="image" src="https://github.com/user-attachments/assets/2e26addb-8ecd-4307-9445-9f825da5e1f2" />

  3. Updating Lstm information;
       - <img width="330" height="242" alt="image" src="https://github.com/user-attachments/assets/649f906b-574f-4b8d-878e-6fb69543273e" />
       - this type => multiply earlier state(Ct-1 by ft)+multiply It by Ct(input value)
         - this is the equation: <img width="273" height="59" alt="image" src="https://github.com/user-attachments/assets/16bad481-8c90-475f-9b22-e0546f220294" />

    4. Output Gate of LSTM:
        - <img width="366" height="278" alt="image" src="https://github.com/user-attachments/assets/d6e59397-54b9-401e-a41c-876abdfbd9cb" />
         - we need  ht (output)
           - we  first find Ot by figuring out ht-1,xt
           - then we multiply that ot by tanh(ct)
           - <img width="305" height="87" alt="image" src="https://github.com/user-attachments/assets/283c2255-8500-483c-b969-a0e3ec043031" />


-------
# seq2seq
- What is the Encoder-Decoder Architecture?
  - <img width="710" height="178" alt="image" src="https://github.com/user-attachments/assets/0e6627a4-d4bd-44f2-8ec1-11b99ca9983f" />
  - due to that the output of the machine translation is a sequence of variable lengths.
    - so we need encoder,decoder to handle this sequence
      - encoder -> uses variable length sequence ->encode it to encoding state(with a fixed length)
      - decoder -> map fixed length encoding -> variable lenght sequence.
- What is the seq2seq?
  - input as sequence(encoder) --> output as a sequence(decoder)
  - <img width="679" height="226" alt="image" src="https://github.com/user-attachments/assets/36cbe355-1537-4253-8f4c-5a04e1669d37" />
    - encoding part is blue  ends with <eos> end of sequence.
    - decoding part is white starts with <bos> ->beginning of the sequence,<eos>->end of sequence.
   
- What does encoder do?
  - convert input sequence -> context variable c with  a fixed shape  .
  - <img width="201" height="61" alt="image" src="https://github.com/user-attachments/assets/8d5f7391-d1da-4794-be33-49c1332efabc" />
- What does Decoder do?
  - it do the reverse task that encoder did in order to generate the output in a sequence
  - it depend on the previous sequence from encoder,context  variable c , previous hidden state
  - this is the equation:
    - <img width="232" height="40" alt="image" src="https://github.com/user-attachments/assets/d80ed717-a33f-4a20-89dc-5a8c94c902bb" />

example on How LSTM uses decoder,encoder:
  -<img width="860" height="405" alt="image" src="https://github.com/user-attachments/assets/1c535e24-29c7-48bf-9639-06ef9fd10544" />

 Cons of LSTM:
   - however we used it to solve the problem of sequence and memory cell that was in RNN
   - but:
     - it uses one context variable C
     - C has limited length -> longer input -> will be compressed -> data will be lost.
       - as seq2seq it's base uses RNN .
     - seq2seq based RNN parallel computing is  lower -> Rnn when calc data at the moment it  needs data from previous -> so that takes time+low parallelism.
     -   seq2seq based CNN -> generate many parameters -> complex training.
     - so -> each word depened on previous word wait untill that previous one being output.
     - so storing data in a sequence inefficient.
    
  -to solve these problems we will move  forward toward Transformer .


------------
# Transformer mechanism

- what is the Attention Mechanism?
  - it's mechanism used to focus on important things + ignore minor things.
  - <img width="860" height="405" alt="image" src="https://github.com/user-attachments/assets/419a585d-4a37-495e-a224-0986804def60" />
  - What is the Attention principle?
    - attention can be used without encoder-decoder framework
    - <img width="562" height="208" alt="image" src="https://github.com/user-attachments/assets/aea188a3-fef0-4ed5-a7ee-04f011c9b851" />
- How the process of attention working?
  - <img width="461" height="418" alt="image" src="https://github.com/user-attachments/assets/1d020311-2dc5-4252-9d52-48ae40c426cf" />
    - 3phases;
      - phase1:
        - calc similarity between the query + each  key => scores  from s1->s4
        - convert the score into range from 0-1 -> thenn output a weight fo each one using softmax function.
        - use a1,a2...an -> build a matrix from them ->>then sum of them -> to get output value (attenuation)
- What are  the advantages of Attention?
    - less complex than RNN,CNN
    - needs less computing power.
    - support parallel processing(current output doesn't depend on previous one)
      - like CNN
      - unlike RNN(use sequence)
      - so this ability makes him has high speed.
    - better result (it can rememeber the results from the past)
 
  - What is the Transformer:
    - this model uses self attention mechanism.
      - in order to capture relative association at different positions of the input sequence.
    - it's uses to processing long texts..
    - it has  high speed, degree of parallelism.
    - it doesn't uses the structure of the RNN or CNN instead it uses decoder,encoder but with new features.
    - <img width="416" height="527" alt="image" src="https://github.com/user-attachments/assets/81b7dc51-fcf8-4aaa-9523-48918cc118de" />

    - <img width="577" height="57" alt="image" src="https://github.com/user-attachments/assets/82a6af3e-c752-4151-a426-ae6fe23ecba8" />
      - here as you see the source was english -> target was chinees (both objects are different)
      - <img width="367" height="278" alt="image" src="https://github.com/user-attachments/assets/6cd0ca2a-e653-4149-a06a-1c63d2a7fa38" />
        -as you see self attention calc long distance between words to deteremine relationships (whether or not this word (its)  refer to what...)
     
        
          her : it's related to law than application


--------
# Foundation Models:

