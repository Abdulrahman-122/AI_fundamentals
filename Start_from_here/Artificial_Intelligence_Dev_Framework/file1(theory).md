# How these frameworks are working??
- frameworks like: Pytorch,TensorFlow
```
they provide the following features:
1. Data pre-processing
2.Development APIs
3.Debugging and tuning
4.Compiling and execution
5.inference deployment
you can summarize it in these :
preparation -> model development -> Training evaluation -> inference deployment
so the other advantage of using these tools:
1. Accelerates scientific reasearch(provides APIs)
2.Enable high performance training
3.Make deployment easier.
Another advantages:
- it contain models that are already working so you can use it in your building process.
- it doesn't obsessed you with the underlying infrastructure.
-  it uses tensors to compute your code by abstrcts it into computational graphs=>improve computational efficiency.
- 
- 

```
- this is how these frameworks convert code->computational graphs
  - <img width="943" height="258" alt="image" src="https://github.com/user-attachments/assets/2fae734e-1dd6-4659-a8a8-5ac85a4f4388" />

- What is the Pytorch framework?
  - frmework released by Meta
  - based on Torch(tensor operation library optimized for deep learning using GPUs,NPUs.)
  - it's characteristics?
    - support python
    - dynamic neural network -> programs can be working using dynamic computational graphs.
    - easy debugging.
    - has an active community.
- What is the Tensorflow framework?
  - it's an end-to-end ml platform by Google.
  - provide multi-Apis-> in order to build the models like: Keras API
  - provide you with Distributed strategy API so that you can build the model on multi-platforms infrastructure.
 
- Some basics about Tensorflow tools?
  - tensorflow.js -> library for deploying models in javascript on browser
  - tensorflow lite -> open source  framework ->deploy ML models on both mobile+IOT devices
  - Tensorflow extended(TFX) -> deploy models that you trained in research phase.
- some basics about JAX framework?
  - high performance library made by Google
  - combines -> Numpy,Autograd,XLA compiler,Tensorflow.
  - pros:
    - easy to use ->minimize encapsulation+minimize redundant operations.
    - High performance as it contain XLA-based optimization+GPUs,TPUs
    - flexible and easy to use with other frameworks
  - cons:
    - weak community compared with Pytorch,tensorflow.
   
- What is a tensor:
  - it's the basic block of AI framework -> All data is encapsulated in tensors
    - types:
      - <img width="423" height="279" alt="image" src="https://github.com/user-attachments/assets/f0c05593-6f32-47fb-834d-dd3f9f132fb5" />
      - zero-order tensor  ->scalar
      - first-order tensor ->vector
      - second-order tensor ->metrics
      - ....
    - some attributes:
      - shape -> C | H | W...
      - Rank(dimension)=>0 or 1
      - Data type => bool,float...
      - storage position =>GPU or NPU or TPU.
      - name -> tensor identifier.
    - Supported Data Types:
      - float32
      - float16
      - int8
      - bf16
      - complex + bool
    - supported Data loading:
      - these data loading allow opening datasets like : Mnist...
      - data loading like: torchvision...
    - Data Format After Loading:
      - basic format ->[batch_size,sequence_length,feature_length]
      - in pytorch -> [N,C,H,W]
      - ....
     
    - what is TFRecord:
      - <img width="173" height="227" alt="image" src="https://github.com/user-attachments/assets/0a5f2f42-b408-4e89-af0f-214ceb335320" />

      - this allow storing data in a group in order to solve  the problem of huge data
        - so it make serialization  for data in order reduce disk I/O + network 
    - According to the Model file:
      - Inference model  -> support two types  of data -> training parameters+network models according to the file-format:
        - checkpoint -> uses Protocol buffers format to store -> parameter  values on the default network.
                  - also -> it can store the model structure
        - .ONNX ->open neural network exchange => general expression for models.
        - bin -> save+load models  and data.
        - pt -> file format for pytorch in order to load a complete pytorch model.
       
- What is a Computational Graph?
  - it's a way used to describe computational process in deep learning models.
  - like adding a nodes for multiplications,addition....
  - all of these helps you to analyze the entire stack.
 
- What is the composition off the Computational Graph?
  - tensors + operators are the building blocks of it
  - <img width="646" height="133" alt="image" src="https://github.com/user-attachments/assets/19156a0e-000c-442b-bb34-086c36cbfe75" />
  -  operator connected with edges in order to describe the tensors.


- as we mention before
  - computational graphs -> are a way to represent  neural network in a graph way in order to analyze it.
    - <img width="443" height="298" alt="image" src="https://github.com/user-attachments/assets/e0773e1b-9777-43bc-ae1e-04ba955207ca" />

- What is the difference between Dynamic/Static Computational Graph?
  - static graph:
    - graph structure of the graph is generated -> then computation operations involved in the graph.
    - compiler uses technologies->to optimize graph+achieving high executing performance.
  - Dynamic  graph:
    - program is executed in the  coding sequence.
    - the reverse diagram is generated based on backpropagation principle.
    -  difficult to optimize but  it's good for generating graphes .

- Pros+cons of Dynamic ,Static  Graphs:
    - Dynamic Graphs:
      - pros:
        - instant execution ->graph are  computed in real time
        - flexibility+it support varies data sizes+able  to  analyze complex operations.
        - easy to debug.
        - research friendly -> research uses it to iterate  over the experiments.
        - easy to  understand.
      - cons:
        - dynamic  graph may use extra overhead -> lead to  lower performance while static one don't make this.
    - Static Computational Graph:
      - it's defined before running.
      - pros:
        - predefined ->the entire graph needs to be  running before execution.
        - optimization  -> as  this graph is fixed so we can optimize it (memory allocation,parallel computing)..
      - cons:
        - require more complex coding ->when complex flows  are included.
        - less  initiative than dynamic  as  it is  defined before running.
  
  - Computational Graphs in both Pytorch,Tensorflow:
      - Pytorch -> dynamic graphes are used by default.
      - Tensorflow ->..............................
   
-  the end of  this course:
      - some words for the last of  the pdf may benefit you related to pytorch;
        - torchvision->tensorflowlibrary are used for computer vision .
        - Data Loading -> data is loaded using  dataset+dataloader .
        - Methods to  construct datasets  ->torchvision   used  to     construct it (datasets) like:MNIST,CIFAR10,100,ImageNet.
       
        - Model   construction.
        - forward computation.=>model calc backword  propagation  based on gradiend+forward to compute
        - two-dimensional convolution layer 
        - Max Pooling.
        - Recurrent Neural Network.
        - Activation functions with pytorch like (softmax,Relu,LeakyRelu...)
        - Loss functions ->(L1 loss,Mean square error loss,Cross entropy loss,Binary cross  entropy loss,....)
        - pytorch   optimizers -> SGD ,Adam,AdaGrad,AdamW,RMSPROP
        - image  classification   depend on supervised   learning .
        - <img width="940" height="253" alt="image" src="https://github.com/user-attachments/assets/3824fde0-9c18-4af2-96a9-ea8f9b358cd8" />
          - pre-processing-> clean  data...
          - Model training ->
            - train model   on dataset from scratch
            - or pre-trained model ->bring pre-trained model+train it  on this dataset.
            - then Model loading+Prediction
            - <img width="451" height="395" alt="image" src="https://github.com/user-attachments/assets/235aeb07-a1f6-4af3-99b0-63c1d6c8aab0" />
              - now  after training ->Model deployment.
              - then: Model  compression in order to reduce the size of the  model as best as it's can.
              - How to  compress model:
                - using model pruning ->  in  order to reduce the the noise while calculating  the new parameters.
                - using Knowledge Distillation:
                  - extract useful info from the complex model -> then after training it will build small  network  connected with complex system.
              - Common deployment tools:
                - LIama.cpp
            - Tensor  parallelism(to make parallelism  at tensor -> you   can use  these two way):
              - Row-wise  weight
              - column-wise weight
            - This is how pipeline  parallelism look like:
              - <img width="904" height="279" alt="image" src="https://github.com/user-attachments/assets/d8b88dbd-b36e-4960-b7f3-b24dca90dfca" />
            - Supervised Fine-Tuning (SFT);
              - pre-trained model is trained  using new  dataset->then new dataset  will generated.
              - then we make the output layer for this new dataset  and then  we train the target  model  to the output layer.
              - other layers are fine-tuned   using  the source model's parameters.
            - What is the Adapter tuning?
              - you train model on   specific part and make the  other parts frozen ->this lower the  costs of  computing powers.
              - <img width="327" height="319" alt="image" src="https://github.com/user-attachments/assets/02d68457-fcd5-4111-9a30-81f028de2fa3" />
            -  What  is the prefix tuning?
              -  input  we use it's tokens  and then update it while  the othet token frozen.
            -  What is  the Prompt Tuning?
              -  
