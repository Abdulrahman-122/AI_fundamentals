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
        - instant executation ->graph are  computed in real time
        - 
