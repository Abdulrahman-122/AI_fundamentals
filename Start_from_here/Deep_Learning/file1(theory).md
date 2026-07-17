# Deep Learning
This is how the traditional ML working?

  <img width="639" height="454" alt="image" src="https://github.com/user-attachments/assets/a4b8a419-70fa-4f47-b73f-c1e464141e25" />

```
part1:
1. locate the problem then choose ML model for it
2.then choose the data  in order to train the model on
part2:
1.Data cleanising-> you make data preprocessing in order to remove noise..
so you need;
  -feature extraction,selection,..
3.then start train the model
4.then start see result

so this   way is manually so we need to figure out how to make it automatically using DL models.
```

- Introduction to DL:

  <img width="513" height="463" alt="image" src="https://github.com/user-attachments/assets/64bfefda-6149-4a2d-8d7d-8a9fea32f46a" />

```
what is  Perceptron?
  - it's a node that sum signals inputs with their weights
  - output 1 if the summation exceeds specific threshold
  - if it didn't exceeds -> it will generate the summation of them anyway.
  - this nueron or node contain activation funtion to do the summation (sign(x) ..)
  - it's the first practical application of Artificial neural network.
  - this nueron is the basic block of image recognition
  - it has two signals (1 current value) ,(0 no current value)
```
Equation of Perceptron:  
<img width="569" height="176" alt="image" src="https://github.com/user-attachments/assets/911e9f7a-f270-4356-a726-3c131b96a18c" />
```
note;
if signal has high weight -> that means it's important signal from the prespective of the perceptron.
in the equation above:
1 -> means important signal
-1 -> not important signal
```
now why we used the perceptron instead of traditional MLs?

<img width="546" height="425" alt="image" src="https://github.com/user-attachments/assets/fd521597-e870-4f0e-b075-acb844e5629d" />

    = as we faced the XOR probelm  so we can't solve it using single perceptron that ML was build on it.
    = now we need multi-layer perceptron to detect the XOR equation and solve it correctly .
so this is Multi-layer perceptron;

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e7460d0e-367e-44f6-aa69-758a17edd477" />

How is the  Multi-layer-perceptron  working?

  -   <img width="1596" height="544" alt="image" src="https://github.com/user-attachments/assets/29c15c9e-569a-4de8-9eb4-a883a0c29b52" />

  - <img width="1635" height="551" alt="image" src="https://github.com/user-attachments/assets/bf6c4a1f-9437-424b-9da1-31d81ac71ac6" />

```
you see the next image above ;
first
  - we need to calculate the summation of inputs each input multiplied by it's weight
    so we got the linear equation
  - not to figure out the XOR problem we need some linear thing to solve it for us => so we used  Activation function  as it output 1 if the output > 0 , -1 if output<0

```
this is the shape of using multilayer perceptron with multi sign(x) functions(activation func)?

  - <img width="857" height="286" alt="image" src="https://github.com/user-attachments/assets/3ec3a1ae-ae00-4968-9301-d67767778f1b" />
  - this is the output from this diagram:
    - <img width="598" height="217" alt="image" src="https://github.com/user-attachments/assets/4f72929f-8346-4d04-83db-4f68456d89b4" />

to rap up this section: MLP(multi-layer perceptron) -> has the ability to solve complex problems by adding multi layers of perceptrons(nodes) to detect that problem.


---
What is the Deep learning?
<img width="467" height="607" alt="image" src="https://github.com/user-attachments/assets/40523c76-0821-4267-906f-cca93ba1db85" />

```
it's a subset of ML that trying to solve complex problem by simulating human brains or neurons
it's called; Deep neural network science it depends on : neurons functions,connections among neurons.

```
Model types in Deep learning?
  1. FeedForward Neural network :
     <img width="952" height="496" alt="image" src="https://github.com/user-attachments/assets/da06984b-2ed9-4897-ab9b-443dc39586f5" />
     ```
      this model has the following:
       1. b at each layer -> threshold that activate a neuron
       2.X -> are the inputs
       3.y -> output of each layer it's simpley the summation of all previous input by it's weight (x1*wx1 or y1*wy1)
       4 at the hidden layers every neuron take input from previous layer => output the result using Activation function inside it to the next layer...
       5.this type is unidirectional -> signal go in one way no feedback or whatever.
       
     ```
notes:
1.
- <img width="62" height="51" alt="image" src="https://github.com/user-attachments/assets/32459325-da98-4667-ae44-dc7503185263" />
- <img width="260" height="366" alt="image" src="https://github.com/user-attachments/assets/5a5e4814-996c-447e-9fe9-6e52acb66404" />
```
from the image -> this weight for node 4 at layer (hidden layer) connected to layer 0 at node 3
```
2.
- <img width="925" height="626" alt="image" src="https://github.com/user-attachments/assets/f0005b29-39b1-4bdb-a874-0b14294867f5" />
- <img width="561" height="233" alt="image" src="https://github.com/user-attachments/assets/4d3a41a4-4ead-4013-bd49-25fde51554c5" />
```
1.as you see the output of the first layer going to node a1 then node a1 make the summation  for the previous layer mulitplied by their weights then add the bias(b) to put threshold    for that neuron(node)
2.y take output from a1 then it will find it's activation function using f(a1) as you see above
Don't worry about activation function for now just understand the whole picture first.
```
this is the difference between hidden layers ?
  - <img width="1277" height="414" alt="image" src="https://github.com/user-attachments/assets/d10a0189-940f-4a6b-b904-997bb56b71fb" />
  - as you increase it it will solve more complex  problem.

Q: What if we added more layers+ more nodes what will happen?
```
(More Nodes)
 pros:
  = Captures more features at the same level Memorization & 
cons:
  = OverfittingDepth 
(More Layers)
 pros:
  = Builds complex abstractions hierarchically
 cons:
  = Optimization failure (Vanishing Gradients)
  ```
----
Now we now that
- in order to make model learn more problems we need activation function.
  - types of Activation function:
      1. Step function (sign function)
        -   <img width="744" height="417" alt="image" src="https://github.com/user-attachments/assets/fbfd7c91-003f-4b9a-925b-0a480ba958f1" />
        
     2. Segmoid function
          - it's a frequently used function in Deep learning.
          - it maps input that has (- infinity,infinty) to (0,1) values
            <img width="744" height="417" alt="image" src="https://github.com/user-attachments/assets/eaec076a-b4a0-4bf5-aac3-a7256164b455" />
            
          - equation:
              <img width="549" height="125" alt="image" src="https://github.com/user-attachments/assets/75efa4e8-db24-4944-b332-18746ef8b5b0" />
              
     3.Tanch function;
            -<img width="666" height="518" alt="image" src="https://github.com/user-attachments/assets/eb8dcd6c-4df9-4b86-bd1c-05768e41508e" />
            
       - <img width="403" height="105" alt="image" src="https://github.com/user-attachments/assets/aca8fc30-a0db-4d1e-9c14-e643a41faa70" />
       
        - it maps input between (-1,1)

  4.ReLU function (Rectified linear unit):
    - <img width="1517" height="383" alt="image" src="https://github.com/user-attachments/assets/dee92bb1-c60f-4241-81c0-982f180ea5ea" />
            
            - as you see it maps input to (max(0,input),0)
      
 5.Leaky ReLU function:
  
   - <img width="903" height="582" alt="image" src="https://github.com/user-attachments/assets/7fe8c37e-3dd2-450c-94f6-fd8b50ce8fb4" />
  
            - it assigns a non negative slope to all negative values.
            - enhanced function from ReLU funciton
     
 6.Swish Function:
 
  - <img width="641" height="516" alt="image" src="https://github.com/user-attachments/assets/2d83dd0c-f10e-4a56-94e8-720310c0de9f" />
            
     - equation:
     -   <img width="397" height="72" alt="image" src="https://github.com/user-attachments/assets/bb2e6464-2793-4fb9-8fc3-dde4a38436ad" />
        - it maps input between (0,1)
        - B is the learnable parameter
        - note:
            -  it's called self gate function;
            -     as if alpha(x)=1 -> ouput=x
            -     as if alpha(x)=0 -> output=0
            - it's like ReLU function
            - it may considered non-linear interpolation function(between linear+ReLU function-> you control that non-lineariry using bias that put threshold for neurons.
                 
   7. softmax function
      
   - <img width="686" height="455" alt="image" src="https://github.com/user-attachments/assets/96be87a9-01b2-406f-ad6f-21b8b4ed0815" />
   - <img width="281" height="128" alt="image" src="https://github.com/user-attachments/assets/e63f7026-526e-4d02-948e-86bccb36f37b" />
   
                   - it's used for multi-classification problems.
                    - map inputs between (0,1)
                    - the output of this function is interpreted using probability distribution  to deteremin different classifications.

----
How we can train DL models?
   - first recall that -> ML models needs you as an engineer to make a lebels for them in order to extract all of these features from the image or whatever
   - howeever Deep learning extract features from image based on problems he solved before

let's deep dive ....

- <img width="1513" height="562" alt="image" src="https://github.com/user-attachments/assets/16163528-77e3-48c6-9de0-15aae00766d6" />
  - based on that -> training set -> put the feature that you want model to work on top of it
  - then start validate by using validating set  to identify whether his performance at that thing or not.
  - <img width="1325" height="290" alt="image" src="https://github.com/user-attachments/assets/a968741a-a2af-4262-a527-6a51f88cae7f" />
now how to calc error and try to improve the model?
  - we uses loss function to calc error.
    -   by knowing the difference between predicted value+actual value  in both ML,DL
    -   so you can know the performance of the model.
    -   types;
        1.  Main square error:
        -   used for regression function;
            - <img width="368" height="100" alt="image" src="https://github.com/user-attachments/assets/d9c9802e-5e32-40a9-a999-ac0be944dab8" />
            - yk=> output of neural network
            - k => number of data dimension
            - t -> supervised learning data
        2.Cross entropy loss function:
            - <img width="359" height="108" alt="image" src="https://github.com/user-attachments/assets/5ca83aae-4a43-409d-84d1-5dffdd9bbda6" />
    
                  - y -> output of neural network.
                  - tk => correct solution label
                  - it's used in classification problems.
             
What is the Gradient Descent Algorithm ?
<img width="500" height="409" alt="image" src="https://github.com/user-attachments/assets/973a2019-0181-44c7-89c8-4737eb680a14" />

    - it's a way that we uses to make model more optimal + more less error so that this Gradient decrease loss function.
    - as the equation of Gradient increase -> the model is more optimal but not all the way there's a limit
    note;
      - this type of Gradient is no longer used due to -> it's convergence(movement)-> optimality is very slow as all training samples needs to calculated again once weight is updated.

  so We moved to Batch Gradient Descent Algorithm?
      how is  this algorithm working?
        = <img width="794" height="239" alt="image" src="https://github.com/user-attachments/assets/9cbad691-a471-4b03-80ce-2a957914cee7" />

          - X  this is the input
          - t -> target output.
          - o -> actual output
          - eata M -> learning rate.
          however ;
            - this algorithm uses the entire dataset at each time we train the model on.
            - each update is made toward correct direction 
            cons;
            - take learning time + more memory resources.
  so we moved to Stochastic Gradient Descent Algorithm(SGD)?
     - <img width="800" height="123" alt="image" src="https://github.com/user-attachments/assets/e51ac2c0-f244-473a-9ae1-9fa44b99f108" />
     - this model solved the probelm of BGD where it was using  long time+ more memory to update the weight.
     - now this Algorithm uses sample by sample + try to update weight  based on them.
     - look the algorithm;
     -  <img width="1016" height="311" alt="image" src="https://github.com/user-attachments/assets/50188b76-dd46-4b18-924c-fb61bebdfa93" />

          -initialize small random w (weight)
          - then select random sample 
            - calc ouput based on it 
            - then update weight.

  Then we moved toward mini Batch Gradient Descent Algorithm ?
      - this is the  Algorithm workflow.
      - <img width="1643" height="538" alt="image" src="https://github.com/user-attachments/assets/fc1e0ef7-49f0-4d49-a43a-741aeb765634" />

          - first this type uses small size of samples to train model on 
            - then initialize w to 0
            - then find actual output (o) based on input X
            - then find weight
            - then update the whole weight.
            - this type is the mostly used + batch size  is usually set to 32.


------
What is the Netwrok Training Process?

```
simply we uses that process to calc the output function by using chain rule;

```
- <img width="711" height="212" alt="image" src="https://github.com/user-attachments/assets/6dbc42fc-2489-4913-bc27-4e59dcb8817f" />
- that's the chain rule;
  - <img width="239" height="45" alt="image" src="https://github.com/user-attachments/assets/d6f28c47-5984-474a-9e97-b15f264c0f2b" />

What are  the Vanishing and Exploing Gradients?

  - Vanishing Gradients
    -   Derivative value of backward propagation reduce as the number of  network layers increases.
  -   Exploding Gradients:
    -     Derivative value of backward propagation increases as the number of network layers increase 

notes;
- these are errors happen due to using segmoid function  due to using large network(many layers +many chain rules like this
  - <img width="369" height="96" alt="image" src="https://github.com/user-attachments/assets/1bc149be-e52a-4c8e-8aea-81b00846d43f" />
- to solve Gradient Exploding -> use Gradient clipping (in order to clip the value of Gradient once it exceeds a threshold    )
  - you can use weight regularization L1,L2
  - you can use Relu function (if the  derivative of the activation func is 1 -> gradient vanishing+exploding will not occur)
  - you can also use LSTM(long short  term memory )

to rap up:
  - to solve the problem of gradients use(Relu or regularization terms or LSTM)

-----
What is the optimizers?
  - special methods we used to make neural  network most faster.
  - Why are the optimizers used in OOP algorithms?
    - Accelerate  algorithms
    - Preventing+overshooting local extreme
    - simplifying manual parameters (learning rate)
    - ex;
      -   SGD optimizers
      -   AdaGrad
      -   Adam ..
      -   Momentum .
  - What is the momentum Optimizer?
      - <img width="363" height="40" alt="image" src="https://github.com/user-attachments/assets/e101dd50-af50-42e6-9758-b69833cc1901" />
       - this optimizer added the second term to this equation (momentum term)
         - alpha α is called: momentum coefficient.
         - j -> neuron at first  layer ,i -> neuron at second layer.
         - pros:
             -   enhance the stability of the gradient +  reduces mutations.
         - cons;
             -   𝜂 , 𝛼  => these parameters will  need to be set and it will increase the parameters of the model.
    - What is the AdaGrad Optimizer?
      -  the bases that optimizer is built on -> to get different parameters you need to update learning rate which against momentum,SGD optimizers..
      -  these are the parameters that must be added:
        -  <img width="656" height="216" alt="image" src="https://github.com/user-attachments/assets/fc1d047a-a860-4f89-81d8-86a68da3c5d0" />
        - from the image above:
          - r continuously increase  => overall learning rate decreasing => number of updates will increase => we get closer to the optimal solution(learning spead slow)
          - 
      - Advantage:
        - learning rate is  slow =>updates increases =>optimal solution
      - Disadvantage:
        - as learning rate got slower everytime -> so the efficientcy of the algorithm will decrease.
       
    - What is the RMSProp  Optimizer?
      - it's an improved version of AdaGrad optimizer.
        - it addes attenuation factor(𝛽) to the equaiton of square gradient accumulation.
        - <img width="584" height="185" alt="image" src="https://github.com/user-attachments/assets/d5a19072-2a83-42a3-8056-f7a2a2f5146e" />
        - <img width="584" height="185" alt="image" src="https://github.com/user-attachments/assets/1e4e865d-7f48-40a1-bb31-7f9e0be54147" />
      
               - this help adaGrad to end his optimization process too  early.
              - used with RNN(recurrence  neural network)
              - Disadvantages:
              -   all of these factors need to be set at the process.
  - What is The  Adam Optimizer??
      - it's based on AdaGrad,AdaDelta
      - control mt,vt they are called -> gradient's first moment+second moment(uncentered varience)
      -<img width="336" height="87" alt="image" src="https://github.com/user-attachments/assets/ef5d716d-c1b5-42d1-8925-2b6173b1d172" />
      - then we take the  derivate for them as mt,vt are both close to zero when B1,B2 =1
      - so we got that <img width="194" height="120" alt="image" src="https://github.com/user-attachments/assets/f5bd4294-5bd0-4d55-b412-5dd21c9bdf19" />
      - and this is the weight for it:
        - <img width="250" height="80" alt="image" src="https://github.com/user-attachments/assets/82768c5f-3efc-44cf-b6b2-42b7e9ead525" />
      - so at the  process of using this optimizer (initialization process) ->B1=0.9 ,B2=0.999 , E=10^-8,M(learning rate)=0.0001
        - so that  learning rate  will be reduced at saturation level  (stanard optimizing)

----

Why can't the model learn very well especially when we test it with test set?

  - this problem due to normalization branch of the model behind optimizations techniques aren't set  well.
  - <img width="830" height="231" alt="image" src="https://github.com/user-attachments/assets/9294630e-e520-4b18-963f-b600c3a91245" />
How actually overfitting occurs?
  - due to -> small training set+  complex model so that model learn  the dataset many times  that cause to  learn noise inside that dataset =>then overfitting  occured .
  - how to solve this problem?
    - obtain more data + do preprocessing over it
    - user a  proper  model
      - reduce  network layer+neurons to  reduce the happening of fitting.
      - Regularization -> it's done  by reducing the  weight that is  increasing during training
      - reduce the training time(make evaluation test)
      - Data cleaning+pruning.
      - uses Bagging or Boosting (multi-model)
    - Type of Regularizations?
      - L1 Regularization.
        - this way we is used to restrict increasing of thee weight.
        - <img width="231" height="270" alt="image" src="https://github.com/user-attachments/assets/3a6bc420-e150-47ac-a7e8-abdfc69390b6" />
        - <img width="322" height="48" alt="image" src="https://github.com/user-attachments/assets/e94072c7-71aa-4182-a7fa-6f42cf79341a" />
            -  we added 2nd term in order to restrict the  weight (called absolute of w)
              -   <img width="334" height="50" alt="image" src="https://github.com/user-attachments/assets/51d86f7e-3244-491e-83ea-d71e921bf6eb" />
              - another  equatiion for the weight;  <img width="302" height="66" alt="image" src="https://github.com/user-attachments/assets/1f81fc0d-b744-43ba-9397-03228b2a92b0" />

              - note ;
                  - this L1 reduced the term of w   to 0 parameters which is good in  order  to restrict  the weights or the  parameters that model  learned

      - L2 Regularization.
        - <img width="278" height="205" alt="image" src="https://github.com/user-attachments/assets/56dde893-ce4b-4709-9d2d-d8aa3faae4f1" />
        - however we added norm penalty to this equation (2nd term) to reduce the weights but this is better than L1 Normalization as you see the curv intersect with many weights so that the number of reduction is lower than L1 .
        - <img width="373" height="54" alt="image" src="https://github.com/user-attachments/assets/e3a772d5-2565-43dc-8c18-e27d90ddb0fe" />
          - this is overall value
        - This is the output  weight ;
          - <img width="302" height="48" alt="image" src="https://github.com/user-attachments/assets/683b9ae9-d796-4711-adf3-237fd29276ac" />
            - this equation depends on E -> learning rate .
           
    - As we discussed above the L1,2 norm is good for make learning rate of the model good
    - we can add this feature to restrict the model:
      1. Stopping Training Early:
        - it's to track the model once it get to a point where the training error  start to decrease but the testing one increase -> you  need to stop it
        - then solve that   problem  in  order to make it good at testing +  avoid overfitting.
        - <img width="478" height="302" alt="image" src="https://github.com/user-attachments/assets/10ff35ee-4d7c-4910-831d-a6339b656a6c" />
      2.Dropout:
        - <img width="669" height="304" alt="image" src="https://github.com/user-attachments/assets/2b576d7e-3fd5-4f96-b25f-781fe5186daf" />
        - you start  by divide the model into sub networks
          - then see the subnetwork that needs to update then update it's parameters and at the same  time stop updating the other subnetworks.
          - then combine the whole subnetworks at the ends with eachother .
            - the process;
              - pick a random sample from  the hidden layer.
              - delete the other neurons temporarly .
              - go through that sample => update their parameters => store them
              - then return the deleted neurons
              - start again with  another random sample
             
            pros:
              - effective process.
              - can be used with non-deep learning models
              - cheap
             cons;
                - not good with insufficient data.
        3.you can use adversarial trainging,multi-tasklearning,ensamble learning,parameter sharing,semi-supervised learning to avoid overfitting....







      
