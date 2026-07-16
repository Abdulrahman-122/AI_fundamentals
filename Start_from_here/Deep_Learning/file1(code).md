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

