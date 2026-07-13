# Supervised_Learning

  1. Linear Regression(polynomial linear Regression):
      - <img width="447" height="252" alt="image" src="https://github.com/user-attachments/assets/805ea960-48ac-40bd-a75f-6ac78355f028" />

      
            - Regression -> represent relation between two or multi variables in linear way like the image above.
              - also when a something has many factors this means  regression
              ex; an apartment has  factors like; area,price,size...
look at my sketch:
  ??


Polynomial Regression what does it mean?

<img width="495" height="240" alt="image" src="https://github.com/user-attachments/assets/56f64ab4-4809-4c74-8340-19a45da5fd61" />

    - this type of regression we use it in order to solve the problems of polynomial function that isn't in linear way like this.
    - so that the dataset of it will be complex.
        - also, if you used linear regression at these types of dataset-> underfitting will          happen .
        - this is the equation(complex):
  <img width="333" height="43" alt="image" src="https://github.com/user-attachments/assets/bd81ebc5-2be2-455f-afaa-dc7ac9af0dbd" />

How to prevent overfitting from linear Regression or even polynomial one?
  - you will add another term to the loss function called:Regularization term(L2-norm)
        at this case :linear regresssion called( Ridge regression)
        <img width="294" height="50" alt="image" src="https://github.com/user-attachments/assets/0b45b11d-e417-43a2-ae03-9734ac95feb9" />
       
 - there are another one called without square(use absolute) uses in Lasso regression)
        
  - <img width="323" height="47" alt="image" src="https://github.com/user-attachments/assets/42ad346f-9475-44ae-bfaf-f78f90253f30" />

---
What is the Logistic Regression?

    - used in linear+non linear regression 
    - linear regression -> solve binary classification
    - non linear regression -> solve multi classifications
    - classification model (resolve classification problems)
    - overfitting in this type can be prevented by adding regularization term in loss 
    function (look at the notes)
- <img width="464" height="69" alt="image" src="https://github.com/user-attachments/assets/528350f1-55a1-4ab1-94c7-4d513f6f1746" />



- as you see there are two types (one with binary class (linear Regression),one with multi class(non linear)-> depend on softmax function)
  - <img width="756" height="305" alt="image" src="https://github.com/user-attachments/assets/8b670370-2167-4cd3-8202-8c78f9657b52" />
     
  -explain some info about softmax?
<img width="757" height="252" alt="image" src="https://github.com/user-attachments/assets/a8c5adb7-0c0f-4494-a49f-5635456d878b" />

      -it's uses a generalization term for logistic regression.
      - compresses dimensional vectors from real values to another  values locate between(0,1).

  look at my notes here:



-----
2. Decision Tree:

- <img width="453" height="277" alt="image" src="https://github.com/user-attachments/assets/38d29523-b55a-4762-8afd-6dd55c1184b2" />
<img width="553" height="383" alt="image" src="https://github.com/user-attachments/assets/3fd2863e-dcd1-4ff5-8539-3a91cbfafee2" />

      ```
        - Any non-leaf node -> represent a test
        - Any leaf node -> represent a class.
        - branch -> represent the output of the test
          the loop going from the root node (1st test) till leaf node(output class)
        - Learning algorithms -> generate a descision tree with curt+ID2
        
      ```
  How is Decision Tree working?

      -it divide data of all features -> compare all result set according to purity.
      -the data with highest purity is the selected one.
      - we measure purity using;information entropy+GINI coefficient.
        look at my note for the equations:
      - this type is classification .
  notes:
  ???


What is the process of constructing a Decision tree?
  - <img width="855" height="321" alt="image" src="https://github.com/user-attachments/assets/a8cd6e7f-edb0-4cfd-8ced-552d748bb4f3" />
      

        1.feature selection-> select one of the features of the training data
        2.Decision tree generation -> generate subnodes from top down based on selected feature+stop when no split 
        3.Pruning -> pruning the decision tree in order to reduct size and to avoid overfitting.
          (pruning include:pre-pruning+post-proning)
    
    
---
3.Support Vector Machine(SVM):
    - <img width="518" height="204" alt="image" src="https://github.com/user-attachments/assets/f1cc0eec-f236-4c8c-8374-07cbe78d2506" />
    -<img width="415" height="180" alt="image" src="https://github.com/user-attachments/assets/b090a3e8-0cca-4572-89a9-984a737f2b90" />


        - this type is used in classification problems.
        - used with linear classification (binary classification)
        - also it used with non-linear classification using kernek trick
        
 How it's working?
   1.Linear SVM
   - <img width="751" height="218" alt="image" src="https://github.com/user-attachments/assets/53400ee2-b844-44d7-98b8-8872bc522c3b" />

    - this type uses fixed line + keep the nearest point as far as possible from the line
      - the nearby points called ; support vectors.
        so the philosphys is -> keep 2 support vectors on max distances from  the line.
    - note; in 2D we use straight line while in high dimension we use hyperplane 
  2.Non-linear SVM
     -<img width="399" height="314" alt="image" src="https://github.com/user-attachments/assets/df0dca79-f4fb-43f2-87ca-41d4d6259499" />
      - output;
        -<img width="471" height="261" alt="image" src="https://github.com/user-attachments/assets/34133a52-29c2-40d8-b497-9d88039bc3a0" />

              - so SVM uses kernel function to be able to label the classes in this dataset.
                - kerner -> Guassian kernel(the most used)
              types of kernel functions?
                1. Polynomial kernel.
                2.Linear Kernel
                3.Sigmoid kernel
                4.Gaussian Kernel.
----
4.K-Nearest Neighbors Algorithm?
   - <img width="384" height="347" alt="image" src="https://github.com/user-attachments/assets/f5e455b0-1b86-446a-b140-dcb790fbbfc5" />

          - if the closest samples to the current sample that we looking for it's class related to a category then that sample will belong to the same category.
           - ex: the ? the most closest category to it is triangles then this ? will belong to them if we enlarge the class to include th squares then it will belong to them as there are 3 squares vs 3 traingles.
         - Knn is a non-parametric method + used with datasets that have irregular decision boundries.
         -knn => uses majority voting to predict classification
               = it uses mean value to predict regression
         - requires very large amount of computing.
<img width="320" height="265" alt="image" src="https://github.com/user-attachments/assets/0c7eceaa-faff-4ca9-9600-d3acd7a5db74" />
     

           -From the image above => it's successeded to extract the green pixels inside the blue ones so this algorithm is efficient.

notes;
  - <img width="507" height="472" alt="image" src="https://github.com/user-attachments/assets/4bbffaf9-9f87-48e8-adb0-eb12b7af7b30" />

    - as you see;
      -   as k increase => impace of losses will be reduced but the difference between classes will be less obvious
      -   so increasing K =>underfitting will increase because divisions between classes is more hard
      -   so decreasing K => overfitting will increase because divisions between classes is now refined.
     

----
5.Naive Bayes:
    
