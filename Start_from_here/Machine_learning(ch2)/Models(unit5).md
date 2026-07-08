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

  look at my notes here:
  
    
