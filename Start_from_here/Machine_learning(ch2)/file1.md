# Machine Learning(ML)

#unit1
intro:
  <img width="576" height="118" alt="image" src="https://github.com/user-attachments/assets/1dd70996-c4b2-4d8f-9eed-9f4a439cede3" />

      - Computer learn and has Experience based on:
        - Task T =>  how machine learning will process the sample
        - Performance P => accuracy+error rate.
        note:
          -Experience -> samples that we give it to the model in order to be trained based on rules for each model
              in ordre to avoid overfitting or underfitting...
        
      ex:
        - if you have a book and you want the model to learn it:
          - Task -> Learn the book
          - Performance -> classification accuracy.
          - Experience E->  Classified sample library.


-----
How ML works?

- <img width="331" height="276" alt="image" src="https://github.com/user-attachments/assets/7762f3d4-20ad-4a57-871e-a75bcdf488ee" />

      -you pass it (historical data+new one) then it will predict the future .
  Ml vs Rule_based_method(traditional programming)?
      <img width="649" height="245" alt="image" src="https://github.com/user-attachments/assets/4bbaaf04-1768-44ba-9752-241d3ecd6100" />

      - Rule-based-method;
          - you configure it manually (everything has a rule for it)
      - ML
          - you pass it (historical data+new one) then it will predict the future
                -  learn automatically.
  When to use Ml?

    - <img width="561" height="302" alt="image" src="https://github.com/user-attachments/assets/090d91fb-92e7-4544-85f1-df756265bfda" />

          - use it when  working on: complex problems like ; speech recognition...
      

What type of problems ML can solve?

    - Classification -> predict output based on specific categories(classes) -> results are (discrete (1,0) true or false)
    - Regression -> predict based on classes => result are (continuous values)
    - clustering -> predict output based on the similarities.
    

------ 
# unit2

What are the types of ML?

    1. Supervised learning?
       - model are trained on samples or specific data -> generate predictions based on mapping inputs to outputs 
       - so it can classify unknown data.
       - you give it sample+standared answer(you draw the structure that he will go from) -> it try to configure it's parameters=>generate answer
    2. Unsupervised learning?
       - model are built based on highly similarities in data+unlabeled data.
       - it calculate similarity between new,existing data.
       so that it uses clustering to classify data.
       - you give it sample+no standard answer-> it will generate the answer based on comparing..
    3.Semi-supervised learning?
      - model are built based on labeled+unlabeled data.
      - you give it a large labeled data in order to learn -> classify your problems based on them.
        this type is time-consuming.
    4.Reinforcement learning ?
      - model are built using the behavior of the related environment to maximize the reward of the signal function.
      - you give it data + not labeled(not standard answer) -> it will predict it based on behavior in the environment
        -> reward signal will increase.

1.Supervised learning?
    <img width="754" height="359" alt="image" src="https://github.com/user-attachments/assets/ecfcad4b-0f91-4e9a-a145-9b5dd14b290e" />

        - you give it features + standard answer as you see -> then he will solve based on these(uses Regression)
        - you can use it in classification to classify data based on discrete items..
2.Unsupervised learning?
  <img width="272" height="151" alt="image" src="https://github.com/user-attachments/assets/cfe87eac-7089-4c4c-863b-528031a24d8c" />
      
      - it predict output based on features that are similar between data.(uses regression)
      - you can use it in classification to generate discrete data like this;
  <img width="784" height="382" alt="image" src="https://github.com/user-attachments/assets/5443d173-b84e-42bf-ba99-0a0fb2c766a7" />

      - you can use it in clusering (define based on similarities)
        = like ; detering the type of the fish based on attributes..
3.Semi-Supervised Learning ?
  <img width="753" height="377" alt="image" src="https://github.com/user-attachments/assets/f3c477a7-bd0c-455a-bc59-2e74d77f7733" />

      - you give it labeled +unlabeled data -> define unlabeled based on labeled one.
4.Reinforcement learning?
<img width="678" height="272" alt="image" src="https://github.com/user-attachments/assets/b3cf8f2c-a63f-4ed5-982e-52b6e765aaea" />

    - this type learn from the environment by maximizing or minimizing the reward function.
    like; autonomous vechicles learn from the environment.
  <img width="696" height="193" alt="image" src="https://github.com/user-attachments/assets/7d617210-fe01-49ef-8cb2-93cfe6585030" />

-----
# unit3
What is the Ml process?

  <img width="778" height="211" alt="image" src="https://github.com/user-attachments/assets/bf33c265-0c03-4015-aac8-8aad3d1638d3" />

        - data preparation+cleansing+extraction(selection) -> this is the job of data analyst or scientists
        - Model training+ evaluation -> job of Ml engineer
        - Model deployment+integration -> job of MLOps engineer.
        
Some concepts?

    - Dataset;  
        - collection of data -> each piece of data(smaple) + attributes (features)
    -Training set ;
        - dataset that are fed to the model to train it.
        - each sample = training sample
    - Test set ;
        - dataset that are fed to test the model output.
        - each smaple => test sample.
        
  shape of dataset:
    <img width="707" height="319" alt="image" src="https://github.com/user-attachments/assets/341dea60-9e49-4d5c-8a22-f45bc85505b7" />

  What is the importance of Data processing?
    <img width="639" height="257" alt="image" src="https://github.com/user-attachments/assets/92770fa0-89d4-4cc6-93b5-fdc1545d1f8d" />
    <img width="512" height="314" alt="image" src="https://github.com/user-attachments/assets/de2acb40-a5f4-43fa-8150-8fcee64de61a" />

      = Data determine the scope +correctness of model so it must be good.
      3 pillers for this process:
        - Data cleansing-> make clean for data(eliminate noise)
        - Data  dimension reducing => simplify attributes to avoid dimensionality.
        - Data standerdization -> make it standard to reduce noise 
  
  What are the pillers that data cleansing based on?

      1. Data filtering.
      2. Data loss
      3. handle errors.
      4. Merge data from multi sources.
      5. Data consildation.
How does dirty data look like?
  <img width="637" height="215" alt="image" src="https://github.com/user-attachments/assets/6bf4753f-6f02-4318-830a-7ea980d398aa" />

    it contain:
      - incompleteness data.
      - noise.
      - inconsistency.
How does data conversion act in order to fed it into Ml model? (process of it)

    - Encoding data  
    - convert numeric data -> categorical data
    - other data:
        - Embedding words into text 
        - image data processing.
    - standardized data (to meet the requirements)
    - feature augmentation -> convert existing variables-> new features.

Why we need feature selection when building a model?
    <img width="328" height="259" alt="image" src="https://github.com/user-attachments/assets/1d87a7c6-9384-4085-ba4b-c4d91100a8d2" />

    - to improve the process of fedding + generate data from Ml models.
      as if you didn't make data clear+ didn't avoid dimensionality-> you will get overfitting
 Methods of Feature selection?

    1.filter 
  <img width="387" height="84" alt="image" src="https://github.com/user-attachments/assets/72b28136-087d-4a35-87fc-72d89b78edf0" />

      select feature subset by:
        1.find the correlations between features
          scores each features using statistics measurements->then choose the fit one
      
      limitations of filter selection?
        - select redundant varibles + doesn't consider the relation between features.


     2. Wrapper 
   <img width="384" height="141" alt="image" src="https://github.com/user-attachments/assets/10ca093f-ac24-42dc-8582-942a55b34083" />

            - select feature after doing;
                1.eliminate recursive feature.
              note; use model to select the feature subset
            - train model on feature subset (computationally intensive)
            - provide high performance feature set for specific data.

    3. Embedded(Regularization..):
          -treat feature selection as a part of training method.
  <img width="445" height="188" alt="image" src="https://github.com/user-attachments/assets/3afdfd47-0578-4aaf-b484-d1eeea04796b" />

          - it's also called; penalization
          - uses predictive algorithm->to choose less complexity feature.

What is the overall process that we use to build AI model?

  - <img width="482" height="352" alt="image" src="https://github.com/user-attachments/assets/30c59292-8ba2-44d7-9724-7d7e021da53f" />


---
How is supervised Learning working?
  <img width="546" height="309" alt="image" src="https://github.com/user-attachments/assets/4523ce7b-59c4-4a29-8bc4-aa2f452cc50a" />

    -as you see -> we fed model with training set -> then he recogonize (Age(feature selection),Label is the result result)
      - based on this we gave testing set then look how he act
    - to be clear-> you have two stages;
      -learning phase
      -Prediction phase(testing phase)
    note;
      The factors of a good models are:
        1.Generalization(accuracy)  (this itself judge the model)
        2.Explainability  
        3.speed. (speed at giving result)

what are the model Effectiveness?

      - Generalization capability -> accuracy of the model on new sample should be good (robustness)
      - Error -> difference between prediction of error on a sample + error on actual sample.
        - Training error -> error of modle on a training set
        - Generalization erroor ->error of model on new sample
      -Fitting.
          -underfitting -> training error is large(due to small time of training..)
          -overfitting -> training error is small +generalization error is high (accuracy low) due to high time of training
    -Model capacity -> the ability of the model to include varies functions  (complex problems)
        if the model doesn't have large capacity -> underfitting will occur  
        if model can solve complex problems -> overfiting may occur.
  <img width="666" height="221" alt="image" src="https://github.com/user-attachments/assets/51da715a-3456-444e-8672-849cdd2fdde3" />



What are the causes of overfitting?

      - Prediction error -> varience+bias.
        1.variance:
          - how much prediction result deviates than mean.
          -caused by:
              - sensitivity of the model to the fluctuations of training sets
        2.Bias:
          - difference between (predicted values& actual values)

      Typesof variance+Bias?
        1.Low bias+low variance.=>good model
        2.High bias+low variance=>inadequate model
        3.Low bias+high variance->inadequate model
        4.High bias+High variance=>bad 
  <img width="396" height="373" alt="image" src="https://github.com/user-attachments/assets/5d674ea0-6f5c-4299-b674-716e709a8378" />

What are the ralation between Complexity+Erros of Models?

    - complex increase -> training error decreases +test error decrease then increase again
  <img width="382" height="184" alt="image" src="https://github.com/user-attachments/assets/5cfe1d4d-ef5a-427b-b0c6-55fd19d33b9c" />

----
Some Terminologies to be known at ML?
  <img width="304" height="206" alt="image" src="https://github.com/user-attachments/assets/7dd78cfd-a1ed-43d0-90f6-8385ed5d295c" />

      - P -> positive cases
      - N-> Negative cases
      - TP -> true positive -> positive cases that are correctly classified
      - TN -> true Negative -> Negative cases that are correctly classified.
      - FP -> false positive -> Postive cases that are uncorrectly classified
      - FN -> false Negative -> Negative cases that are uncorrectly classified.
    Confusion matrix -> m*m matrix used to determine classification of training 
        - output of it should be at diagnoal(which means these cases are identified by the model)
          -while other entries may contain some data that didn't identified by the model.
  <img width="698" height="383" alt="image" src="https://github.com/user-attachments/assets/1bbd1f0c-1f13-4e43-aa91-2ea8608b2e73" />

ex;
  <img width="790" height="355" alt="image" src="https://github.com/user-attachments/assets/2b85163e-84fa-4b6f-91a6-059480bbed7a" />

----
# unit4

Some Machine Learning Methods?

    1.Gradient Descent(1).
      - method uses -> negative gradient direct as search direction.
              -negative gradient direction -> fastest descent direction of the current one.
  formula:<img width="270" height="37" alt="image" src="https://github.com/user-attachments/assets/da14ad24-4246-4428-a216-627b5e871b67" />
  
    - Eata (M) -> learning rate
    - negative termm (M delta f (x))-> indicate change of w (weight parameter) in each             learning phase.
    - Wk+1 -> new weight of the model (may be small or large)
  <img width="252" height="202" alt="image" src="https://github.com/user-attachments/assets/78bbde1c-c869-4dc7-b7c8-b14d6a7cf7cb" />

    2.Gradient Descent(2)
      types;
        1. Batch gradient descent(BGD)
            - sum of the gradients of m samples of dataset at the point you train the model
  eqauation:
      <img width="367" height="80" alt="image" src="https://github.com/user-attachments/assets/730a4f57-dd06-4dab-a916-973a3697a1a5" />

      2.Stochastic gradient descent(SGD)
          - use random samples to calc new weight;
  equation;
    <img width="282" height="59" alt="image" src="https://github.com/user-attachments/assets/c2791582-98eb-4040-89f7-b9b63134d495" />
      
      3.Mini-batch gradient descent(MBGD)
        cobine (BGD,SGD) to update the weight.
  equation:
    <img width="336" height="84" alt="image" src="https://github.com/user-attachments/assets/c1188ff4-c2f0-4def-aa77-32aa764e6c16" />

 What are the difference between Parameters + hyperparameters?

     - Parameters -> are automatically learned by the model during historical data
           so that ;
              - the model can't function without them.
              
     - Hyperparameters -> are manually set (keys you set in order to control the training)
           - neurons  or activation function,....
           
What are the Hyperparameter search general process?

      -Divide dataset ->training set,validation set,test set.
      -opimize model parameters using training set
      - search for model hyperparameters using the validation set based on model performance matrics.

What are some of the hyperparameter tuning methods?

      1.Grid search;
<img width="292" height="283" alt="image" src="https://github.com/user-attachments/assets/25b27fcd-ca51-476d-b296-e8a3c78246bf" />

        -it performs an exhaustive search for all possible hyberparameters.
        - it's manually specified.
        - it's time consuming 
      2.Random search:
  <img width="286" height="265" alt="image" src="https://github.com/user-attachments/assets/4f4b33c3-8b82-4be4-a01f-aad4900038e1" />

        - more appropriate than grid search if hyperparamaters are large.
----
  
what is the Cross Validation?

    - statistical analysis method -> check the performance of classifiers.
    - it splits dataset ->traing+validation set.
    - K-fold cross validation(k-fold cv)
        - divide dataset into multi datasets(k)
        -all of these group are considered validation set --> 1-k is the training set
          note; average classification score of k models on each validation set are set a performance matrix for k-fold cv.
            -k in k-fold cv (hyperparameter)
            - k starts from 2 or 3 if the original dataset is small.
  <img width="727" height="234" alt="image" src="https://github.com/user-attachments/assets/799b2f53-8491-437a-9298-39adeb556e79" />
          
            - we divide the training set into ks sets(ks(training,validation,1-k(test set))
            - then put k=3(3 groups)
              - start train 2 of them by the model(like k1,k2)
              - then validate the remaining set(k3)
              - then change the 2 sets for train and train  again(k2,k3)
                - now validate by k1
              - then change train to k1,k3 +validate by k2 
              after all of these choses the best accuracy between them.
  <img width="727" height="234" alt="image" src="https://github.com/user-attachments/assets/844e1127-3c72-4f07-9a33-6fd5f7fdbb6f" />


-----
# unit5
->Common ML Algorithms:
