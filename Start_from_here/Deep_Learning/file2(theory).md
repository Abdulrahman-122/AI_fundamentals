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
