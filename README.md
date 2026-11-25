# Learning AI
From this repository I learn to create models and train them from the small ML models like logistic regression to the base of transformers.

### Installation
To install the required dependencies, run the following command:
    
Using `uv`: 
```bash
uv sync
```
Using `pip`:
```bash
pip install uv 
uv sync
```

### Learning From Logistic Regression
While implementing logistic regression, I learned about the following concepts:
    
- **Activations** : The activation function is used to insert the   non-linearity into the model. The most common activation function used in logistic regression is the `Sigmoid` loss function.

    But here using `Sigmoid` activation i found out that the model when reached a very high accuracy, the nudge to the weights became neglegible due to the vanishing gradient problem. Which began due to the combination of `Sigmoid` activation and the `MSE` loss function.

    
    >**Warning:** Do not use `MSE` loss function with `Sigmoid` activation function. Use `Binary Cross Entropy` loss function instead.
    
- **Loss Function** : The loss function which is used to calculate the error between the predicted and actual values. This helps us to update the weights and biases of model in correct direction.

    Here I used `MSE` loss function to play around and study its effects on model, due to its ill compatibility with `Sigmoid` activation function. 

- **BackPropagation** : The process of updating the weights and biases of the model using the gradients calculated from the loss function. This is done using the chain rule of calculus.

    Here for its implementation I used the concept of Dependency Injection to inject the `Loss` and `Activation` into the `LogisticRegression` class. This lets me use the same `LogisticRegression` class with different combinations of `Loss` and `Activation` functions.

    > This Behaviour of Dependency Injection is achieved by Abstract Base Classes (ABC) in Python. The `Loss` and `Activation` classes are abstract base classes which define the interface, and the `MSE` and `Sigmoid` classes are implementations of these interfaces.

- **Extra:**
    - `Learning rate` and how it affects the convergence of the model.
    
    - I also learned about the importance of data preprocessing and how it affects the performance of the model. Due to bad data preprocessing, the model was not able to converge and was stuck without the major changes in the weights and biases. This was due to the fact that the data was not normalized and the features were not scaled properly.

    - The importance of correctly defining the matrix dimensions and beleive me, how many times I got stuck due to the wrong matrix dimensions. It is very important to correctly define the matrix dimensions and to keep track of them throughout the implementation.

PS: Now I may be able to advance faster as the Logistic Regression is base for the Deep Neural Networks and other complex models.

### TODO:
- [x] Implement the `MSE` loss function and use it with `Sigmoid` activation function to see the difference in performance. 
- [x] Implement the Logistic Regression model using the `MSE` loss function and `Sigmoid` activation function.
- [ ] Refactor the code to make it more modular and divide it into different files for better readability and maintainability.
- [ ] Implement the Neural Network model using the `BinaryCrossEntropy` loss function and `Sigmoid` activation function.
- [ ] Create an interactive interface to visualize the training process and the performance of the model.
- [ ] Implement the `Adam` optimizer to improve the convergence of the model.
- [ ] Implement the `Dropout` regularization technique to prevent overfitting of the model.
- [ ] Implement the `Batch Normalization` technique to improve the performance of the model.
- [ ] Implement the `Convolutional Neural Network` model for image classification tasks.
- [ ] Implement the `Recurrent Neural Network` model for sequence prediction tasks.
- [ ] Implement the `Long Short-Term Memory` (LSTM) model for sequence prediction tasks.
- [ ] Implement the `Gated Recurrent Unit` (GRU) model for sequence prediction tasks.
- [ ] Implement the `Transformer` model for natural language processing tasks.
- [ ] Implement the self-attention mechanism used in the `Transformer` model.
- [ ] Is this the end? 
