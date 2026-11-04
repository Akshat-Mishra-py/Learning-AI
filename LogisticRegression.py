import os
from abc import ABC, abstractmethod


import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset",output_dir=".\\dataset")
dataset_path = os.path.join(path, "Titanic-Dataset.csv")

class Loss(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class MeanSquaredError(Loss):
    ErrorType = 0
    def init(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        return self
    def forward(self):
        return np.mean((self.Y_true - self.Y_pred)**2)

    def backward(self):
        '''Results give us the partial defferential with respect to Y_pred i.e. : del(L)/del(Y_pred)'''
        return self.Y_true - self.Y_pred

class Activation(ABC):
    @abstractmethod
    def forward(self) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass

class Sigmoid(Activation):
    activationType = 0
    def init(self, Z:np.ndarray) -> Activation:
        self.Z = Z
        return self

    def forward(self):
        self.Y_pred = (1-(1/np.exp(self.Z)))**2
        return self.Y_pred

    def backward(self) -> np.ndarray:
        '''returns partial defferential with respect to Z i.e. : del(Y_pred)/del(Z)'''
        return self.Y_pred * (1 - self.Y_pred)
    
    def tobytes(self):
        return b"0"

class LogisticRegression:
    def __init__(self,Weight=None, Bias=None, LossFunc: MeanSquaredError|None = None, Activation: Sigmoid|None = None ,WShape=(1,), BShape=(1,), n_iter = 100):
        self._model_type = 0
        if Weight is not None and Bias is not None:
            self.weight = Weight
            self.bias = Bias
        else:
            self.weight = np.random.rand(*WShape)
            self.bias = np.random.rand(*BShape)

        self.loss = LossFunc
        self.activation = Activation
        self.iter = n_iter

    def feed_forward(self, X):
        Z = self.weight * X + self.bias
        self.activation = self.activation.init(Z) #type:ignore[reportOptionalMemberAccess]
        A = self.activation.forward()
        return A

    def validate(self, X_test, Y_test):
        pred = self.feed_forward(X_test)
        self.loss = self.loss.init(pred, Y_test) #type:ignore[reportOptionalMemberAccess]
        loss = self.loss.forward()
        return loss

    def save(self, filePath:str, fileName:str):
        "Saves as .model file"
        try:
            with open(filePath+fileName, "wb") as f:
                f.write(f"{self._model_type} \n".encode("utf-8"))
                f.write(self.weight.tobytes())
                f.write(self.bias.tobytes())
                f.write(b"\n")
                f.write(self.activation.activationType.to_bytes()) #type:ignore[reportOptionalMemberAccess]
        except IOError as io:
            print(io)
            return -1
        except Exception as e:
            print(e)
            return 0
        
        return 1


    def fit(self, X, Y):
        for _ in range(self.iter):
            print(f"Epoch - {_}", end=" - ")
            pred = self.feed_forward(X)
            self.loss = self.loss.init(pred,Y)  # pyright: ignore[reportOptionalMemberAccess]
            loss = self.loss.forward()
            dB = self.loss.backward() * self.activation.backward() #type:ignore[reportOptionalMemberAccess]
            dW = dB * X
            self.weight = self.weight + np.mean(dW)
            self.bias = self.bias + np.mean(dB)
            print(f"Total Loss = {loss}")
            

if __name__ == "__main__":
    
    df = pd.read_csv(dataset_path)
    df["Sex"] = (df["Sex"]=="male").astype(int)
    df = df.rename(columns={"Sex":"IsMale"})
    X = np.array(df["IsMale"].values)
    Y = np.array(df["Survived"].values)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Training 
    model = LogisticRegression(LossFunc=MeanSquaredError(), Activation=Sigmoid(), n_iter=10000)
    model.fit(x_train,y_train)

    #Validate
    print(f"Loss on Test Set {model.validate(x_test, y_test)}")

    model.save("./models/", "LogisticRegressModel.model")
    y_pred = model.feed_forward(x_test)
    for i in range(len(x_test)):
        print(x_test[i],"\t|\t", y_pred[i],"\t|\t", y_test[i])
    