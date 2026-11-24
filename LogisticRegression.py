import os
from abc import ABC, abstractmethod


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Download latest version
# path = kagglehub.dataset_download("yasserh/titanic-dataset",output_dir=".\dataset")
dataset_path = os.path.join("./dataset", "Titanic-Dataset.csv")

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
        return self.Y_pred - self.Y_true

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
        # Check this 
        self.Y_pred = (1/(1+np.exp(-self.Z)))
        return self.Y_pred

    def backward(self) -> np.ndarray:
        '''returns partial defferential with respect to Z i.e. : del(Y_pred)/del(Z)'''
        return self.Y_pred * (1 - self.Y_pred)
    
    def tobytes(self):
        return b"0"

class LogisticRegression:
    def __init__(self,Weight=None, Bias=None, LossFunc: MeanSquaredError|None = None, Activation: Sigmoid|None = None ,WShape=(1,), BShape=(1,), n_iter = 100, lr:float = 0.1):
        self._model_type = 0
        self.lr = lr
        if Weight is not None and Bias is not None:
            self.weight = Weight
            self.bias = Bias
        else:
            self.weight = np.zeros(*WShape)
            self.bias = np.zeros(*BShape)

        self.loss = LossFunc
        self.activation = Activation
        self.iter = n_iter

    def feed_forward(self, X):
        Z =  X @ self.weight + self.bias
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
            os.makedirs(filePath, exist_ok=True)
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
            print(f"Epoch - {_}", end=" \t ")
            
            pred = self.feed_forward(X)
            self.loss = self.loss.init(pred,Y)  # pyright: ignore[reportOptionalMemberAccess]
            loss = self.loss.forward()
            dB = self.loss.backward() * self.activation.backward() #type:ignore[reportOptionalMemberAccess]
            dW =  X.T @ dB / len(X)
            self.weight = self.weight - (dW * self.lr)
            self.bias = self.bias - (np.mean(dB)*self.lr)
            print(f"Total Loss = {loss}")
            

if __name__ == "__main__":

    # Data fetching and Fixing
    df = pd.read_csv(dataset_path)
    df["Sex"] = (df["Sex"]=="male").astype(int)
    df = df.rename(columns={"Sex":"IsMale"})
    df = df.select_dtypes(include=["int64","float64"])
    df["AgeMissing"] = df["Age"].isna().astype(float)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df_feature = df.drop(columns=["Survived", "PassengerId"])

    # Here starts the ML
    X = np.array(df_feature.values)
    Y = np.array(df["Survived"].values)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Training 
    model = LogisticRegression(LossFunc=MeanSquaredError(), Activation=Sigmoid(), n_iter=2000, WShape=(X.shape[-1],), lr=0.02)
    model.fit(x_train,y_train)
    #Validate
    print(f"Loss on Test Set {model.validate(x_test, y_test)}")

    model.save("./models/", "LogisticRegressModel.model")
    y_pred = model.feed_forward(x_test)
    print("Surviving Probab|\t Did Survive")
    for i in range(len(x_test)):
        print(f"|\t {y_pred[i]:.2f} \t|\t{ y_test[i]}")
    