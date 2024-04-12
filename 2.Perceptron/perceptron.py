import numpy as np
from chart import make_line

def map_to(datas:np.ndarray, _min:float, _max:float):
    '''Map datas to _min ~ _max'''

    return (_max - _min) * datas + _min

def make_fake_datas(n_datas:int):
    '''Make fade datas'''

    n_datas = n_datas // 2

    # type 1
    data1 = np.random.rand(n_datas, 2)
    label = np.reshape(np.zeros(n_datas), (n_datas, 1))
    data1[:, 0] = map_to(data1[:, 0], -1, 2)
    data1[:, 1] = map_to(data1[:, 1], 1, 3)
    data1 = np.append(data1, label, axis=1)

    # type 2
    data2 = np.random.rand(n_datas, 2)
    label = np.reshape(np.ones(n_datas), (n_datas, 1))
    data2[:, 0] = map_to(data2[:, 0], 2, 5)
    data2[:, 1] = map_to(data2[:, 1], -2, 1)
    data2 = np.append(data2, label, axis=1)

    # shuffle datas
    result = np.append(data1, data2, axis=0)
    np.random.shuffle(result)
    return result


def ReLU(x):
    return max(0., x)

def ReLU_(x):
    return 1. if x > 0. else 0.

def Sigmond(x):
    return 1. / (1. + np.exp(-x))

def Sigmond_(x):
    return Sigmond(x) * (1. - Sigmond(x))

def Abs(x, y):
    return abs(x - y)

def Abs_(x, y):
    return 1. if x > y else -1.

def Mse(x, y):
    return (x - y) ** 2

def Mse_(x, y):
    return 2. * (x - y)




def Active(x):
    return Sigmond(x)

def Active_(x):
    return Sigmond_(x)

def Loss(p, l):
    return Abs(p, l)

def Loss_(p, l):
    return Abs_(p, l)



class Perceptron:
    '''Perceptron class'''

    def __init__(self, n_inputs):

        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)

    def predict(self, inputs:np.ndarray):
        '''Predict'''

        return Active(np.dot(inputs, self.weights) + self.bias)
    
    def train(self, datas:np.ndarray, n_epochs:int, lr:float):
        '''Train perceptron'''

        loss_records = []
        for epoch in range(n_epochs):
            for data in datas:

                inputs = data[:2]
                label = data[2]

                y = np.dot(inputs, self.weights) + self.bias
                pred = Active(y)
                loss = Loss(pred, label)

                # update weights
                loss_ = lr * Loss_(pred, label) * Active_(y)
                self.weights -= loss_ * inputs
                self.bias -= loss_

            print(f'epoch: {epoch}, loss: {loss}, weights: {self.weights}, bias: {self.bias}')
            loss_records.append(round(float(loss), 2))
        return loss_records


def test(perceptron:Perceptron, test_count = 50, equal_threshold = 0.1):

    print("Test ----------------------------")
    fake_datas = make_fake_datas(test_count)
    count = 0
    for data in fake_datas:
        inputs = data[:2]
        label = data[2]
        if abs(label - perceptron.predict(inputs)) < equal_threshold:
            count += 1

    acc = str(round(count / 50, 2) * 100) + "%"
    print(f"Accuracy: {acc}")


fake_datas = make_fake_datas(100)
perceptron = Perceptron(2)
records = perceptron.train(fake_datas, 100, 0.1)
make_line(range(len(records)), records, "Loss Records", "loss.html")

test(perceptron, 50, 0.3)