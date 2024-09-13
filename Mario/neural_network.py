import numpy as np
from typing import List, Callable, NewType, Optional

ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])

# 활성 함수를 하나의 새 타입으로 지정
sigmoid = ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))
tanh = ActivationFunction(lambda X: np.tanh(X))
relu = ActivationFunction(lambda X: np.maximum(0, X))
leaky_relu = ActivationFunction(lambda X: np.where(X > 0, X, X * 0.01))
linear = ActivationFunction(lambda X: X)

class FeedForwardNetwork(object):
    def __init__(self,
                 layer_nodes: List[int],
                 hidden_activation: ActivationFunction,
                 output_activation: ActivationFunction,
                 init_method: Optional[str] = 'uniform',
                 seed: Optional[int] = None):
        self.params = {}
        self.layer_nodes = layer_nodes
        # print(self.layer_nodes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.inputs = None
        self.out = None

        self.rand = np.random.RandomState(seed)

        # 무게와 편향 초기 지정
        for l in range(1, len(self.layer_nodes)):
            if init_method == 'uniform':
                self.params['W' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], self.layer_nodes[l-1]))
                self.params['b' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], 1))
            
            else:
                raise Exception("더 많은 옵션을 지정받아야 합니다.")

            self.params['A' + str(l)] = None
        
        
    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        A_prev = X
        L = len(self.layer_nodes) - 1  # 참고값: len(self.params) // 2

        # 숨은 레이어에 정보 전달
        for l in range(1, L):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_prev = self.hidden_activation(Z)
            self.params['A' + str(l)] = A_prev

        # 아웃풋에 정보 전달
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A_prev) + b
        out = self.output_activation(Z)
        self.params['A' + str(L)] = out

        self.out = out
        return out

    def softmax(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) / np.sum(np.exp(X), axis=0)

# 지정한 활성화 함수 타입에 따라 동작 방식을 할당
def get_activation_by_name(name: str) -> ActivationFunction:
    activations = [('relu', relu),
                   ('sigmoid', sigmoid),
                   ('linear', linear),
                   ('leaky_relu', leaky_relu),
                   ('tanh', tanh),
    ]

    func = [activation[1] for activation in activations if activation[0].lower() == name.lower()]
    assert len(func) == 1

    return func[0]