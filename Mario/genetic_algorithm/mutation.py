import numpy as np
from typing import List, Union, Optional
from .individual import Individual

def gaussian_mutation(chromosome: np.ndarray, prob_mutation: float, 
                      mu: List[float] = None, sigma: List[float] = None,
                      scale: Optional[float] = None) -> None:
    """
    확률 prob_mutation를 가지고 개체의 각 유전자에 대해 가우시안 변이를 시행.
    어떤 유전인자가 변이를 일으키는지 규정하는 역할을 담당.
    """
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # Mu와 Sigma가 사전에 규정되어 있으면, 각각의 인자에 대해 가우시안 분포를 그린다
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # 그렇지 않으면 각각의 모양은 N(0, 1)로부터 
    else:
        gaussian_mutation = np.random.normal(size=chromosome.shape)
    
    if scale:
        gaussian_mutation[mutation_array] *= scale

    # 업데이트 
    chromosome[mutation_array] += gaussian_mutation[mutation_array]

def random_uniform_mutation(chromosome: np.ndarray, prob_mutation: float,
                            low: Union[List[float], float],
                            high: Union[List[float], float]
                            ) -> None:
    """
    각 유전자를 확률인자 "prob mutation"를 통해 무작위로 변이시킵니다.
    변이 대상이 될 유전자는 (Low)와 (High) 사이에서 균일한 확률로 결정됩니다. 
    @Note [low, high) is defined for each gene to help get the full range of possible values
    """
    assert type(low) == type(high), 'low and high must have the same type'
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    if isinstance(low, list):
        uniform_mutation = np.random.uniform(low, high)
    else:
        uniform_mutation = np.random.uniform(low, high, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]

def uniform_mutation_with_respect_to_best_individual(chromosome: np.ndarray, best_chromosome: np.ndarray, prob_mutation: float) -> None:
    """
    각 유전자를 확률인자 "prob mutation"를 통해 무작위로 변이시킵니다.
    변이 대상으로 선택된 유전자는 현재 그룹 내에서 가장 좋은 개체의 유전자를 향해 조금씩 움직입니다.
    """
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(size=chromosome.shape)
    chromosome[mutation_array] += uniform_mutation[mutation_array] * (best_chromosome[mutation_array] - chromosome[mutation_array])

def cauchy_mutation(individual: np.ndarray, scale: float) -> np.ndarray:
    pass

def exponential_mutation(chromosome: np.ndarray, xi: Union[float, np.ndarray], prob_mutation: float) -> None:
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # 필요할 때 xi 값을 채운다
    if not isinstance(xi, np.ndarray):
        xi_val = xi
        xi = np.empty(chromosome.shape)
        xi.fill(xi_val)
        
    # E(0, xi) 대신 E(0, 1)를 얻을 수 있도록 한다
    xi_div = 1.0 / xi
    xi.fill(1.0)
    
    # Eq 11.17
    y = np.random.uniform(size=chromosome.shape)
    x = np.empty(chromosome.shape)
    x[y <= 0.5] = (1.0 / xi[y <= 0.5]) * np.log(2 * y[y <= 0.5])
    x[y > 0.5] = -(1.0 / xi[y > 0.5]) * np.log(2 * (1 - y[y > 0.5]))

    # Eq 11.16
    delta = np.empty(chromosome.shape)
    delta[mutation_array] = (xi[mutation_array] / 2.0) * np.exp(-xi[mutation_array] * np.abs(x[mutation_array]))

    # 업데이트: E(0, xi) = (1 / xi) * E(0 , 1)
    delta[mutation_array] = xi_div[mutation_array] * delta[mutation_array]

    # 개별적 업데이트 시행 
    chromosome[mutation_array] += delta[mutation_array]

def mmo_mutation(chromosome: np.ndarray, prob_mutation: float) -> None:
    from scipy import stats
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    normal = np.random.normal(size=chromosome.shape)  # Eq 11.21
    cauchy = stats.cauchy.rvs(size=chromosome.shape)  # Eq 11.22
    
    # Eq 11.20
    delta = np.empty(chromosome.shape)
    delta[mutation_array] = normal[mutation_array] + cauchy[mutation_array]

    # 개별적 업데이트 시행 
    chromosome[mutation_array] += delta[mutation_array]