import numpy as np
from typing import List
from .individual import Individual
    
class Population(object):
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @property
    def num_individuals(self) -> int:
        return len(self.individuals)

    @num_individuals.setter
    def num_individuals(self, val) -> None:
        raise Exception("Individual의 수를 설정할 수 없습니다. Population.individuals를 변경해 주세요.")

    @property
    def num_genes(self) -> int:
        return self.individuals[0].chromosome.shape[1]

    @num_genes.setter
    def num_genes(self, val) -> None:
        raise Exception("Genes의 수를 설정할 수 없습니다. Population.individuals를 변경헤 주세요.")

    @property
    def average_fitness(self) -> float:
        return (sum(individual.fitness for individual in self.individuals) / float(self.num_individuals))

    @average_fitness.setter
    def average_fitness(self, val) -> None:
        raise Exception("Fitness 평균을 설정할 수 없습니다. 파일이 읽기 전용으로 설정되어 있습니다.")

    @property
    def fittest_individual(self) -> Individual:
        return max(self.individuals, key = lambda individual: individual.fitness)

    @fittest_individual.setter
    def fittest_individual(self, val) -> None:
        raise Exception("Fittest 개체를 설정할 수 없습니다. 파일이 읽기 전용으로 설정되어 있습니다.")

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()

    def get_fitness_std(self) -> float:
        return np.std(np.array([individual.fitness for individual in self.individuals]))