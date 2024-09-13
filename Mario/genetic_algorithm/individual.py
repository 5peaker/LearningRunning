 
from abc import abstractmethod
from typing import Optional, Union
import numpy as np

class Individual(object):
    def __init__(self):
        pass

    @abstractmethod
    def calculate_fitness(self):
        raise Exception("calculate_fitness 기능이 먼저 정의되어야 합니다.")

    @property
    @abstractmethod
    def fitness(self):
        raise Exception("Property 'fitness'가 먼저 정의되어야 합니다.")

    @fitness.setter
    @abstractmethod
    def fitness(self, val):
        raise Exception("Property 'fitness'를 설정할 수 없습니다. calculate_fitness를 먼저 동작해 주세요.")

    @abstractmethod
    def encode_chromosome(self):
        raise Exception("encode_chromosome 기능이 먼저 정의되어야 합니다.")

    @abstractmethod
    def decode_chromosome(self):
        raise Exception("decode_chromosome 기능이 먼저 정의되어야 합니다.")

    @property
    @abstractmethod
    def chromosome(self):
        raise Exception("Property 'chromosome'이 먼저 정의되어야 합니다.")

    @chromosome.setter
    def chromosome(self, val):
        raise Exception("Property 'chromosome'을 설정할 수 없습니다.")