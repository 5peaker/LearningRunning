import configparser
import os
from typing import Any, Dict

_params = {
    # 그래픽 출력 패러미터
    'Graphics': {
        'tile_size': (tuple, float),
        'neuron_radius': float,
    },

    # 통계 저장용 패러미터
    'Statistics': {
        'save_best_individual_from_generation': str,
        'save_population_stats': str,
    },

    # 신경망 정의용 패러미터
    'NeuralNetwork': {
        'input_dims': (tuple, int),
        'hidden_layer_architecture': (tuple, int),
        'hidden_node_activation': str,
        'output_node_activation': str,
        'encode_row': bool,
    },

    # 유전 알고리즘
    'GeneticAlgorithm': {
        'fitness_func': type(lambda : None)
    },

    # 크로스오버 알고리즘
    'Crossover': {
        'probability_sbx': float,
        'sbx_eta': float,
        'crossover_selection': str,
        'tournament_size': int,
    },

    # 변이 결정 관련 패러미터
    'Mutation': {
        'mutation_rate': float,
        'mutation_rate_type': str,
        'gaussian_mutation_scale': float,
    },

    # 선택 결정 관련 패러미터
    'Selection': {
        'num_parents': int,
        'num_offspring': int,
        'selection_type': str,
        'lifespan': float
    },

    # 그 외 
    'Misc': {
        'level': str,
        'allow_additional_time_for_flagpole': bool
    }
}

class DotNotation(object):
    def __init__(self, d: Dict[Any, Any]):
        for k in d:
            # key가 다른 딕셔너리로 지정되어 있을 경우 그냥 속행한다 
            if isinstance(d[k], dict):
                self.__dict__[k] = DotNotation(d[k])
            # 리스트나 튜플로 지정된 경우 내부에 딕셔너리가 있는지 탐색한다
            elif isinstance(d[k], (list, tuple)):
                l = []
                for v in d[k]:
                    if isinstance(v, dict):
                        l.append(DotNotation(v))
                    else:
                        l.append(v)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = d[k]
    
    def __getitem__(self, name) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)

class Config(object):
    def __init__(self,
                 filename: str
                 ):
        self.filename = filename
        
        if not os.path.isfile(self.filename):
            raise Exception('파일이 존재하지 않습니다: "{}"'.format(self.filename))

        with open(self.filename) as f:
            self._config_text_file = f.read()

        self._config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._config.read(self.filename)

        self._verify_sections()
        self._create_dict_from_config()
        self._set_dict_types()
        dot_notation = DotNotation(self._config_dict)
        self.__dict__.update(dot_notation.__dict__)

    def _create_dict_from_config(self) -> None:
        d = {}
        for section in self._config.sections():
            d[section] = {}
            for k, v in self._config[section].items():
                d[section][k] = v

        self._config_dict = d

    def _set_dict_types(self) -> None:
        for section in self._config_dict:
            for k, v in self._config_dict[section].items():
                try:
                    _type = _params[section][k]
                except:
                    raise Exception('값: "{}" 이 섹션: "{}" 에 대하여 존재하지 않습니다. _params에서 설정해 주십시오.'.format(k, section))\
                # 기본적으로 _type은 int, str, float를 비롯한 내장 타입 중 하나를 가진다 (튜플일 경우 쪼개야 함)
                if isinstance(_type, tuple):
                    if len(_type) == 2:
                        cast = _type[1]
                        v = v.replace('(', '').replace(')', '')  # 괄호 제거
                        self._config_dict[section][k] = tuple(cast(val) for val in v.split(','))
                    else:
                        raise Exception('구문 분석 및 캐스팅 유형 확인을 위해 튜플 인자 두(2) 개가 필요합니다.')
                elif 'lambda' in v:
                    try:
                        self._config_dict[section][k] = eval(v)
                    except:
                        pass
                # Boolean인지 체크
                elif _type == bool:
                    self._config_dict[section][k] = _type(eval(v))
                # 그 외의 경우 정상적으로 정리한다
                else:
                    self._config_dict[section][k] = _type(v)

    def _verify_sections(self) -> None:
        # 검증
        for section in self._config.sections():
            # 섹션에서 작업이 허용되어 있는지 체크
            if section not in _params:
                raise Exception('현재 섹션 "{}" 에 허용된 패러미터가 없습니다. 이 문제를 해결하고 다시 동작시켜 주십시오.'.format(section))

    def _get_reference_from_dict(self, reference: str) -> Any:
        path = reference.split('.')
        d = self._config_dict
        for p in path:
            d = d[p]
        
        assert type(d) in (tuple, int, float, bool, str)
        return d

    def _is_number(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False