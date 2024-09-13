import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import random, os, csv

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
from config import Config

class Mario(Individual):
    def __init__(self,
                 config: Config,
                 chromosome: Optional[Dict[str, np.ndarray]] = None,
                 hidden_layer_architecture: List[int] = [12, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Union[int, float] = np.inf,
                 name: Optional[str] = None,
                 debug: Optional[bool] = False,
                 ):
        
        self.config = config

        self.lifespan = lifespan
        self.name = name
        self.debug = debug

        self._fitness = 0  # 전반적 fitness 설정
        self._frames_since_progress = 0  # 마리오(플레이어 오브젝트)가 골인 지점을 향해 나아간 프레임의 수 
        self._frames = 0  # 마리오(플레이어 오브젝트)가 생존했던 프레임 수
        
        self.hidden_layer_architecture = self.config.NeuralNetwork.hidden_layer_architecture
        self.hidden_activation = self.config.NeuralNetwork.hidden_node_activation
        self.output_activation = self.config.NeuralNetwork.output_node_activation

        self.start_row, self.viz_width, self.viz_height = self.config.NeuralNetwork.input_dims

        if self.config.NeuralNetwork.encode_row:
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height
        # print(f'num inputs:{num_inputs}')
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # 인풋 노드
        self.network_architecture.extend(self.hidden_layer_architecture)  # 숨은 레이어 노드 
        self.network_architecture.append(6)                        # 도합 아웃풋 여섯 개 지정 ['u', 'd', 'l', 'r', 'a', 'b']

        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
                                         )

        # chromosome이 지정되어 있다면 가져온다
        if chromosome:
            self.network.params = chromosome
        
        self.is_alive = True
        self.x_dist = None
        self.game_score = None
        self.did_win = False
        self.allow_additional_time  = self.config.Misc.allow_additional_time_for_flagpole # 마리오의 "골인"하는 모습을 보기 위해서 필요
        self.additional_timesteps = 0
        self.max_additional_timesteps = int(60*2.5)
        self._printed = False

        # 어떤 키와 상응하는지?             B, NULL, SELECT, START, U, D, L, R, A
        # 인덱스                           0  1     2       3      4  5  6  7  8
        self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.farthest_x = 0


    @property
    def fitness(self):
        return self._fitness

    @property
    def chromosome(self):
        pass

    def decode_chromosome(self):
        pass

    def encode_chromosome(self):
        pass

    def calculate_fitness(self):
        frames = self._frames
        distance = self.x_dist
        score = self.game_score

        self._fitness = self.config.GeneticAlgorithm.fitness_func(frames, distance, score, self.did_win)

    def set_input_as_array(self, ram, tiles) -> None:
        mario_row, mario_col = SMB.get_mario_row_col(ram)
        arr = []
        
        for row in range(self.start_row, self.start_row + self.viz_height):
            for col in range(mario_col, mario_col + self.viz_width):
                try:
                    t = tiles[(row, col)]
                    if isinstance(t, StaticTileType):
                        if t.value == 0:
                            arr.append(0)
                        else:
                            arr.append(1)
                    elif isinstance(t, EnemyType):
                        arr.append(-1)
                    else: # 시스템적으로 위의 두 타입 외에는 화면 내에 있을 수 없기 때문 
                        raise Exception("좀 곤란한 상황, 진짜로")
                except:
                    t = StaticTileType(0x00)
                    arr.append(0) # 비운다

        self.inputs_as_array[:self.viz_height*self.viz_width, :] = np.array(arr).reshape((-1,1))
        if self.config.NeuralNetwork.encode_row:
            # mario_row 대상 원 핫 인코딩
            row = mario_row - self.start_row
            one_hot = np.zeros((self.viz_height, 1))
            if row >= 0 and row < self.viz_height:
                one_hot[row, 0] = 1
            self.inputs_as_array[self.viz_height*self.viz_width:, :] = one_hot.reshape((-1, 1))

    def update(self, ram, tiles, buttons, ouput_to_buttons_map) -> bool:
        """
        마리오 오브젝트 대상 메인 업데이트 콜. 주위 에어리어를 인풋으로 받고 신경망에 정보를 공급합니다.
        마리오 오브젝트 생존 시 True, 아니면 False 반환
        """
        if self.is_alive:
            self._frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            # 깃발(골인 지점) 도달
            if ram[0x001D] == 3:
                self.did_win = True
                if not self._printed and self.debug:
                    name = 'Mario '
                    name += f'{self.name}' if self.name else ''
                    print(f'{name} won')
                    self._printed = True
                if not self.allow_additional_time:
                    self.is_alive = False
                    return False
                
            # 기존의 지점보다 멀리 갔을 경우 기록 업데이트 
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += 1

            # 마리오의 골인 지점을 기록할 경우 (추가 시간 없을 경우 골인 즉시 스테이지 종료함)
            if self.allow_additional_time and self.did_win:
                self.additional_timesteps += 1
                
            if self.allow_additional_time and self.additional_timesteps > self.max_additional_timesteps:
                self.is_alive = False
                return False
            elif not self.did_win and self._frames_since_progress > 60*3:
                self.is_alive = False
                return False            
        else:
            return False

        # 빈 공간(난간 아래, 용암, 구멍 등)로 떨어질 경우 마리오는 즉사함
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        self.set_input_as_array(ram, tiles)

        # 아웃풋 계산
        output = self.network.feed_forward(self.inputs_as_array)
        threshold = np.where(output > 0.5)[0]
        self.buttons_to_press.fill(0)  # Clear

        # 버튼 설정
        for b in threshold:
            self.buttons_to_press[ouput_to_buttons_map[b]] = 1

        return True
    
def save_mario(population_folder: str, individual_name: str, mario: Mario) -> None:
    # population 폴더가 없을 경우 새로 생성 
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # 현재 환경에 대해 settings.config 파일 저장
    if 'settings.config' not in os.listdir(population_folder):
        with open(os.path.join(population_folder, 'settings.config'), 'w') as config_file:
            config_file.write(mario.config._config_text_file)
    
    # 각각의 시행 결과에 대해 기록 저장
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    L = len(mario.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = mario.network.params[w_name]
        bias = mario.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)
    
def load_mario(population_folder: str, individual_name: str, config: Optional[Config] = None) -> Mario:
    # 폴더 내에 지정한 기록이 확실히 있는지 체크
    if not os.path.exists(os.path.join(population_folder, individual_name)):
        raise Exception(f'{individual_name} 가 {population_folder} 내에 존재하지 않습니다.')

    # 따로 지정되지 않을 경우 기본 설정 불러오기
    if not config:
        settings_path = os.path.join(population_folder, 'settings.config')
        config = None
        try:
            config = Config(settings_path)
        except:
            raise Exception(f'settings.config 파일이 {population_folder} 내에 없습니다.')

    chromosome: Dict[str, np.ndarray] = {}
    # .npy files, i.e. W1.npy, b1.npy 등을 불러와서 chromosome으로 전달한다
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            chromosome[param] = np.load(os.path.join(population_folder, individual_name, fname))
        
    mario = Mario(config, chromosome=chromosome)
    return mario

# 현재 스테이지 진행상황 계산
def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))

    return (mean, median, std, _min, _max)

# 스테이지 진행상황 저장
def save_stats(population: Population, fname: str):
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = fname

    frames = [individual._frames for individual in population.individuals]
    max_distance = [individual.farthest_x for individual in population.individuals]
    fitness = [individual.fitness for individual in population.individuals]
    wins = [sum([individual.did_win for individual in population.individuals])]

    write_header = True
    if os.path.exists(f):
        write_header = False

    trackers = [('frames', frames),
                ('distance', max_distance),
                ('fitness', fitness),
                ('wins', wins)
                ]

    stats = ['mean', 'median', 'std', 'min', 'max']

    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(f, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',')
        if write_header:
            writer.writeheader()

        row = {}
        # csv 파일 내에 새로 삽입할 열을 생성
        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat

        # 지정한 열 안에 내용 작성 
        writer.writerow(row)

def load_stats(path_to_stats: str, normalize: Optional[bool] = False):
    data = {}

    fieldnames = None
    trackers_stats = None
    trackers = None
    stats_names = None

    # 존재하는 csv 파일 불러와서 필요한 값 추출
    with open(path_to_stats, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = reader.fieldnames
        trackers_stats = [f.split('_') for f in fieldnames]
        trackers = set(ts[0] for ts in trackers_stats)
        stats_names = set(ts[1] for ts in trackers_stats)
        
        for tracker, stat_name in trackers_stats:
            if tracker not in data:
                data[tracker] = {}
            
            if stat_name not in data[tracker]:
                data[tracker][stat_name] = []

        for line in reader:
            for tracker in trackers:
                for stat_name in stats_names:
                    value = float(line['{}_{}'.format(tracker, stat_name)])
                    data[tracker][stat_name].append(value)
    
    # 정규화: 
    if normalize:
        factors = {}
        for tracker in trackers:
            factors[tracker] = {}
            for stat_name in stats_names:
                factors[tracker][stat_name] = 1.0

        for tracker in trackers:
            for stat_name in stats_names:
                max_val = max([abs(d) for d in data[tracker][stat_name]])
                if max_val == 0:
                    max_val = 1
                factors[tracker][stat_name] = float(max_val)

        for tracker in trackers:
            for stat_name in stats_names:
                factor = factors[tracker][stat_name]
                d = data[tracker][stat_name]
                data[tracker][stat_name] = [val / factor for val in d]

    return data

# 수치 입력 체크
def get_num_inputs(config: Config) -> int:
    _, viz_width, viz_height = config.NeuralNetwork.input_dims
    if config.NeuralNetwork.encode_row:
        num_inputs = viz_width * viz_height + viz_height
    else:
        num_inputs = viz_width * viz_height
    return num_inputs

# 훈련용 패러미터 체크
def get_num_trainable_parameters(config: Config) -> int:
    num_inputs = get_num_inputs(config)
    hidden_layers = config.NeuralNetwork.hidden_layer_architecture
    num_outputs = 6  # U, D, L, R, A, B

    layers = [num_inputs] + hidden_layers + [num_outputs]
    num_params = 0
    for i in range(0, len(layers)-1):
        L      = layers[i]
        L_next = layers[i+1]
        num_params += L*L_next + L_next

    return num_params