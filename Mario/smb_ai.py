
## 필요한 패키지 불러오기 -----------------
import retro, os, shutil
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from typing import Tuple, List, Optional
import random, sys, math
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")

from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config
from nn_viz import NeuralNetworkViz
from mario import Mario, save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation

## Python 3.8 기반 ----------------------

# 실행 때마다 details 폴더 초기화할 것! 
def check_details():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    details_folder = os.path.join(current_folder, "details")
    
    if os.path.exists(details_folder) and os.path.isdir(details_folder):
        print(f"기록이 저장된 폴더가 이미 존재합니다: {details_folder}")
        
        for filename in os.listdir(details_folder):
            file_path = os.path.join(details_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"파일 삭제하였음: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"폴더 삭제함: {file_path}")
            except Exception as e:
                print(f"파일 또는 폴더 {file_path} 삭제 중 에러 발생: {e}")
        print("과거에 저장된 모든 기록을 삭제했습니다.")
    
## 기초 설정 및 함수 ------------------
normal_font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)

# PyQt 기반 창 경계 그리기 
def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)

# 시각화 기능 구현 함수
class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config, nn_viz: NeuralNetworkViz):
        super().__init__(parent)
        self.size = size
        self.config = config
        self.nn_viz = nn_viz
        self.ram = None
        self.x_offset = 150
        self.tile_width, self.tile_height = self.config.Graphics.tile_size
        self.tiles = None
        self.enemies = None
        self._should_update = True

    def _draw_region_of_interest(self, painter: QPainter) -> None:
        # 행렬에서 마리오 오브젝트의 위치 추적
        mario = SMB.get_mario_location_on_screen(self.ram)
        mario_row, mario_col = SMB.get_mario_row_col(self.ram)
        x = mario_col
       
        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))

        start_row, viz_width, viz_height = self.config.NeuralNetwork.input_dims
        painter.drawRect(x*self.tile_width + 5 + self.x_offset, start_row*self.tile_height + 5, viz_width*self.tile_width, viz_height*self.tile_height)

    def draw_tiles(self, painter: QPainter):
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                painter.setPen(QPen(Qt.black,  1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (self.tile_width * col) + self.x_offset
                y_start = 5 + (self.tile_height * row)

                loc = (row, col)
                tile = self.tiles[loc]

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
                    color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass

                painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        if self._should_update:
            draw_border(painter, self.size)
            if not self.ram is None:
                self.draw_tiles(painter)
                self._draw_region_of_interest(painter)
                self.nn_viz.show_network(painter)
        else:
            # draw_border(painter, self.size)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
            txt = '디스플레이가 숨겨져 있습니다.\n Ctrl+V를 눌러 표시해 주세요.\n 설정: {}'.format(args.config)
            painter.drawText(event.rect(), Qt.AlignCenter, txt)
            pass

        painter.end()

    def _update(self):
        self.update()

class GameWindow(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config):
        super().__init__(parent)
        self._should_update = True
        self.size = size
        self.config = config
        self.screen = None
        self.img_label = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.img_label)
        self.setLayout(self.layout)
        
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        if self._should_update:
            draw_border(painter, self.size)
            if not self.screen is None:
                # self.img_label = QtWidgets.QLabel(self.centralWidget)
                # screen = self.env.reset()
    
                width = self.screen.shape[0] * 3 
                height = int(self.screen.shape[1] * 2)
                resized = self.screen
                original = QImage(self.screen, self.screen.shape[1], self.screen.shape[0], QImage.Format_RGB888)
                
                # 이미지와 라벨 설정
                qimage = QImage(original)
                # 오브젝트를 스크린 진행의 중심에 둘 것
                x = (self.screen.shape[0] - width) // 2
                y = (self.screen.shape[1] - height) // 2
                self.img_label.setGeometry(0, 0, width, height)
                
                # 이미지를 더한다
                pixmap = QPixmap(qimage)
                pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
                self.img_label.setPixmap(pixmap)
        else:
            self.img_label.clear()
            # draw_border(painter, self.size)
        painter.end()

    def _update(self):
        self.update()

# 스크린 하단에 표시할 위젯 관련 클래스
class InformationWidget(QtWidgets.QWidget):
    def __init__(self, parent, size, config):
        super().__init__(parent)
        self.size = size
        self.config = config

        self.grid = QtWidgets.QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self._init_window()
        # self.grid.setSpacing(20)
        self.setLayout(self.grid)

    # 화면 초기화
    def _init_window(self) -> None:
        info_vbox = QVBoxLayout()
        info_vbox.setContentsMargins(0, 0, 0, 0)
        ga_vbox = QVBoxLayout()
        ga_vbox.setContentsMargins(0, 0, 0, 0)

        # 현재 세대 정보 표시 
        generation_label = QLabel()
        generation_label.setFont(font_bold)
        generation_label.setText('세대: ')
        generation_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.generation = QLabel()
        self.generation.setFont(normal_font)
        self.generation.setText("<font color='red'>" + '1' + '</font>')
        self.generation.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_generation = QHBoxLayout()
        hbox_generation.setContentsMargins(5, 0, 0, 0)
        hbox_generation.addWidget(generation_label, 1)
        hbox_generation.addWidget(self.generation, 1)
        info_vbox.addLayout(hbox_generation)

        # 현재 개별학습 과정 정보 표시
        current_individual_label = QLabel()
        current_individual_label.setFont(font_bold)
        current_individual_label.setText('개별 학습:')
        current_individual_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.current_individual = QLabel()
        self.current_individual.setFont(normal_font)
        self.current_individual.setText('1/{}'.format(self.config.Selection.num_parents))
        self.current_individual.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_current_individual = QHBoxLayout()
        hbox_current_individual.setContentsMargins(5, 0, 0, 0)
        hbox_current_individual.addWidget(current_individual_label, 1)
        hbox_current_individual.addWidget(self.current_individual, 1)
        info_vbox.addLayout(hbox_current_individual)

        # 최적합 모델 
        best_fitness_label = QLabel()
        best_fitness_label.setFont(font_bold)
        best_fitness_label.setText('최적합:')
        best_fitness_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.best_fitness = QLabel()
        self.best_fitness.setFont(normal_font)
        self.best_fitness.setText('0')
        self.best_fitness.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_best_fitness = QHBoxLayout()
        hbox_best_fitness.setContentsMargins(5, 0, 0, 0)
        hbox_best_fitness.addWidget(best_fitness_label, 1)
        hbox_best_fitness.addWidget(self.best_fitness, 1)
        info_vbox.addLayout(hbox_best_fitness) 

        # 최장거리 
        max_distance_label = QLabel()
        max_distance_label.setFont(font_bold)
        max_distance_label.setText('최장거리 도달:')
        max_distance_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.max_distance = QLabel()
        self.max_distance.setFont(normal_font)
        self.max_distance.setText('0')
        self.max_distance.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_max_distance = QHBoxLayout()
        hbox_max_distance.setContentsMargins(5, 0, 0, 0)
        hbox_max_distance.addWidget(max_distance_label, 1)
        hbox_max_distance.addWidget(self.max_distance, 1)
        info_vbox.addLayout(hbox_max_distance)

        # 입력:
        num_inputs_label = QLabel()
        num_inputs_label.setFont(font_bold)
        num_inputs_label.setText('입력: ')
        num_inputs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        num_inputs = QLabel()
        num_inputs.setFont(normal_font)
        num_inputs.setText(str(get_num_inputs(self.config)))
        num_inputs.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_num_inputs = QHBoxLayout()
        hbox_num_inputs.setContentsMargins(5, 0, 0, 0)
        hbox_num_inputs.addWidget(num_inputs_label, 1)
        hbox_num_inputs.addWidget(num_inputs, 1)
        info_vbox.addLayout(hbox_num_inputs)

        # 훈련가능 인자: 
        trainable_params_label = QLabel()
        trainable_params_label.setFont(font_bold)
        trainable_params_label.setText('훈련가능 인자: ')
        trainable_params_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        trainable_params = QLabel()
        trainable_params.setFont(normal_font)
        trainable_params.setText(str(get_num_trainable_parameters(self.config)))
        trainable_params.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_trainable_params = QHBoxLayout()
        hbox_trainable_params.setContentsMargins(5, 0, 0, 0)
        hbox_trainable_params.addWidget(trainable_params_label, 1)
        hbox_trainable_params.addWidget(trainable_params, 1)
        info_vbox.addLayout(hbox_trainable_params)

        # 선택
        selection_type = self.config.Selection.selection_type
        num_parents = self.config.Selection.num_parents
        num_offspring = self.config.Selection.num_offspring
        if selection_type == 'comma':
            selection_txt = '{}, {}'.format(num_parents, num_offspring)
        elif selection_type == 'plus':
            selection_txt = '{} + {}'.format(num_parents, num_offspring)
        else:
            raise Exception('알 수 없는 선택 유형 "{}"'.format(selection_type))
        selection_hbox = self._create_hbox('자녀:', font_bold, selection_txt, normal_font)
        ga_vbox.addLayout(selection_hbox)

        # 생명 (기본값은 무제한)
        lifespan = self.config.Selection.lifespan
        lifespan_txt = 'Infinite' if lifespan == np.inf else str(lifespan)
        lifespan_hbox = self._create_hbox('생명:', font_bold, lifespan_txt, normal_font)
        ga_vbox.addLayout(lifespan_hbox)

        # 변이 확률 
        mutation_rate = self.config.Mutation.mutation_rate
        mutation_type = self.config.Mutation.mutation_rate_type.capitalize()
        mutation_txt = '{} {}% '.format(mutation_type, str(round(mutation_rate*100, 2)))
        mutation_hbox = self._create_hbox('변이:', font_bold, mutation_txt, normal_font)
        ga_vbox.addLayout(mutation_hbox)

        # 결합
        crossover_selection = self.config.Crossover.crossover_selection
        if crossover_selection == 'roulette':
            crossover_txt = 'Roulette'
        elif crossover_selection == 'tournament':
            crossover_txt = 'Tournament({})'.format(self.config.Crossover.tournament_size)
        else:
            raise Exception('알 수 없는 결합 지정입니다: "{}"'.format(crossover_selection))
        crossover_hbox = self._create_hbox('결합:', font_bold, crossover_txt, normal_font)
        ga_vbox.addLayout(crossover_hbox)

        # SBX에서 eta 지정
        sbx_eta_txt = str(self.config.Crossover.sbx_eta)
        sbx_hbox = self._create_hbox('SBX Eta:', font_bold, sbx_eta_txt, normal_font)
        ga_vbox.addLayout(sbx_hbox)

        # 레이어
        num_inputs = get_num_inputs(self.config)
        hidden = self.config.NeuralNetwork.hidden_layer_architecture
        num_outputs = 6
        L = [num_inputs] + hidden + [num_outputs]
        layers_txt = '[' + ', '.join(str(nodes) for nodes in L) + ']'
        layers_hbox = self._create_hbox('레이어:', font_bold, layers_txt, normal_font)
        ga_vbox.addLayout(layers_hbox)

        self.grid.addLayout(info_vbox, 0, 0)
        self.grid.addLayout(ga_vbox, 0, 1)

    # 박스 생성
    def _create_hbox(self, title: str, title_font: QtGui.QFont,
                     content: str, content_font: QtGui.QFont) -> QHBoxLayout:
        title_label = QLabel()
        title_label.setFont(title_font)
        title_label.setText(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        content_label = QLabel()
        content_label.setFont(content_font)
        content_label.setText(content)
        content_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(5, 0, 0, 0)
        hbox.addWidget(title_label, 1)
        hbox.addWidget(content_label, 1)
        return hbox

# 메인 창 생성
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        global args
        self.config = config
        self.top = 150
        self.left = 150
        self.width = 1100
        self.height = 700

        self.title = 'Super Mario Bros AI'
        self.current_generation = 0
        # 실제로 "0"을 가질 세대를 정의
        self._true_zero_gen = 0

        self._should_display = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        # 할당된 키 나타내기:   B, NULL, SELECT, START, U, D, L, R, A
        # 인덱스                0  1     2       3      4  5  6  7  8
        self.keys = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # 주어진 키 매핑
        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }

        # 단일 과정으로 훈련 시작
        individuals: List[Individual] = []

        # args.load_inds에 기록된 개별 과정 불러오기
        num_loaded = 0
        if args.load_inds:
            # 특정되지 않았다면 config 파일을 덮어쓴다
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.load_file, 'settings.config'))
                except:
                    raise Exception(f'settings.config 파일이 {args.load_file} 아래에 없습니다.')

            set_of_inds = set(args.load_inds)

            for ind_name in os.listdir(args.load_file):
                if ind_name.startswith('best_ind_gen'):
                    ind_number = int(ind_name[len('best_ind_gen'):])
                    if ind_number in set_of_inds:
                        individual = load_mario(args.load_file, ind_name, self.config)
                        # 필요한 경우 디버그 지정
                        if args.debug:
                            individual.name = f'm{num_loaded}_loaded'
                            individual.debug = True
                        individuals.append(individual)
                        num_loaded += 1
            
            # 세대 지정
            self.current_generation = max(set_of_inds) + 1  # 다음 세대로 넘어가면 '+1' 지정 
            self._true_zero_gen = self.current_generation

        # args.replay_inds에 기록된 개별 과정 불러오기
        if args.replay_inds:
            # 특정되지 않았다면 config 파일을 덮어쓴다
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.replay_file, 'settings.config'))
                except:
                    raise Exception(f'settings.config 파일이 {args.replay_file} 아래에 없습니다.')

            for ind_gen in args.replay_inds:
                ind_name = f'best_ind_gen{ind_gen}'
                fname = os.path.join(args.replay_file, ind_name)
                if os.path.exists(fname):
                    individual = load_mario(args.replay_file, ind_name, self.config)
                    # 필요한 경우 디버그 지정
                    if args.debug:
                        individual.name= f'm_gen{ind_gen}_replay'
                        individual.debug = True
                    individuals.append(individual)
                else:
                    raise Exception(f'{ind_name} 파일이 {args.replay_file} 아래에 없습니다.')
        # 리플레이를 재생하는 게 아닐 경우 목적 달성을 위해 학습을 계속한다
        else:
            num_parents = max(self.config.Selection.num_parents - num_loaded, 0)
            for _ in range(num_parents):
                individual = Mario(self.config)
                # 필요한 경우 디버그 지정
                if args.debug:
                    individual.name = f'm{num_loaded}'
                    individual.debug = True
                individuals.append(individual)
                num_loaded += 1

        self.best_fitness = 0.0
        self._current_individual = 0
        self.population = Population(individuals)

        self.mario = self.population.individuals[self._current_individual]
        
        self.max_distance = 0  # 레벨 내에서 가장 멀리 간 지점을 체크 
        self.max_fitness = 0.0
        self.env = retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.Misc.level}')

        # 선택된 유형에 기초에 다음 세대의 사이즈를 책정한다
        self._next_gen_size = None
        if self.config.Selection.selection_type == 'plus':
            self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
        elif self.config.Selection.selection_type == 'comma':
            self._next_gen_size = self.config.Selection.num_offspring

        # 디스플레이 중이 아니라면 환경을 리셋한다 
        if args.no_display:
            self.env.reset()
        else:
            self.init_window()

            # 필요할 경우 라벨 내에서 세대 설정 
            if args.load_inds:
                txt = "<font color='red'>" + str(self.current_generation + 1) + '</font>'  # 인덱스 0에서 1로 변경
                self.info_window.generation.setText(txt)

            # 현재 리플레이 재생 중이라면, 학습 과정으로 간주하지 않고 설정을 조정한다
            if args.replay_file:
                self.info_window.current_individual.setText('Replay')
                txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
                self.info_window.generation.setText(txt)

            self.show()

        if args.no_display:
            self._timer.start(1000 // 1000)
        else:
            self._timer.start(1000 // 60)

    def init_window(self) -> None:
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.game_window = GameWindow(self.centralWidget, (514, 480), self.config)
        self.game_window.setGeometry(QRect(1100-514, 0, 514, 480))
        self.game_window.setObjectName('game_window')
        # 환경 설정을 리셋하고 게임 윈도우에 스크린을 넘긴다
        screen = self.env.reset()
        self.game_window.screen = screen
 
        self.viz = NeuralNetworkViz(self.centralWidget, self.mario, (1100-514, 700), self.config)

        self.viz_window = Visualizer(self.centralWidget, (1100-514, 700), self.config, self.viz)
        self.viz_window.setGeometry(0, 0, 1100-514, 700)
        self.viz_window.setObjectName('viz_window')
        self.viz_window.ram = self.env.get_ram()
        
        self.info_window = InformationWidget(self.centralWidget, (514, 700-480), self.config)
        self.info_window.setGeometry(QRect(1100-514, 480, 514, 700-480))

    def keyPressEvent(self, event):
        k = event.key()
        # m = {
        #     Qt.Key_Right : 7,
        #     Qt.Key_C : 8,
        #     Qt.Key_X: 0,
        #     Qt.Key_Left: 6,
        #     Qt.Key_Down: 5
        # }
        # if k in m:
        #     self.keys[m[k]] = 1
        # if k == Qt.Key_D:
        #     tiles = SMB.get_tiles(self.env.get_ram(), False)
        modifier = int(event.modifiers())
        if modifier == Qt.CTRL:
            if k == Qt.Key_V:
                self._should_display = not self._should_display

    def keyReleaseEvent(self, event):
        k = event.key()
        m = {
            Qt.Key_Right : 7,
            Qt.Key_C : 8,
            Qt.Key_X: 0,
            Qt.Key_Left: 6,
            Qt.Key_Down: 5
        }
        if k in m:
            self.keys[m[k]] = 0

    def next_generation(self) -> None:
        self._increment_generation()
        self._current_individual = 0

        if not args.no_display:
            self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, self._next_gen_size))

        # 적합성 계산 
        # print(', '.join(['{:.2f}'.format(i.fitness) for i in self.population.individuals]))

        if args.debug:
            print(f'현재 세대: {self.current_generation}, 영점 지정: {self._true_zero_gen}')
            fittest = self.population.fittest_individual
            print(f'현재 세대 최적값: {fittest.fitness}, 현재 세대 최장 도달점: {fittest.farthest_x}')
            num_wins = sum(individual.did_win for individual in self.population.individuals)
            pop_size = len(self.population.individuals)
            print(f'Wins: {num_wins}/{pop_size} (~{(float(num_wins)/pop_size*100):.2f}%)')

        if self.config.Statistics.save_best_individual_from_generation:
            folder = self.config.Statistics.save_best_individual_from_generation
            best_ind_name = 'best_ind_gen{}'.format(self.current_generation - 1)
            best_ind = self.population.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        if self.config.Statistics.save_population_stats:
            fname = self.config.Statistics.save_population_stats
            save_stats(self.population, fname)

        self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)

        random.shuffle(self.population.individuals)
        next_pop = []

        # "Parents + offspring"
        if self.config.Selection.selection_type == 'plus':
            # 남은 생명을 하나 차감
            for individual in self.population.individuals:
                individual.lifespan -= 1

            for individual in self.population.individuals:
                config = individual.config
                chromosome = individual.network.params
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan
                name = individual.name

                # 생명이 남아 있는 한, 오브젝트는 스테이지 클리어에 계속 도전한다 
                if lifespan > 0:
                    m = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, lifespan)
                    # 필요한 경우 디버그 지정 
                    if args.debug:
                        m.name = f'{name}_life{lifespan}'
                        m.debug = True
                    next_pop.append(m)

        num_loaded = 0

        while len(next_pop) < self._next_gen_size:
            selection = self.config.Crossover.crossover_selection
            if selection == 'tournament':
                p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
            elif selection == 'roulette':
                p1, p2 = roulette_wheel_selection(self.population, 2)
            else:
                raise Exception('결합 선택 "{}" 은 지원되지 않습니다.'.format(selection))

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # W_l and b_l 둘 다 chromosome으로 취급된다. (따라서 각 chromosome마다 결합 및 변이를 시행한다)
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]  
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # 결합
                # @NOTE: 가중치와 편향에 따라 사용할 결합의 종류를 지정한다.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # 변이
                # @NOTE: 가중치와 편향에 따라 만들어낼 변이의 종류를 지정한다.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # 자식 세대 결정
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                # [-1, 1]로 보낸다(clip)
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

            c1 = Mario(self.config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
            c2 = Mario(self.config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

            # 필요한 경우 디버그 지정 
            if args.debug:
                c1_name = f'm{num_loaded}_new'
                c1.name = c1_name
                c1.debug = True
                num_loaded += 1

                c2_name = f'm{num_loaded}_new'
                c2.name = c2_name
                c2.debug = True
                num_loaded += 1

            next_pop.extend([c1, c2])

        # 다음 세대를 설정
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eta = self.config.Crossover.sbx_eta

        # SBX에서 가중치와 편향 지정
        child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
        child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, eta)

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        mutation_rate = self.config.Mutation.mutation_rate
        scale = self.config.Mutation.gaussian_mutation_scale

        if self.config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)
        
        # 가중치에 변이 생성 
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

        # 편향에 변이 생성
        gaussian_mutation(child1_bias, mutation_rate, scale=scale)
        gaussian_mutation(child2_bias, mutation_rate, scale=scale)

    def _increment_generation(self) -> None:
        self.current_generation += 1
        if not args.no_display:
            txt = "<font color='red'>" + str(self.current_generation + 1) + '</font>'
            self.info_window.generation.setText(txt)

    def _update(self) -> None:
        """
        FPS Timer에 기반한 주 업데이트 방법. 알고리즘 업데이트, 윈도우 업데이트 등은 여기서 동작하게 하였습니다. 
        """
        ret = self.env.step(self.mario.buttons_to_press)

        if not args.no_display:
            if self._should_display:
                self.game_window.screen = ret[0]
                self.game_window._should_update = True
                self.info_window.show()
                self.viz_window.ram = self.env.get_ram()
            else:
                self.game_window._should_update = False
                self.info_window.hide()
            self.game_window._update()

        ram = self.env.get_ram()
        tiles = SMB.get_tiles(ram)  # 스크린 내의 타일 체크
        enemies = SMB.get_enemy_locations(ram)

        # self.mario.set_input_as_array(ram, tiles)
        self.mario.update(ram, tiles, self.keys, self.ouput_to_keys_map)
        
        if not args.no_display:
            if self._should_display:
                self.viz_window.ram = ram
                self.viz_window.tiles = tiles
                self.viz_window.enemies = enemies
                self.viz_window._should_update = True
            else:
                self.viz_window._should_update = False
            self.viz_window._update()
    
        if self.mario.is_alive:
            # 마리오가 최장기록을 경신한 경우
            if self.mario.farthest_x > self.max_distance:
                if args.debug:
                    print('최장 생존기록:', self.mario.farthest_x)
                self.max_distance = self.mario.farthest_x
                if not args.no_display:
                    self.info_window.max_distance.setText(str(self.max_distance))
        else:
            self.mario.calculate_fitness()
            fitness = self.mario.fitness
            
            if fitness > self.max_fitness:
                self.max_fitness = fitness
                max_fitness = '{:.2f}'.format(self.max_fitness)
                if not args.no_display:
                    self.info_window.best_fitness.setText(max_fitness)
            # 다음 개별학습으로
            self._current_individual += 1

            # 특정 파일을 통한 리플레이 중일 경우
            if args.replay_file:
                if not args.no_display:
                    # 진행 중인 개별학습임을 스크린에 반영할 것
                    # 개별학습을 더 진행할 필요가 없을 경우 exit()
                    if self._current_individual >= len(args.replay_inds):
                        if args.debug:
                            print(f'{len(args.replay_inds)} 기록 리플레이를 종료했습니다.')
                        sys.exit()

                    txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
                    self.info_window.generation.setText(txt)
            else:
                # 다음 세대 체크
                if (self.current_generation > self._true_zero_gen and self._current_individual == self._next_gen_size) or\
                    (self.current_generation == self._true_zero_gen and self._current_individual == self.config.Selection.num_parents):
                    self.next_generation()
                else:
                    if self.current_generation == self._true_zero_gen:
                        current_pop = self.config.Selection.num_parents
                    else:
                        current_pop = self._next_gen_size
                    if not args.no_display:
                        self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, current_pop))
            
            if args.no_display:
                self.env.reset()
            else:
                self.game_window.screen = self.env.reset()
            
            self.mario = self.population.individuals[self._current_individual]

            if not args.no_display:
                self.viz.mario = self.mario
        
# 명령어 실행 시 지정할 인자 정의 및 정리
def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros AI')

    # 설정 
    parser.add_argument('-c', '--config', dest='config', required=False, help='사용할 config 파일을 지정합니다.')
    # 로딩용 인자
    parser.add_argument('--load-file', dest='load_file', required=False, help='재개할 개별학습 파일의 위치를 지정합니다.')
    parser.add_argument('--load-inds', dest='load_inds', required=False, help='재개할 개별학습 파일에서 학습 과정의 번호를 지정합니다.')
    # 실시간 상황 표시 
    parser.add_argument('--no-display', dest='no_display', required=False, default=False, action='store_true', help='지정되어 있다면, 학습 과정이 그래픽으로 표기되지 않습니다.')
    # 디버그 
    parser.add_argument('--debug', dest='debug', required=False, default=False, action='store_true', help='지정되어 있다면 디버그 메뉴를 함께 표시합니다.')
    # 리플레이 인자
    parser.add_argument('--replay-file', dest='replay_file', required=False, default=None, help='리플레이 파일의 주소를 지정합니다.')
    parser.add_argument('--replay-inds', dest='replay_inds', required=False, default=None, help='리플레이 파일에서 재생할 개별학습 번호를 지정합니다.')

    args = parser.parse_args()
    
    load_from_file = bool(args.load_file) and bool(args.load_inds)
    replay_from_file = bool(args.replay_file) and bool(args.replay_inds)

    # 파일 체크로부터 불러온다 
    if bool(args.load_file) ^ bool(args.load_inds):
        parser.error('--load-file 인자 및 --load-inds 인자는 같이 쓰여야 합니다.')
    if load_from_file:
        # 'load_inds'를 리스트로 변환 
        # 범위가 있다면: 
        if '[' in args.load_inds and ']' in args.load_inds:
            args.load_inds = args.load_inds.replace('[', '').replace(']', '')
            ranges = args.load_inds.split(',')
            start_idx = int(ranges[0])
            end_idx = int(ranges[1])
            args.load_inds = list(range(start_idx, end_idx + 1))
        # 그렇지 않다면 가용 개별학습 범위의 목록으로 간주 
        else:
            args.load_inds = [int(ind) for ind in args.load_inds.split(',')]

    # 파일 체크로부터 리플레이 
    if bool(args.replay_file) ^ bool(args.replay_inds):
        parser.error('--replay-file 및 --replay-inds 는 같이 쓰여야 합니다.')
    if replay_from_file:
        # 'replay_inds'를 리스트로 변환
        # 범위가 있다면: 
        if '[' in args.replay_inds and ']' in args.replay_inds:
            args.replay_inds = args.replay_inds.replace('[', '').replace(']', '')
            ranges = args.replay_inds.split(',')
            has_end_idx = bool(ranges[1])
            start_idx = int(ranges[0])
            # 종료하는 값이라면: 
            if has_end_idx:
                end_idx = int(ranges[1])
                args.replay_inds = list(range(start_idx, end_idx + 1))
            # 시작하는 값이라면:
            else:
                end_idx = start_idx
                for fname in os.listdir(args.replay_file):
                    if fname.startswith('best_ind_gen'):
                        ind_num = int(fname[len('best_ind_gen'):])
                        if ind_num > end_idx:
                            end_idx = ind_num
                args.replay_inds = list(range(start_idx, end_idx + 1))
        # 그렇지 않다면 개별학습 범위의 목록으로 간주 
        else:
            args.replay_inds = [int(ind) for ind in args.replay_inds.split(',')]

    if replay_from_file and load_from_file:
        parser.error('파일로부터 불러올 수 없습니다.')

    # 명령어가 학습을 시행하기 위해 필요한 인자를 다 갖고 있는지 체크
    if not (bool(args.config) or (load_from_file or replay_from_file)):
        parser.error('학습 시작을 위해 설정값, 그리고/또는 재개 및 리플레이 인자를 지정해 주세요.')

    return args

# 메인 함수 (명령인자 받아서 직접 호출하여 가동)
if __name__ == "__main__":
    global args
    check_details()
    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())