from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor, QBrush
import sys
from typing import List
from neural_network import *
from mario import Mario
from config import Config

class NeuralNetworkViz(QtWidgets.QWidget):
    def __init__(self, parent, mario: Mario, size, config: Config):
        super().__init__(parent)
        self.mario = mario
        self.size = size
        self.config = config
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10
        l = self.config.NeuralNetwork.hidden_layer_architecture + [6]
        self.num_neurons_in_largest_layer = max(l[1:])
        self.neuron_locations = {}
        self.tile_size = self.config.Graphics.tile_size
        self.neuron_radius = self.config.Graphics.neuron_radius

        # 인풋 레이어에 있는 모든 뉴런의 위치를 동일한 지점으로 둔다.
        # 인풋의 수가 스크린에 보이기엔 너무 많아질 수 있음: 모든 노드를 직접 보이지 않고 바운딩 박스를 마리오의 지각영역 삼아 보이는 게 효과적이라 판단함. 
        self.x_offset = 150 + 16//2*self.tile_size[0] + 5
        self.y_offset = 5 + 15*self.tile_size[1] + 5
        for nid in range(l[0]):
            t = (0, nid)
            
            self.neuron_locations[t] = (self.x_offset, self.y_offset)

        self.show()

    def show_network(self, painter: QtGui.QPainter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))
        horizontal_space = 20  # 동일 레이어 노드 사이의 간격 설정
        
        layer_nodes = self.mario.network.layer_nodes

        default_offset = self.x_offset
        h_offset = self.x_offset
        v_offset = self.y_offset + 50
        inputs = self.mario.inputs_as_array

        out = self.mario.network.feed_forward(inputs)

        active_outputs = np.where(out > 0.5)[0]
        max_n = self.size[0] // (2* self.neuron_radius + horizontal_space)
        
        # 노드 그리기
        for layer, num_nodes in enumerate(layer_nodes[1:], 1):
            h_offset = (((max_n - num_nodes)) * (2*self.neuron_radius + horizontal_space))/2
            activations = None
            if layer > 0:
                activations = self.mario.network.params['A' + str(layer)]

            for node in range(num_nodes):
                x_loc = node * (self.neuron_radius*2 + horizontal_space) + h_offset
                y_loc = v_offset
                t = (layer, node)
                if t not in self.neuron_locations:
                    self.neuron_locations[t] = (x_loc + self.neuron_radius, y_loc)
                
                painter.setBrush(QtGui.QBrush(Qt.white, Qt.NoBrush))
                # 인풋 레이어
                if layer == 0:
                    # 학습되고 있는 값이 있다면
                    if inputs[node, 0] > 0:
                        painter.setBrush(QtGui.QBrush(Qt.green))
                    else:
                        painter.setBrush(QtGui.QBrush(Qt.white))
                # 숨은 레이어 (화면 상에 드러내지 않는다)
                elif layer > 0 and layer < len(layer_nodes) - 1:
                    saturation = max(min(activations[node, 0], 1.0), 0.0)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor.fromHslF(125/239, saturation, 120/240)))
                # Output layer
                elif layer == len(layer_nodes) - 1:
                    text = ('U', 'D', 'L', 'R', 'A', 'B')[node]
                    painter.drawText(h_offset + node * (self.neuron_radius*2 + horizontal_space), v_offset + 2*self.neuron_radius + 2*self.neuron_radius, text)
                    if node in active_outputs:
                        painter.setBrush(QtGui.QBrush(Qt.green))
                    else:
                        painter.setBrush(QtGui.QBrush(Qt.white))

                painter.drawEllipse(x_loc, y_loc, self.neuron_radius*2, self.neuron_radius*2)
            v_offset += 150

        # 수평적 효과를 리셋
        h_offset = default_offset

        # 가중치 그리기
        # 각 레이어는 1부터 시작 
        for l in range(2, len(layer_nodes)):
            weights = self.mario.network.params['W' + str(l)]
            prev_nodes = weights.shape[1]
            curr_nodes = weights.shape[0]
            # 지난 레이어의 각 노드를 참조
            for prev_node in range(prev_nodes):
                # 현재의 각 노드에 매겨진 가중치 체크
                for curr_node in range(curr_nodes):
                    # 현재 긍정적 영향을 주는 노드는 파란색으로 표시
                    if weights[curr_node, prev_node] > 0:
                        painter.setPen(QtGui.QPen(Qt.blue))
                    # 현재 부정적 영향을 주는 노드는 붉은색으로 표시
                    else:
                        painter.setPen(QtGui.QPen(Qt.red))
                    # 걱 노드 위치 체크
                    start = self.neuron_locations[(l-1, prev_node)]
                    end = self.neuron_locations[(l, curr_node)]
                    # Offset 간의 연결 표시
                    painter.drawLine(start[0], start[1] + self.neuron_radius*2, end[0], end[1])
        
        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))

        x_start = 5 + 150 + (16/2 * self.tile_size[0])
        y_start = 5 + (15 * self.tile_size[1])
        x_end = x_start
        y_end = y_start + 5 + (2 * self.neuron_radius)
        painter.drawLine(x_start, y_start, x_end, y_end)

        # 연결을 나타내는 선을 더 얇게 처리
        painter.setPen(QPen(color, 1.0, Qt.SolidLine))
        for nid in range(layer_nodes[1]):
            start = self.neuron_locations[(1, nid)]
            painter.drawLine(start[0], start[1], x_end, y_end)
        