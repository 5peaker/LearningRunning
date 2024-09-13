from collections import namedtuple
import numpy as np
from enum import Enum, unique

@unique
# 마리오가 맵 상에서 만나게 될 에너미들
class EnemyType(Enum):
    Green_Koopa1 = 0x00
    Red_Koopa1   = 0x01
    Buzzy_Beetle = 0x02
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba      = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15

    Generic_Enemy = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

@unique
# 마리오가 맵 상에서 마주할 수 있는 개체들 (동전, 파이프, 벽돌 등)
class StaticTileType(Enum):
    Empty = 0x00
    Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top =  0x24
    Flagpole = 0x25
    Coin_Block1 = 0xC0
    Coin_Block2 = 0xC1 
    Coin = 0xC2
    Breakable_Block = 0x51

    Generic_Static_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

@unique
# 맵 상에서 움직이는 개체들 (마리오 본인, 리프트 등)
class DynamicTileType(Enum):
    Mario = 0xAA

    Static_Lift1 = 0x24
    Static_Lift2 = 0x25
    Vertical_Lift1 = 0x26
    Vertical_Lift2 = 0x27
    Horizontal_Lift = 0x28
    Falling_Static_Lift = 0x29
    Horizontal_Moving_Lift=  0x2A
    Lift1 = 0x2B
    Lift2 = 0x2C
    Vine = 0x2F
    Flagpole = 0x30
    Start_Flag = 0x31
    Jump_Spring = 0x32
    Warpzone = 0x34
    Spring1 = 0x67
    Spring2 = 0x68

    Generic_Dynamic_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

class ColorMap(Enum):
    Empty = (255, 255, 255)   # 하얀색
    Ground = (128, 43, 0)     # 갈색
    Fake = (128, 43, 0)
    Mario = (0, 0, 255)
    Goomba = (255, 0, 20)
    Top_Pipe1 = (0, 15, 21)  # 검은 녹색
    Top_Pipe2 = (0, 15, 21)  # 검은 녹색
    Bottom_Pipe1 = (5, 179, 34)  # 밝은 녹색
    Bottom_Pipe2 = (5, 179, 34)  # 밝은 녹색
    Coin_Block1 = (219, 202, 18)  # 금색
    Coin_Block2 = (219, 202, 18)  # 금색
    Breakable_Block = (79, 70, 25)  # 갈색 

    Generic_Enemy = (255, 0, 20)  # 붉은색
    Generic_Static_Tile = (128, 43, 0) 
    Generic_Dynamic_Tile = (79, 70, 25)

Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])

class Tile(object):
    __slots__ = ['type']
    def __init__(self, type: Enum):
        self.type = type

class Enemy(object):
    # 위 에너미들이 맵에서 돌아다니도록 정의
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        enemy_type = EnemyType(enemy_id)
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location

class SMB(object):
    # 스크린은 한 번에 다섯 개의 에너미 개체만 로드할 수 있습니다. (그 이상은 필요하지 않습니다)
    MAX_NUM_ENEMIES = 5
    PAGE_SIZE = 256
    NUM_BLOCKS = 8
    RESOLUTION = Shape(256, 240)
    NUM_TILES = 416  # 0x69f - 0x500 + 1
    NUM_SCREEN_PAGES = 2
    TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

    sprite = Shape(width=16, height=16)
    resolution = Shape(256, 240)
    status_bar = Shape(width=resolution.width, height=2*sprite.height)

    xbins = list(range(16, resolution.width, 16))
    ybins = list(range(16, resolution.height, 16))

    @unique
    class RAMLocations(Enum):
        # 스크린은 한 번에 다섯 개의 에너미까지만 표시될 수 있기 때문에, RAM에서 에너미의 주소는 최대 5바이트로 족함
        # Enemy_Drawn+0은 에너미 0의 존재여부를 판별하고, +1은 에너미 1을 판별하고, +2는 에너미 2를... 
        Enemy_Drawn = 0x0F
        Enemy_Type = 0x16
        Enemy_X_Position_In_Level = 0x6E
        Enemy_X_Position_On_Screen = 0x87
        Enemy_Y_Position_On_Screen = 0xCF

        Player_X_Postion_In_Level       = 0x06D
        Player_X_Position_On_Screen     = 0x086

        Player_X_Position_Screen_Offset = 0x3AD
        Player_Y_Position_Screen_Offset = 0x3B8
        Enemy_X_Position_Screen_Offset = 0x3AE

        Player_Y_Pos_On_Screen = 0xCE
        Player_Vertical_Screen_Position = 0xB5

    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # 플레이어(마리오)가 신경써야 할 것은 스크린 내에 표시된 에너미로 족하다. 나머지는 (아마도) 메모리 안에 존재한다. 
        # 그러나 메모리에 있고 스크린에 없다면, 에너미는 마리오에게 위해를 가하지 못함. <당연>
        # enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]
        enemies = []

        for enemy_num in range(cls.MAX_NUM_ENEMIES):
            enemy = ram[cls.RAMLocations.Enemy_Drawn.value + enemy_num]
            # 에너미가 존재하는가? 
            if enemy:
                # 에너미 X의 위치를 판별
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen #- ram[0x71c]
                # enemy_loc_x = ram[cls.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                
                # 에너미 Y의 위치를 판별
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # 위치 설정
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, cls.ybins)
                xbin = np.digitize(enemy_loc_x, cls.xbins)
                tile_location = Point(xbin, ybin)

                # 에너미 ID 잡아오기 
                enemy_id = ram[cls.RAMLocations.Enemy_Type.value + enemy_num]
                # 에너미 개체 생성
                e = Enemy(0x6, location, tile_location)

                enemies.append(e)

        return enemies

    @classmethod
    # 각 레벨(스테이지)별로 맵이 다르고 구조가 다르다 
    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level.value] * 256 + ram[cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value]
        return Point(mario_x, mario_y)

    @classmethod
    # 코인을 얻거나, 에너미를 격파하면 점수 획득 가능
    def get_mario_score(cls, ram: np.ndarray) -> int:
        multipllier = 10
        score = 0
        for loc in range(0x07DC, 0x07D7-1, -1):
            score += ram[loc]*multipllier
            multipllier *= 10

        return score

    @classmethod
    def get_mario_location_on_screen(cls, ram: np.ndarray):
        mario_x = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Pos_On_Screen.value] * ram[cls.RAMLocations.Player_Vertical_Screen_Position.value] + cls.sprite.height
        return Point(mario_x, mario_y)

    @classmethod
    def get_tile_type(cls, ram:np.ndarray, delta_x: int, delta_y: int, mario: Point):
        x = mario.x + delta_x
        y = mario.y + delta_y + cls.sprite.height

        # 마리오가 어떤 페이지에 있는지 판별
        page = (x // 256) % 2
        # 마리오가 페이지의 어디에 있는지 판별
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # PPU(스테이터스 바 등)는 게임 내 세계의 일부가 아니다
        if sub_page_y not in range(13): # or sub_page_x not in range(16):
            return StaticTileType.Empty.value

        addr = 0x500 + page*208 + sub_page_y*16 + sub_page_x
        return ram[addr]

    @classmethod
    def get_tile_loc(cls, x, y):
        row = np.digitize(y, cls.ybins) - 2
        col = np.digitize(x, cls.xbins)
        return (row, col)

    @classmethod
    def get_tiles(cls, ram: np.ndarray):
        tiles = {}
        row = 0
        col = 0

        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x

        enemies = cls.get_enemy_locations(ram)
        y_start = 0
        mx, my = cls.get_mario_location_in_level(ram)
        my += 16
        # 스크린 오프셋에 맞추어 mx 설정
        mx = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]

        for y_pos in range(y_start, 240, 16):
            for x_pos in range(x_start, x_start + 256, 16):
                loc = (row, col)
                tile = cls.get_tile(x_pos, y_pos, ram)
                x, y = x_pos, y_pos
                page = (x // 256) % 2
                sub_x = (x % 256) // 16
                sub_y = (y - 32) // 16                
                addr = 0x500 + page*208 + sub_y*16 + sub_x
                
                # PPU가 존재하는 구역엔 타일이 없음 (게임 내 화면이지만 게임 내 세계의 일부는 아니기 때문에)
                if row < 2:
                    tiles[loc] =  StaticTileType.Empty
                else:

                    try:
                        tiles[loc] = StaticTileType(tile)
                    except:
                        tiles[loc] = StaticTileType.Fake
                    for enemy in enemies:
                        ex = enemy.location.x
                        ey = enemy.location.y + 8
                        # 우리가 분별할 수 있는 타일은 마리오 주위의 8개뿐 (에너미가 그 안에 들어와야 대처 가능)
                        if abs(x_pos - ex) <=8 and abs(y_pos - ey) <=8:
                            tiles[loc] = EnemyType.Generic_Enemy
                # 다음 행으로
                col += 1
            # 다음 열로 
            col = 0
            row += 1

        # 맵 내에서 마리오의 위치 표기\
        mario_row, mario_col = cls.get_mario_row_col(ram)
        loc = (mario_row, mario_col)
        tiles[loc] = DynamicTileType.Mario

        return tiles

    @classmethod
    def get_mario_row_col(cls, ram):
        x, y = cls.get_mario_location_on_screen(ram)
        # PPU에 16비트 할당
        y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value] + 16
        x += 12
        col = x // 16
        row = (y - 0) // 16
        return (row, col)


    @classmethod
    def get_tile(cls, x, y, ram, group_non_zero_tiles=True):
        page = (x // 256) % 2
        sub_x = (x % 256) // 16
        sub_y = (y - 32) // 16

        if sub_y not in range(13):
            return StaticTileType.Empty.value

        # 마리오가 딛을 수 있는 타일(바닥)
        addr = 0x500 + page*208 + sub_y*16 + sub_x
        if group_non_zero_tiles:
            if ram[addr] != 0:
                return StaticTileType.Fake.value

        return ram[addr]

        
