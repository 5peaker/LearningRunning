## Project Title

Designed as a Portfolio (Last Update: September 15, 2024)
포트폴리오로서 고안됨 (최신 업데이트: 2024년 9월 15일)

## Getting Started

This Repository will demonstrate you some projects based on Python. 
이 레포지토리 안에는 Python 기반의 프로젝트가 몇 개 들어 있습니다. 

With these codes, we expect to show you our efforts and skills.
이 코드들을 통해, 우리의 노력과 기술을 선보일 수 있기를 기대합니다. 

## Prerequisites

Codes in this repository needs the following environments for correct function.
이 레포지토리 안의 코드들의 정상 동작을 위해선 다음과 같은 환경이 필요합니다. 

* Python 3.8
* gym==0.25.2
* PyQt5==5.15.11
* numpy==1.24.4

Please install correct environments via "requirements.txt" per project if needed.
"requirements.txt"를 사용하여 각 프로젝트마다 요구되는 환경을 설치해 주세요. 

Otherwise, the code might not work properly. 
이와 같지 않을 경우, 코드가 작동하지 않을 수 있습니다. 

## Installation 

1. 명령어: `cd /path/to/SuperMarioBros-AI`
2. 명령어: `pip install -r requirements.txt`
3. ROM 파일은 아래 링크에서 구할 수 있습니다:
   - 이용 규약 상, 페이지 안의 ROM을 허락 없이 배포해서는 안 됩니다. 
   - [Super Mario Bros.](https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html) 
4. 실행: `python -m retro.import "/path/to/unzipped/super mario bros. (world)"`

## Options

* `python smb_ai.py -h` 명령어를 통해 사용 가능한 옵션을 볼 수 있습니다. 
* "Settings.config" 파일이 존재하는 경우, `python smb.ai_py -c settings.config` 명령어를 통해 파일에서 지정한 스테이지를 학습합니다. config 파일은 smi_ai.py 또는 리플레이 기록이 위치한 폴더에서 개별적으로 확인합니다. 
* `python smb_ai.py --replay-file "Example\W1-1" --replay-inds 1213, 1214` 명령어를 통해 해당 기록을 재생합니다. 이 명령어는 1-1 스테이지의 1213번, 1214번 기록을 재생하는 예시입니다. *메모:* config 파일이 없더라도 기존 기록은 재생할 수 있습니다.
* `--load-file FOLDER` 명령어를 통해, 컴퓨터 크래시 등의 요소로 중단된 학습을 다시 시작할 수 있습니다. 
* `PyQt` 모듈 상의 문제로 화면 상에 학습을 표시하길 원하지 않을 때, `--no-display` 플래그를 사용할 수 있습니다.
* `--debug` 플래그를 사용한 경우 진행 중인 학습에 대해 더 많은 정보가 표시됩니다. 성능이 더 좋아지고 있는지, 어떤 방향으로 진행되는지 등을 체크할 수 있습니다. 

## Results

폴더 내의 .gif 파일은 각각 신경망의 클리어 기록을 담고 있습니다.
이 중, SMB4-1_walljump.gif 파일은 마리오가 벽타기 기술을 배우는 과정을 담고 있습니다. 
또한, 리플레이 기록과 함께 만들어지는 .csv 파일을 통해 학습 과정에서의 디테일을 확인할 수 있습니다. 다음 코드를 홝용해 주세요. 

~~~python
from mario import load_stats
import matplotlib.pyplot as plt

stats = load_stats('/path/to/stats.csv')
tracker = 'distance'
stat_type = 'max'
values = stats[tracker][stat_type]

plt.plot(range(len(values)), values)
ylabel = f'{stat_type.capitalize()} {tracker.capitalize()}' 
plt.title(f'{ylabel} vs. Generation')
plt.ylabel(ylabel)
plt.xlabel('Generation')
plt.show()
~~~

## Usage

These codes containing below aspects:
레포지토리 안의 코드들은 다음과 같은 요소를 포함합니다:

* Artificial Intelligence | 인공지능
* Deep Learning | 딥 러닝

## Citation

본 프로젝트의 코드는 [여기]를 참조하여 재현하였습니다. (https://chrispresso.github.io/AI_Learns_To_Play_SMB_Using_GA_And_NN).