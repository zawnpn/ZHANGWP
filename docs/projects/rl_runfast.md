# RL RunFast

Github: [zawnpn/RL_RunFast](https://github.com/zawnpn/RL_RunFast)

## 简介

一款基于DQN算法的牌类游戏AI框架 / An AI framework for card games based on DQN algorithm

- Game Environment: 跑得快 (A traditional Chinese card game, [Game Rule](https://baike.baidu.com/item/%E8%B7%91%E5%BE%97%E5%BF%AB/12998100))
- Algorithm: DQN ([Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), [Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl))

## 代码结构

```shell
RL_RunFast
├── GameEnv # 游戏环境
│   └── RunFastGame.py # 游戏的规则定义等
├── RL_framework
│   └── DQN.py # DQN算法实现
├── extra
│   ├── config.py # 程序的参数设置
│   └── utils.py # 一些额外函数
├── save # 用于保存训练进度和训练模型
├── train.py # 训练模型，并将模型保存至save/
└── test.py # 根据训练得到的模型进行测试
```

## 使用说明

首先使用`conda`或者`virtualenv`设置独立`python`环境

conda：

```shell
# for conda
conda create -n rl_runfast python=3.6
conda activate rl_runfast
```

virtualenv:

```shell
# for virtualenv
cd RL_RunFast
python3 -m venv venv
source venv/bin/activate
```

设置好基本环境后，根据你的硬件情况，选择`requirements_gpu.txt`或`requirements_cpu.txt`来安装依赖

```shell
pip install -r requirements_gpu.txt
## if you do not have GPU:
# pip install -r requirements_cpu.txt
```

至此，环境和依赖已设置好，可以直接开始训练（训练脚本会自动检测GPU/CPU设备）

```shell
python train.py
```

训练过程中，会将训练日志输出至`./runs/`目录，可通过`tensorboard`实时查看。

训练过程可随时终止，会将当前的参数和模型保存至`./save/`目录，再次执行`train.py`时，若检测到`./save/`目录存在这些参数和模型的保存文件，会加载他们并继续接着训练。若要重新开始训练，请清空`./save/`目录下的文件。

训练结束后，可以通过`test.py`来测试模型效果：

```shell
python test.py
```

他将基于你的`./save/`目录下的模型文件来测试多局游戏，并计算平均胜率。

## 程序接口说明

若需修改调整程序，请阅读下面的接口说明。

### 游戏环境

游戏环境主要部分定义在`GameEnv/RunFastGame.py`中，该文件定义了了一个`RunfastGameEnv`类，可以调用的接口有：

- `get_state(self)`：返回当前玩家的状态
- `get_cards(self)`： 返回当前玩家的手牌
- `show_cards(self)`：print当前玩家的手牌
- `set_cards(self, cards)`：通过传入一个`cards`数组，修改当前玩家的手牌
- `get_next_player(self)`：返回当前玩家的下一位接牌玩家
- `update_next(self, new_next)`：修改当前玩家的下一位接牌玩家
- `get_position(self)`：返回当前玩家在三位玩家中的位置
- `update_position(self, new_position)`：修改当前玩家的位置
- `play_cards(self, cards_toplay, test=False)`：根据传入的`cards_toplay`来执行这一action
- `search_play_methods(self, pattern)`：根据传入的pattern，来查找并返回该pattern下所有能打出的牌（关于pattern的说明：根据不同的牌型，总共定义了22种pattern，详情可参考`search_pattern`函数）

### 算法

本框架使用了DQN算法，定义于`RL_framework/DQN.py`中。根据DQN的特性，定义了`QNet`类和`DQN`类，主要的函数有：

- `store_transition(self, player, action_cards, current_state)`：每次出牌结束，存储当时的状态s、出牌a、 玩家ABC的位置，当一整局结束后，根据 ABC 的位置分配reward
- `learn(self)`：读取之前存储的batch data，然后进行Q-Learning训练
- `choose_action(self, player, EPSILON, ways_toplay=[])`：根据提供的`EPSILON`和卡池`ways_toplay`，计算不同action的Q值，并作出$\varepsilon$-greedy决策

### 额外函数

在`extra/config.py`中，定义了训练需要的超参数，以及模型和参数文件的保存路径等。

在`extra/utils.py`中，定义了一些工具函数，如发牌`divide_cards()`、计分`calculate_score(cards_left)`等等。

### 训练和测试

在`train.py`和`test.py`中，会调用环境、算法、额外函数等，进行self-play训练和测试。训练日志会保存在`./runs/`目录，可通过tensorboard实时查看，训练模型会保存至`./save/`目录，可用于继续训练，也可通过`test.py`调用来作测试，测试结果会保存在`./save/test_result.txt`。