# rl-demo 强化学习 demo

---

大概在 47300 轮左右，模型已经收敛，测试奖励在 200 左右

```
Episode 47280    avg length: 281         avg reward: 166         total reward: 3331.024326444193
Episode 47300    avg length: 274         avg reward: 174         total reward: 3487.327132826473
Episode 47320    avg length: 245         avg reward: 152         total reward: 3042.0886191572436
Episode 47340    avg length: 208         avg reward: 170         total reward: 3415.834436247817
Episode 47360    avg length: 214         avg reward: 209         total reward: 4198.338869905
########## Solved! ##########
```

## 环境要求

- python >= 3.12.10


## 环境准备

```bash
brew install uv
brew install swig
```

## 安装依赖

```bash
uv sync --extra default
```

## 训练

```bash
uv run -m lunar_launcher_rl.train
```

## 运行

```bash
uv run -m lunar_launcher_rl.test
```