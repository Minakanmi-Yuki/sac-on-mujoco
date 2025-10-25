## 简介
此项目实现了Soft Actor Critic算法，并在mujoco环境中进行了测试。

### 环境配置
安装Python3.10（需要安装anaconda）
```
conda create -n sac_mujoco python=3.10
```

安装依赖
```
pip install -r requirements.txt
```

### 训练
训练智能体
```
python main.py --env_name 'Ant-v5'
```

产生gif
```
python render.py --env_name 'Ant-v5' --checkpoint 300000
```

### 结果
作者未针对环境进行超参数的微调，只有部分环境下智能体的性能比较突出。结果如下：
<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Ant-v5_animation.gif" width="250">
    <figcaption>Ant-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/HalfCheetah-v5_animation.gif" width="250">
    <figcaption>HalfCheetah-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Hopper-v5_animation.gif" width="250">
    <figcaption>Hopper-v5</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Humanoid-v5_animation.gif" width="250">
    <figcaption>Humanoid-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/HumanoidStandup-v5_animation.gif" width="250">
    <figcaption>HumanoidStandup-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/InvertedDoublePendulum-v5_animation.gif" width="250">
    <figcaption>InvertedDoublePendulum-v5</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/InvertedPendulum-v5_animation.gif" width="250">
    <figcaption>InvertedPendulum-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Pusher-v5_animation.gif" width="250">
    <figcaption>Pusher-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Reacher-v5_animation.gif" width="250">
    <figcaption>Reacher-v5</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Swimmer-v5_animation.gif" width="250">
    <figcaption>Swimmer-v5</figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="./gif/Walker2d-v5_animation.gif" width="250">
    <figcaption>Walker2d-v5</figcaption>
  </figure>
</p>