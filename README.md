#  <div align="center"> SpatioTemporal-Predictions </div>

##  <div align="center"> Predicting Future Occupancy Grids in Dynamic Environment with Spatio-Temporal Learning  </div>                      

##  <div align="center">  ![INRIA](https://www.google.com/search?q=inria+logo&client=ubuntu&hs=Lv2&sxsrf=APq-WBuQDX_cpjppX0WBfeI1DB3O7JTpbg:1650973698798&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjfsuam1LH3AhWvzYUKHeQrC0oQ_AUoAXoECAEQAw&biw=1621&bih=949&dpr=1#imgrc=OQNwxBaUIox83M)  </div>


## Motivation

Dynamically changing constraints in robotics demand the ability to learn, adapt, and reproduce tasks. The robotic workspace is sometimes unpredictable and high dimensional, limiting the scalability of Supervised and Reinforcement Learning (RL). In order to tackle these constraints, we undertake self-supervised learning (SSL) approaches for inferring and updating the system dynamics within model-based reinforcement learning algorithms. 

The project explores representation learning for adaptive control strategies in state space with a focus on inferring internal system dynamics through self-supervised learning. In particular, we are interested in combining recent progress in domain adaptation and transfer learning with the SSL framework, allowing  the agent to adapt across diverse environments and tasks.  

## Environments
We proceed by making pertubations in RL agent's internal dynamics and testing the performance of RL policies over this dynamic changes. For reference, the performance of TD3 on different perturbed agents can be seen as:
 
![ezgif-4-9a3009d65e09](https://user-images.githubusercontent.com/24546547/99555041-2a980e80-29c0-11eb-9cd2-d5f0ac69c3e2.gif)
![ezgif-4-d586b8042543](https://user-images.githubusercontent.com/24546547/99555089-35eb3a00-29c0-11eb-8ecd-24c3bb9cde36.gif)
![ezgif-4-88d258069438](https://user-images.githubusercontent.com/24546547/99555100-397ec100-29c0-11eb-9d85-36a3ca9c6b60.gif)
![ezgif-4-a96fdd903d91](https://user-images.githubusercontent.com/24546547/99555115-3daade80-29c0-11eb-80b0-5075bff435ec.gif)


### &emsp; &emsp; Hopper &emsp; &emsp; &emsp; &emsp;  &emsp; Walker &emsp; &emsp; &emsp; &emsp; &emsp; Ant &emsp; &emsp; &emsp; &emsp; &emsp; Halfcheetah

## Research hypothesis 

Representations obtained by self supervised learning improve robustness of reinforcement learning policies

![Screenshot from 2020-11-17 17-23-07 (1)](https://user-images.githubusercontent.com/24546547/99554045-1273bf80-29bf-11eb-98a8-ac5131d0eaa1.gif)


## Implications: 

Self-supervised learning based on representations assists to develop policies that are close to the true state of the system. The project explores the adaptiveness and robustness of these policies across different systems.  

## Most related work: 
### Robustness in Reinforcement Learning

<a href="https://bair.berkeley.edu/blog/2019/03/18/rl-generalization/">C. Packer et al. (2019)</a> proposed a generalization framework for assessing Deep RL algorithms.<a href="https://arxiv.org/pdf/1803.11347.pdf"> I. Clavera et al. (2019)</a> argue that unexpected perturbations and unseen situations cause the specialized RL policies to fail at test environments. Accordingly, authors proposes a framework that demonstrates a simulated agent adapting their behavior online to novel terrains, crippled body parts, and highly-dynamic environments. 

### Using Representations in Reinforcement Learning
<a href="https://arxiv.org/pdf/2004.04136.pdf">A. Srinivas et al. (2020)</a> presents Contrastive Unsupervised Representations for RL (CURL). CURL extracts high-level features from raw pixels using contrastive learning and performs off policy control on top of the extracted features. <a href="https://arxiv.org/pdf/2006.10742.pdf">A. Zhang et al. (2020)</a> demonstrate the effectiveness of learning representations at disregarding task-irrelevant information in RL. <a href="https://arxiv.org/pdf/2007.04309.pdf">N. Hansen et al. (2020)</a> explores the use of self-supervision to allow the RL policy to continue training after deployment without using any rewards.

## Experimentation pipeline

### Basic Setup (Local)

To get started, please set up your virtual environment using

``` python
virtualenv .env -p python3
source .env/bin/activate
pip install -r requirements.txt
```

### Basic Setup (Server)

On the server, we will use docker for running experiments.
When first starting, please ensure that you have a `~/local` and `~/shared` link pointing to the local and network storage, respectively. You can set the links by running

``` bash
ln -s /mnt/nvme/(id -un) ~/local
ln -s /mnt/qb/bethge/(id -un) ~/shared
```

This only has to be done *once on each node*.
Afterwards, the container can be build and started using

``` bash
make interact 
```

Building might take a while. You'll see a shell. In this shell, you can now run experiments.
For adding additional dependencies, have a look at the `Dockerfile` and adapt it as needed.

### Running experiments

Ensure you followed the steps above and either set up a python virtual environment (for local use) or a docker container (for use on the server).

Single experiments can be run using (where the first argument will be the log directory)

``` bash
❯ ['./main.py', '--environment', 'hopper', '--algorithm', 'TD3', '--max_timesteps', '10000', '--start_timesteps', '1000', '--eval_freq', '50', '--resume', 'logs/hopper/td3/200910-121753-957281654/checkpoint.pt', '--fine_tuning', '0', '--config_file', 'None', '--render', '0', '--lowlimit', '0.5', '--highlimit', '1.5', '--Wtorso', '0.037', '--Wthigh', '0.036', '--Wleg', '0.025', '--Wfoot', '0.057', '--Ltorso', '0.923', '--Lthigh', '1.555', '--Lleg', '0.609', '--Lfoot', '0.16', '--logdir', '/home/ubuntu/shared/logs/cont_agents/hopper3test_td3/200918-054157-051298797']

| Starting with arguments 


| Namespace(Lfoot=0.16, Lleg=0.609, Lthigh=1.555, Ltorso=0.923, Wfoot=0.057, Wleg=0.025, Wthigh=0.036, Wtorso=0.037, actor_lr=0.0003, algorithm='TD3', batch_size=256, config_file='/home/ubuntu/run/src/environments/pybullet_data/mjcf/created_hopperxmls/hopperhx5ox2nb.xml', critic_lr=0.0003, device='cuda', environment='hopper', eval_freq=50, evaluate=True, expl_noise=0.1, fine_tuning='0', gamma=0.99, h1_dim=400, h2_dim=300, highlimit=1.5, justname=None, logdir='/home/ubuntu/shared/logs/cont_agents/hopper3test_td3/200918-054157-051298797', lowlimit=0.5, max_timesteps=10000, noise_clip=0.5, policy_freq=2, policy_noise=0.2, render=0, resume='logs/hopper/td3/200910-121753-957281654/checkpoint.pt', start_timesteps=1000, tau_value=0.005)

| Environment <HopperBulletEnv instance>

| Algorithm <src.algorithms.TD3.TD3 object at 0x7f049e7923d0>

|The max timesteps for an episode is set to  1000

----------The previous learning will be transfered-----------

| The checkpoint is resumed from  logs/hopper/td3/200910-121753-957281654/checkpoint.pt

| Enable fine_tuning on Test environment: False 

-----------------------------------------------------
checkpoint is loaded
/home/ubuntu/run/src/environments/pybullet_data/mjcf/created_hopperxmls/hopperhx5ox2nb.xml
| Start data recorder.
| Start new game.
Total T: 1000 Episode Num: 1 Episode T: 1000 Reward: 2583.046
| Finished game with 1000 data points.
| Start new game.
Total T: 2000 Episode Num: 2 Episode T: 1000 Reward: 2577.633
| Finished game with 1000 data points.
| Start new game.
Total T: 3000 Episode Num: 3 Episode T: 1000 Reward: 2574.136
| Finished game with 1000 data points.
| Start new game.
Total T: 4000 Episode Num: 4 Episode T: 1000 Reward: 2574.383
| Finished game with 1000 data points.
| Start new game.
Total T: 4428 Episode Num: 5 Episode T: 428 Reward: 1055.148
| Finished game with 428 data points.
| Start new game.
Total T: 4506 Episode Num: 6 Episode T: 78 Reward: 142.667
| Finished game with 78 data points.
| Start new game.
Total T: 5506 Episode Num: 7 Episode T: 1000 Reward: 2596.638
| Finished game with 1000 data points.
| Start new game.
Total T: 5628 Episode Num: 8 Episode T: 122 Reward: 238.758
| Finished game with 122 data points.
| Start new game.
Total T: 5727 Episode Num: 9 Episode T: 99 Reward: 194.021
| Finished game with 99 data points.
| Start new game.
Total T: 5827 Episode Num: 10 Episode T: 100 Reward: 160.182
| Finished game with 100 data points.
| Start new game.

|The mean test reward obtained was  1584.293

|The median test reward obtained was  2563.884

|The std deviation obtained was  1147.863

|The max test reward obtained was  2599.398

|The min test reward obtained was  137.793
Saved checkpoint to /home/ubuntu/shared/logs/cont_agents/hopper3test_td3/200918-054157-051298797/checkpoint.pt
Done
```

There are various arguments that you can specify to modify the parameters:

``` bash
python main.py --help

usage: main.py [-h] --environment {hopper} --algorithm {TD3}
               [--Ltorso LTORSO] [--Wtorso WTORSO] [--Lthigh LTHIGH] [--Wthigh WTHIGH] 
               [--Lfoot LFOOT] [--Wfoot WFOOT] [--Lleg LLEG] [--Wleg WLEG]
               [--seed SEED]

Running RL Experiments

optional arguments:
  -h, --help            show this help message and exit
  --environment {hopper}, -e {hopper}
  --algorithm {TD3}, -a {TD3}
  --Ltorso LTORSO
  --Wtorso WTORSO
  --Lthigh LTHIGH
  --Wthigh WTHIGH       
```

### Hyperparameter Sweeping

We typically want to start many experiments.
Have a look at `sweep.py`.

Please use the following workflow:

1. Create a method where you specify the hyperparameters of interest.
   Have a look at `test_sweep` for reference.

2. Run the file. The output can be directly used to start a job for running the experiment.

    ``` bash
    ❯ python sweep.py 
    --environment hopper --algorithm TD3 --Ltorso 1.45 --Wtorso 0.05
    --environment hopper --algorithm TD3 --Ltorso 1.45 --Wtorso 0.05
    [...]
    ```
3. These are the arguments you can directly pass to `python main.py`. To do this efficiently and parallelize all runs, you can use:

    ``` bash
    mkdir -p logs # this is where logs will be written to
    python sweep.py | parallel --will-cite --ungroup ./main.sh logs
    ```
4. The scripts print out log directories etc; each log directory will have a `checkpoint.pt` and `stdout.log` file. [You can load the checkpoint file using torch.](https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load)

