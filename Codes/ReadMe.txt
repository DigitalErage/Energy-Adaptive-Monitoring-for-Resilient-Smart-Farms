The code base is a part of the source codes of the paper entitled "Attack-Resistant, Energy-Adaptive Monitoring for Smart Farms: Uncertainty-Aware Deep Reinforcement Learning Approach". This paper is accepted to IEEE IoT Journal.

To reproduce the results for proposed DRL schemes, please follow these steps:
1: run "enumerate.py" to collect phase 1 pretrain data
2: run "tree2.py" to do phase 1 training
3: run "enumerate2.py" to collect phase 2 pretrain data
4: run "tree3.py" to do phase 2 training

To run other baseline schemes, including "Data Mitigation", "Greedy", and "Random", please refer to "dm.py", "greedy.py", and "randomm.py" respectively.