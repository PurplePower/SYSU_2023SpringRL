助教或老师您好，我使用的 MPE 环境不是助教所给的，而是 openai 官方的，因此 reward 是求和还是
取平均会略有不同，请您查看以下说明以使其运行，或者直接查看输出的视频以评判。


项目结构如下：
SYSU_2023SpringRL/Assignment2
 |
 |--- agents
 |      |--- maddpg 
 |              |--- submission_output_dir # 是 evaluate 的输出文件位置
 |              |--- submission_model.py # 提交的单个模型文件（加载后内含3个agents）
 |--- algorithms
 |      |--- actor_critic.py # Actor 和 Critic 网络定义
 |      |--- NewMADDPG.py    # MADDPG 算法 和 DDPGAgent 类定义
 |      |--- * # legacy files
 |
 |--- utils
 |      |--- ReplayBuffer.py # 经验重放池和 优先级经验重放池的实现
 |
 |--- saves # 训练过程中模型结果保存文件夹
 |
 |--- new_learn_simple_spread.py # 训练脚本
 |--- new_evaluate.py   # 评估脚本，请使用这个脚本进行加载模型并评估


主要的项目结构及说明如上，一些未列出的文件为弃用的或无关紧要的文件。评估时，请设置 current
working directory 为 SYSU_2023SpringRL 的上一级使得路径正确，并使用 new_evaluate.py 进行评估，该
文件将报告平均 100 条轨迹平均每步奖励及平均每步耗时。保存的模型文件是单个的，在 ‘agents/maddpg’
下的.pt 文件，其可直接被加载并用于 agents 的推断。

请注意，此中使用的是 openai 官方的 MPE 环境（https://github.com/openai/multiagent-particle-envs），
默认使用连续动作（5 维度），而非离散动作。同时发现作业标准中的 Reward 是指：平均每条轨迹每步
中，所有 agents 的奖励和；但在本项目中 Reward 应当是：平均每条轨迹每步中，所有 agents 的奖励平均
值。否则，即使从输出视频中 agents 都迅速且准确地找到目标并覆盖，也同样只有最高 -12 左右的按作业
说明中的 Reward。因此，烦请您同时参考输出视频中 agents 的表现进行评判。

同时推荐您使用代码包中 new_evaluate.py 进行评估，这会输出每条轨迹每步，agents 的平均奖励、平
均每步耗时，以及评估过程中 agents 运动的样例视频，输出在 ‘agents/maddpg/submission_output_dir‘ 中。
注意设置上面所述的 current working directory 使其正常运行，并需要安装相应依赖包。也可以在原有的
run_test.py 中运行，但无法输出视频，且输出了 agents 的奖励和，其除以 3 应该与作业标准的相对应。

此中也对 multiagent particle  environment 进行了修改，以适配 pyglet 等渲染时版本不适配导致的一些错误及输出过多的问题，
一并附上这些代码。











